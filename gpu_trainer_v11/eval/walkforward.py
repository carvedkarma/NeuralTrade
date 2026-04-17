"""
Honest 6-fold walk-forward harness for V11.

Per fold, per specialist:
    1. Build features + labels for [train_start - warmup, test_end].
    2. Compute primary-rule signals; eligible bars = (signal != 0) & valid.
    3. Three-way split (HONEST conformal):
         train  = first 60% of fold-train window
         val    = next  20%   (early-stop only)
         cal    = last  20%   (Mondrian conformal threshold)
       Purge `horizon_bars` at every boundary.
    4. Per-fold transfer-entropy ranking on TRAIN ONLY (logged for audit).
    5. Trunk init: load pooled pretrain checkpoint if present (locked
       contract); else pretrain inline on the train sub-window
       (smoke-test fallback only — production runs MUST use the
       pretrained checkpoint).
    6. Bagged finetune on train; early-stop on val; conformal-fit on cal.
    7. Score test, admit by per-regime threshold, record metrics.
    8. Adversarial-validation AUC (train-vs-test) logged but does NOT gate.
    9. Persist last fold's artifacts (state dicts + conformal + cols) for
       the per-symbol diversification probe.

Note on feature dim: the trunk consumes ALL 79 bundle features so that
ONE pretrained checkpoint is shared across folds. The per-fold causal
ranking is computed and logged for audit but is not used as a hard
input gate (it would require a different trunk dim per fold).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpu_trainer_v11.eval.adversarial_drift import adversarial_auc
from gpu_trainer_v11.features.compose import compute_features
from gpu_trainer_v11.labels.horizon_conditional import per_bar_barrier_mult
from gpu_trainer_v11.labels.meta_label_v11 import compute_meta_labels_v11
from gpu_trainer_v11.labels.primary_rules import primary_rule
from gpu_trainer_v11.labels.sample_weights import weights_from_label_df
from gpu_trainer_v11.models.causal_transformer import CausalTransformer, V11ModelConfig
from gpu_trainer_v11.models.conformal import MondrianCalibrator, fit_mondrian
from gpu_trainer_v11.models.datasets import build_sequences
from gpu_trainer_v11.models.finetune import finetune_bagged, predict_proba_bagged
from gpu_trainer_v11.models.pretrain import pretrain
from gpu_trainer_v11.selection.transfer_entropy import select_top_k

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"

TRAIN_MONTHS = 24
TEST_MONTHS = 6
MAX_FOLDS = 6
SEQ_LEN = 128
PASS_FOLD_PF_MIN = 1.3
PASS_FOLD_TRADES_MIN = 500
PASS_FOLD_FLOOR = 1.0


@dataclass
class FoldResult:
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_eligible: int
    n_val_eligible: int
    n_cal_eligible: int
    n_test_eligible: int
    n_features_kept: int
    selected_feature_top10: list = field(default_factory=list)
    adversarial_auc: float = float("nan")
    n_trades: int = 0
    win_rate: float = 0.0
    expectancy_R: float = 0.0
    pf: float = 0.0
    max_dd_R: float = 0.0
    cal_diag: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class WalkForwardReport:
    rule: str
    horizon_bars: int
    folds: list = field(default_factory=list)
    avg_pf: float = 0.0
    min_pf: float = 0.0
    avg_trades: float = 0.0
    passed: bool = False
    fail_reasons: list = field(default_factory=list)
    pretrain_source: str = "per_fold_fallback"
    last_fold_artifact: str | None = None


def _build_folds(ts_ms: np.ndarray) -> list:
    from dateutil.relativedelta import relativedelta
    data_start = datetime.utcfromtimestamp(ts_ms[0] / 1000)
    data_end = datetime.utcfromtimestamp(ts_ms[-1] / 1000)
    first_test = data_start + relativedelta(months=TRAIN_MONTHS)
    folds = []
    cur = first_test
    while cur < data_end and len(folds) < MAX_FOLDS:
        train_end = cur
        train_start = max(data_start, train_end - relativedelta(months=TRAIN_MONTHS))
        test_end = min(data_end, cur + relativedelta(months=TEST_MONTHS))
        if (test_end - cur).total_seconds() < 7 * 86400:
            break
        folds.append((train_start, train_end, cur, test_end))
        cur = test_end
    return folds


def _slice_idx(ts_dt: pd.Series, start: datetime, end: datetime) -> tuple[int, int]:
    a = int(ts_dt.searchsorted(pd.Timestamp(start), side="left"))
    b = int(ts_dt.searchsorted(pd.Timestamp(end), side="left"))
    return a, b


def _max_dd(R: np.ndarray) -> float:
    if R.size == 0:
        return 0.0
    eq = np.cumsum(R); peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def _pf(R: np.ndarray) -> float:
    if R.size == 0:
        return 0.0
    pos = R[R > 0].sum(); neg = -R[R < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def _rebase_meta_indices(meta_slice: pd.DataFrame, base_offset: int) -> pd.DataFrame:
    out = meta_slice.copy()
    out["entry_idx"] = out["entry_idx"] - base_offset
    out["exit_idx"] = out["exit_idx"] - base_offset
    return out


def _load_pretrained_trunk(ckpt_path: Path, expected_feature_cols: list[str]) -> CausalTransformer | None:
    if not ckpt_path.exists():
        return None
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_d = blob["cfg"]; cfg = V11ModelConfig(n_features=cfg_d["n_features"], seq_len=cfg_d["seq_len"])
    if blob.get("feature_cols") != expected_feature_cols:
        print(f"  WARN: pretrained checkpoint feature_cols differ from current bundle; "
              f"NOT loading (using fallback per-fold pretrain).")
        return None
    trunk = CausalTransformer(cfg)
    trunk.load_state_dict(blob["trunk_state_dict"])
    trunk.eval()
    return trunk


def _save_last_fold_artifacts(
    rule: str, horizon_bars: int,
    cfg: V11ModelConfig, models: list, conformal: MondrianCalibrator,
    feature_cols: list[str], fold_num: int,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"last_fold_{rule}_h{horizon_bars}.pt"
    torch.save({
        "rule": rule,
        "horizon_bars": horizon_bars,
        "fold_num": fold_num,
        "cfg": {"n_features": cfg.n_features, "seq_len": cfg.seq_len},
        "feature_cols": feature_cols,
        "model_state_dicts": [m.state_dict() for m in models],
        "conformal_thresholds": dict(conformal.thresholds),
        "conformal_diag": dict(conformal.diag),
        "saved_at": datetime.utcnow().isoformat(),
    }, out_path)
    return out_path


def run_walk_forward(
    bars: pd.DataFrame,
    btc_bars: pd.DataFrame | None,
    symbol: str,
    rule: str,
    horizon_bars: int,
    *,
    n_bag: int = 5,
    pretrain_epochs: int = 20,
    finetune_epochs: int = 30,
    top_k_features: int = 64,
    pretrain_checkpoint: Path | None = None,
    progress_log=print,
) -> WalkForwardReport:
    bundle = compute_features(bars, btc_bars, symbol)
    feats_full = bundle.features
    feature_cols = list(feats_full.columns)
    side = bundle.side_data
    bm_full = per_bar_barrier_mult(feats_full["atr_pct_bucket"])
    primary_full = primary_rule(bars, side, rule)
    meta_full = compute_meta_labels_v11(bars, primary_full, horizon_bars, bm_full)
    feat_mat = feats_full.to_numpy(dtype=np.float32)
    regime_per_bar = feats_full["atr_pct_bucket"].to_numpy(dtype=np.int8)

    # Try to load pooled pretrained trunk (locked contract path).
    pooled_trunk = None
    pretrain_source = "per_fold_fallback"
    if pretrain_checkpoint is not None:
        pooled_trunk = _load_pretrained_trunk(pretrain_checkpoint, feature_cols)
        if pooled_trunk is not None:
            pretrain_source = f"pooled_checkpoint:{pretrain_checkpoint.name}"
            progress_log(f"  loaded pooled pretrained trunk from {pretrain_checkpoint}")
        else:
            progress_log(f"  pretrained checkpoint at {pretrain_checkpoint} not usable; falling back")

    ts_dt = pd.to_datetime(bars["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    folds = _build_folds(bars["timestamp"].to_numpy())
    progress_log(f"folds: {len(folds)}  pretrain_source={pretrain_source}")

    report = WalkForwardReport(rule=rule, horizon_bars=horizon_bars,
                               pretrain_source=pretrain_source)
    pfs: list[float] = []
    trades_list: list[int] = []
    last_models: list = []
    last_conformal: MondrianCalibrator | None = None
    last_cfg: V11ModelConfig | None = None
    last_fold_num = 0

    for fold_num, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        i_tr_a, i_tr_b = _slice_idx(ts_dt, tr_s, tr_e)
        i_te_a, i_te_b = _slice_idx(ts_dt, te_s, te_e)

        i_train_window_end = max(i_tr_a, i_tr_b - horizon_bars)
        win_len = i_train_window_end - i_tr_a
        if win_len < (SEQ_LEN + 4 * horizon_bars + 50):
            report.folds.append(FoldResult(
                fold_num=fold_num, train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=0, n_val_eligible=0, n_cal_eligible=0, n_test_eligible=0,
                n_features_kept=0, notes="train window too small after purge"))
            pfs.append(0.0); trades_list.append(0); continue

        i_tr_end = i_tr_a + int(0.6 * win_len)
        i_val_a = i_tr_end + horizon_bars
        i_val_end = i_tr_a + int(0.8 * win_len)
        i_cal_a = i_val_end + horizon_bars
        i_cal_end = i_train_window_end
        if i_val_a >= i_val_end or i_cal_a >= i_cal_end:
            report.folds.append(FoldResult(
                fold_num=fold_num, train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=0, n_val_eligible=0, n_cal_eligible=0, n_test_eligible=0,
                n_features_kept=0, notes="purged sub-windows degenerate"))
            pfs.append(0.0); trades_list.append(0); continue

        progress_log(f"== fold {fold_num}: train {tr_s.date()}→{tr_e.date()}, test {te_s.date()}→{te_e.date()} ==")
        progress_log(f"  splits: train[{i_tr_a}:{i_tr_end}] val[{i_val_a}:{i_val_end}] "
                     f"cal[{i_cal_a}:{i_cal_end}] test[{i_te_a}:{i_te_b}]")

        # Per-fold causal feature ranking on TRAIN ONLY — audit log.
        eligible = meta_full["eligible"].to_numpy()
        train_mask_sup = np.zeros(len(bars), dtype=bool)
        train_mask_sup[i_tr_a:i_tr_end] = True
        train_mask_sup &= eligible
        ranking_top10: list[str] = []
        n_kept_audit = 0
        keep_mask = np.ones(feat_mat.shape[1], dtype=bool)   # default: all features
        if train_mask_sup.sum() >= 200:
            ranking = select_top_k(
                feats_full.iloc[train_mask_sup.nonzero()[0]],
                meta_full["meta_label"].to_numpy()[train_mask_sup],
                k=top_k_features,
            )
            keep_mask = np.asarray(ranking.keep_mask, dtype=bool)
            ranking_top10 = [c for c, k in zip(feats_full.columns, keep_mask) if k][:10]
            n_kept_audit = int(keep_mask.sum())
        progress_log(f"  per-fold causal selection: kept {int(keep_mask.sum())}/"
                     f"{feat_mat.shape[1]}; top10={ranking_top10}")
        # Per-fold input gate: zero out non-selected feature columns so the
        # trunk sees only the top-K ranked features (contract: top-K=64 used,
        # not just logged). The trunk dim stays 79 so the pooled pretrained
        # checkpoint loads cleanly; gating is enforced on inputs.
        feat_gate = keep_mask.astype(np.float32)[None, :]
        feat_mat_fold = feat_mat * feat_gate

        # Trunk: load pooled pretrain or fallback to per-fold pretrain on train sub-window.
        cfg = V11ModelConfig(n_features=feat_mat.shape[1], seq_len=SEQ_LEN)
        last_cfg = cfg
        if pooled_trunk is not None:
            import copy as _copy
            trunk = _copy.deepcopy(pooled_trunk)
        else:
            train_bar_range = np.arange(i_tr_a + SEQ_LEN - 1, i_tr_end, 4)
            if len(train_bar_range) < 100:
                report.folds.append(FoldResult(fold_num=fold_num,
                    train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                    test_start=str(te_s.date()), test_end=str(te_e.date()),
                    n_train_eligible=int(train_mask_sup.sum()),
                    n_val_eligible=0, n_cal_eligible=0, n_test_eligible=0,
                    n_features_kept=n_kept_audit,
                    selected_feature_top10=ranking_top10,
                    notes="pretrain pool too small (fallback path)"))
                pfs.append(0.0); trades_list.append(0); continue
            pretrain_X = np.stack([feat_mat_fold[i - SEQ_LEN + 1: i + 1] for i in train_bar_range])
            progress_log(f"  fallback pretrain on {len(pretrain_X)} sequences")
            trunk = pretrain(cfg, pretrain_X, epochs=pretrain_epochs, batch=128, seed=17 + fold_num)

        # Build supervised splits with REBASED entry/exit indices.
        train_meta_g = meta_full.iloc[i_tr_a:i_tr_end].reset_index(drop=True)
        val_meta_g   = meta_full.iloc[i_val_a:i_val_end].reset_index(drop=True)
        cal_meta_g   = meta_full.iloc[i_cal_a:i_cal_end].reset_index(drop=True)
        test_meta_g  = meta_full.iloc[i_te_a:i_te_b].reset_index(drop=True)

        train_meta = _rebase_meta_indices(train_meta_g, i_tr_a)
        val_meta   = _rebase_meta_indices(val_meta_g,   i_val_a)
        cal_meta   = _rebase_meta_indices(cal_meta_g,   i_cal_a)
        test_meta  = _rebase_meta_indices(test_meta_g,  i_te_a)

        train_w = weights_from_label_df(train_meta[train_meta["eligible"]],
                                        n_bars=i_tr_end - i_tr_a)
        val_w  = np.ones(int(val_meta["eligible"].sum()),  dtype=np.float64)
        cal_w  = np.ones(int(cal_meta["eligible"].sum()),  dtype=np.float64)
        test_w = np.ones(int(test_meta["eligible"].sum()), dtype=np.float64)

        train_seqs = build_sequences(feat_mat_fold[i_tr_a:i_tr_end], train_meta, train_w,
                                     regime_per_bar[i_tr_a:i_tr_end], seq_len=SEQ_LEN)
        val_seqs   = build_sequences(feat_mat_fold[i_val_a:i_val_end], val_meta, val_w,
                                     regime_per_bar[i_val_a:i_val_end], seq_len=SEQ_LEN)
        cal_seqs   = build_sequences(feat_mat_fold[i_cal_a:i_cal_end], cal_meta, cal_w,
                                     regime_per_bar[i_cal_a:i_cal_end], seq_len=SEQ_LEN)
        test_seqs  = build_sequences(feat_mat_fold[i_te_a:i_te_b], test_meta, test_w,
                                     regime_per_bar[i_te_a:i_te_b], seq_len=SEQ_LEN)

        progress_log(f"  sequences: train={len(train_seqs.X)} val={len(val_seqs.X)} "
                     f"cal={len(cal_seqs.X)} test={len(test_seqs.X)}")
        if len(train_seqs.X) < 200 or len(val_seqs.X) < 30 or len(cal_seqs.X) < 30 or len(test_seqs.X) < 30:
            report.folds.append(FoldResult(fold_num=fold_num,
                train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=len(train_seqs.X), n_val_eligible=len(val_seqs.X),
                n_cal_eligible=len(cal_seqs.X), n_test_eligible=len(test_seqs.X),
                n_features_kept=n_kept_audit, selected_feature_top10=ranking_top10,
                notes="too few eligible sequences in one of train/val/cal/test"))
            pfs.append(0.0); trades_list.append(0); continue

        progress_log(f"  finetune (N={n_bag} bags)")
        models = finetune_bagged(
            trunk, train_seqs.X, train_seqs.y, train_seqs.w,
            val_seqs.X, val_seqs.y, val_seqs.w,
            n_bag=n_bag, base_seed=17 + fold_num,
            epochs=finetune_epochs, batch=64,
        )

        p_cal = predict_proba_bagged(models, cal_seqs.X)
        p_test = predict_proba_bagged(models, test_seqs.X)
        cal = fit_mondrian(p_cal, cal_seqs.y, cal_seqs.R, cal_seqs.regime)
        admit = cal.admit(p_test, test_seqs.regime)
        R_admit = test_seqs.R[admit]
        n_trades = int(admit.sum())
        if n_trades > 0:
            wr = float((R_admit > 0).mean()); expR = float(R_admit.mean())
            pf = _pf(R_admit); mdd = _max_dd(R_admit)
        else:
            wr = expR = pf = mdd = 0.0

        adv_auc = adversarial_auc(feat_mat_fold[i_tr_a:i_tr_end], feat_mat_fold[i_te_a:i_te_b])

        progress_log(f"  TEST trades={n_trades} WR={wr:.3f} expR={expR:+.4f} PF={pf:.2f} "
                     f"maxDD={mdd:.2f} adv_auc={adv_auc:.3f}")

        report.folds.append(FoldResult(
            fold_num=fold_num,
            train_start=str(tr_s.date()), train_end=str(tr_e.date()),
            test_start=str(te_s.date()), test_end=str(te_e.date()),
            n_train_eligible=len(train_seqs.X),
            n_val_eligible=len(val_seqs.X),
            n_cal_eligible=len(cal_seqs.X),
            n_test_eligible=len(test_seqs.X),
            n_features_kept=n_kept_audit,
            selected_feature_top10=ranking_top10,
            adversarial_auc=adv_auc, n_trades=n_trades, win_rate=wr,
            expectancy_R=expR, pf=pf, max_dd_R=mdd,
            cal_diag={int(k): v for k, v in cal.diag.items()},
        ))
        pfs.append(pf); trades_list.append(n_trades)
        last_models = models; last_conformal = cal; last_fold_num = fold_num

    # Persist last fold's artifacts for the diversification probe.
    if last_models and last_conformal is not None and last_cfg is not None:
        artifact = _save_last_fold_artifacts(
            rule, horizon_bars, last_cfg, last_models, last_conformal,
            feature_cols, last_fold_num,
        )
        report.last_fold_artifact = str(artifact)
        progress_log(f"  persisted last-fold artifacts -> {artifact}")

    if pfs:
        report.avg_pf = float(np.mean(pfs))
        report.min_pf = float(np.min(pfs))
        report.avg_trades = float(np.mean(trades_list))
    fail = []
    if report.avg_pf < PASS_FOLD_PF_MIN:
        fail.append(f"avg PF {report.avg_pf:.2f} < {PASS_FOLD_PF_MIN}")
    if any(t < PASS_FOLD_TRADES_MIN for t in trades_list):
        fail.append(f"trades<{PASS_FOLD_TRADES_MIN} in some folds")
    if any(p < PASS_FOLD_FLOOR for p in pfs):
        fail.append(f"PF<{PASS_FOLD_FLOOR} in some folds")
    report.fail_reasons = fail
    report.passed = len(fail) == 0
    return report
