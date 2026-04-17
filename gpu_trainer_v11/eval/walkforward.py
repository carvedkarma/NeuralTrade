"""
Honest 6-fold walk-forward harness for V11.

Per fold, per specialist:
    1. Build features + labels for [train_start - warmup, test_end].
    2. Compute primary-rule signals; eligible bars = (signal != 0) & valid.
    3. Three-way split (HONEST conformal):
         train  = first 60% of fold-train window
         val    = next  20%   (early-stop only)
         cal    = last  20%   (Mondrian conformal threshold)
       Purge `horizon_bars` at every boundary: train→val, val→cal, cal→test.
    4. Per-fold transfer-entropy ranking on TRAIN ONLY.
    5. Pretrain trunk on TRAIN-portion sequences only (or load pooled
       checkpoint if a future patch supplies one).
    6. Bagged finetune on train; early-stop on val; conformal-fit on cal.
    7. Score test, admit by per-regime threshold, record metrics.
    8. Adversarial-validation AUC (train-vs-test) logged but does NOT gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_trainer_v11.eval.adversarial_drift import adversarial_auc
from gpu_trainer_v11.features.compose import compute_features
from gpu_trainer_v11.labels.horizon_conditional import per_bar_barrier_mult
from gpu_trainer_v11.labels.meta_label_v11 import compute_meta_labels_v11
from gpu_trainer_v11.labels.primary_rules import primary_rule
from gpu_trainer_v11.labels.sample_weights import weights_from_label_df
from gpu_trainer_v11.models.causal_transformer import V11ModelConfig
from gpu_trainer_v11.models.conformal import MondrianCalibrator, fit_mondrian
from gpu_trainer_v11.models.datasets import build_sequences
from gpu_trainer_v11.models.finetune import finetune_bagged, predict_proba_bagged
from gpu_trainer_v11.models.pretrain import pretrain
from gpu_trainer_v11.selection.transfer_entropy import select_top_k

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    """Convert global entry_idx/exit_idx to fold-local indices."""
    out = meta_slice.copy()
    out["entry_idx"] = out["entry_idx"] - base_offset
    out["exit_idx"] = out["exit_idx"] - base_offset
    return out


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
    progress_log=print,
) -> WalkForwardReport:
    bundle = compute_features(bars, btc_bars, symbol)
    feats_full = bundle.features
    side = bundle.side_data
    bm_full = per_bar_barrier_mult(feats_full["atr_pct_bucket"])
    primary_full = primary_rule(bars, side, rule)
    meta_full = compute_meta_labels_v11(bars, primary_full, horizon_bars, bm_full)

    ts_dt = pd.to_datetime(bars["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    folds = _build_folds(bars["timestamp"].to_numpy())
    progress_log(f"folds: {len(folds)}")

    report = WalkForwardReport(rule=rule, horizon_bars=horizon_bars)
    pfs: list[float] = []
    trades_list: list[int] = []

    for fold_num, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        i_tr_a, i_tr_b = _slice_idx(ts_dt, tr_s, tr_e)
        i_te_a, i_te_b = _slice_idx(ts_dt, te_s, te_e)

        # Three-way split with horizon-purge at every boundary.
        # Total fold-train window minus a final purge before test.
        i_train_window_end = max(i_tr_a, i_tr_b - horizon_bars)  # purge train→test
        win_len = i_train_window_end - i_tr_a
        if win_len < (SEQ_LEN + 4 * horizon_bars + 50):
            report.folds.append(FoldResult(
                fold_num=fold_num, train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=0, n_val_eligible=0, n_cal_eligible=0, n_test_eligible=0,
                n_features_kept=0, notes="train window too small after purge"))
            pfs.append(0.0); trades_list.append(0); continue

        i_tr_end = i_tr_a + int(0.6 * win_len)
        i_val_a = i_tr_end + horizon_bars                # purge train→val
        i_val_end = i_tr_a + int(0.8 * win_len)
        i_cal_a = i_val_end + horizon_bars               # purge val→cal
        i_cal_end = i_train_window_end                   # already purged before test
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

        # Per-fold causal feature ranking on TRAIN portion only.
        eligible = meta_full["eligible"].to_numpy()
        train_mask_sup = np.zeros(len(bars), dtype=bool)
        train_mask_sup[i_tr_a:i_tr_end] = True
        train_mask_sup &= eligible
        if train_mask_sup.sum() < 200:
            report.folds.append(FoldResult(fold_num=fold_num,
                train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=int(train_mask_sup.sum()),
                n_val_eligible=0, n_cal_eligible=0, n_test_eligible=0,
                n_features_kept=0, notes="train too small"))
            pfs.append(0.0); trades_list.append(0); continue
        ranking = select_top_k(
            feats_full.iloc[train_mask_sup.nonzero()[0]],
            meta_full["meta_label"].to_numpy()[train_mask_sup],
            k=top_k_features,
        )
        kept_cols = [c for c, k in zip(feats_full.columns, ranking.keep_mask) if k]
        feats_sel = feats_full[kept_cols]
        progress_log(f"  features kept: {len(kept_cols)}; top10: {kept_cols[:10]}")

        feat_mat = feats_sel.to_numpy(dtype=np.float32)
        regime_per_bar = feats_full["atr_pct_bucket"].to_numpy(dtype=np.int8)

        # Pretrain on train sub-window only (strictly causal).
        train_bar_range = np.arange(i_tr_a + SEQ_LEN - 1, i_tr_end, 4)
        if len(train_bar_range) < 100:
            report.folds.append(FoldResult(fold_num=fold_num,
                train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=int(train_mask_sup.sum()),
                n_val_eligible=0, n_cal_eligible=0, n_test_eligible=0,
                n_features_kept=len(kept_cols),
                selected_feature_top10=kept_cols[:10], notes="pretrain pool too small"))
            pfs.append(0.0); trades_list.append(0); continue
        pretrain_X = np.stack([feat_mat[i - SEQ_LEN + 1: i + 1] for i in train_bar_range])
        cfg = V11ModelConfig(n_features=feat_mat.shape[1], seq_len=SEQ_LEN)
        progress_log(f"  pretrain on {len(pretrain_X)} sequences")
        trunk = pretrain(cfg, pretrain_X, epochs=pretrain_epochs, batch=128, seed=17 + fold_num)

        # Build supervised splits — REBASE entry/exit indices to fold-local for sample weights.
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

        train_seqs = build_sequences(feat_mat[i_tr_a:i_tr_end], train_meta, train_w,
                                     regime_per_bar[i_tr_a:i_tr_end], seq_len=SEQ_LEN)
        val_seqs   = build_sequences(feat_mat[i_val_a:i_val_end], val_meta, val_w,
                                     regime_per_bar[i_val_a:i_val_end], seq_len=SEQ_LEN)
        cal_seqs   = build_sequences(feat_mat[i_cal_a:i_cal_end], cal_meta, cal_w,
                                     regime_per_bar[i_cal_a:i_cal_end], seq_len=SEQ_LEN)
        test_seqs  = build_sequences(feat_mat[i_te_a:i_te_b], test_meta, test_w,
                                     regime_per_bar[i_te_a:i_te_b], seq_len=SEQ_LEN)

        progress_log(f"  sequences: train={len(train_seqs.X)} val={len(val_seqs.X)} "
                     f"cal={len(cal_seqs.X)} test={len(test_seqs.X)}")
        if len(train_seqs.X) < 200 or len(val_seqs.X) < 30 or len(cal_seqs.X) < 30 or len(test_seqs.X) < 30:
            report.folds.append(FoldResult(fold_num=fold_num,
                train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()), test_end=str(te_e.date()),
                n_train_eligible=len(train_seqs.X), n_val_eligible=len(val_seqs.X),
                n_cal_eligible=len(cal_seqs.X), n_test_eligible=len(test_seqs.X),
                n_features_kept=len(kept_cols), selected_feature_top10=kept_cols[:10],
                notes="too few eligible sequences in one of train/val/cal/test"))
            pfs.append(0.0); trades_list.append(0); continue

        # Bagged finetune — early stop on VAL (NOT cal).
        progress_log(f"  finetune (N={n_bag} bags)")
        models = finetune_bagged(
            trunk, train_seqs.X, train_seqs.y, train_seqs.w,
            val_seqs.X, val_seqs.y, val_seqs.w,
            n_bag=n_bag, base_seed=17 + fold_num,
            epochs=finetune_epochs, batch=64,
        )

        # Mondrian conformal — fit on CAL (held out from training & early stop)
        p_cal = predict_proba_bagged(models, cal_seqs.X)
        p_test = predict_proba_bagged(models, test_seqs.X)
        cal = fit_mondrian(p_cal, cal_seqs.y, cal_seqs.R, cal_seqs.regime)
        admit = cal.admit(p_test, test_seqs.regime)
        R_admit = test_seqs.R[admit]
        n_trades = int(admit.sum())
        if n_trades > 0:
            wr = float((R_admit > 0).mean())
            expR = float(R_admit.mean())
            pf = _pf(R_admit)
            mdd = _max_dd(R_admit)
        else:
            wr = expR = pf = mdd = 0.0

        adv_auc = adversarial_auc(feat_mat[i_tr_a:i_tr_end], feat_mat[i_te_a:i_te_b])

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
            n_features_kept=len(kept_cols),
            selected_feature_top10=kept_cols[:10],
            adversarial_auc=adv_auc, n_trades=n_trades, win_rate=wr,
            expectancy_R=expR, pf=pf, max_dd_R=mdd,
            cal_diag={int(k): v for k, v in cal.diag.items()},
        ))
        pfs.append(pf); trades_list.append(n_trades)

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
