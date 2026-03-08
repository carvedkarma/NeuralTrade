"""v5_correlation.py — Cross-asset correlation tracking & smart correlation blocking.

v5.0.8: Provides RollingDailyCorr for computing rolling daily R-unit correlations
across symbols and CorrBlocker for gating correlated same-direction entries.
"""

import logging
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CorrConfig:
    enabled: bool = True
    window_days: int = 30
    threshold: float = 0.70
    same_side_only: bool = True
    log_matrix: bool = True
    min_aligned_days: int = 10
    max_block: int = 5


class RollingDailyCorr:
    """Tracks daily R per symbol and computes rolling pairwise Pearson correlation."""

    def __init__(self, symbols: List[str], window_days: int = 30,
                 min_aligned_days: int = 10):
        self.symbols = sorted(symbols)
        self.window_days = window_days
        self.min_aligned_days = min_aligned_days
        self.daily_r: Dict[str, Dict[str, float]] = {s: {} for s in self.symbols}

    def record_trade(self, symbol: str, date_str: str, r_value: float):
        if symbol not in self.daily_r:
            return
        if np.isnan(r_value):
            return
        self.daily_r[symbol][date_str] = self.daily_r[symbol].get(date_str, 0.0) + r_value

    def get_aligned_daily_series(self, sym_a: str, sym_b: str,
                                 max_days: Optional[int] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        dates_a = set(self.daily_r.get(sym_a, {}).keys())
        dates_b = set(self.daily_r.get(sym_b, {}).keys())
        common = sorted(dates_a & dates_b)
        if max_days is not None and len(common) > max_days:
            common = common[-max_days:]
        if len(common) == 0:
            return np.array([]), np.array([])
        series_a = np.array([self.daily_r[sym_a][d] for d in common])
        series_b = np.array([self.daily_r[sym_b][d] for d in common])
        return series_a, series_b

    def pairwise_corr(self, sym_a: str, sym_b: str,
                      max_days: Optional[int] = None) -> Optional[float]:
        sa, sb = self.get_aligned_daily_series(sym_a, sym_b, max_days)
        if len(sa) < self.min_aligned_days:
            return None
        std_a, std_b = np.std(sa), np.std(sb)
        if std_a < 1e-12 or std_b < 1e-12:
            return 0.0
        corr = float(np.corrcoef(sa, sb)[0, 1])
        if np.isnan(corr):
            return 0.0
        return corr

    def correlation_matrix(self, max_days: Optional[int] = None
                           ) -> Tuple[np.ndarray, List[str]]:
        n = len(self.symbols)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                c = self.pairwise_corr(self.symbols[i], self.symbols[j], max_days)
                if c is not None:
                    mat[i, j] = c
                    mat[j, i] = c
        return mat, self.symbols

    def get_daily_r_compact(self) -> Dict[str, Dict[str, float]]:
        return {s: dict(sorted(d.items())) for s, d in self.daily_r.items()}


class CorrBlocker:
    """Gates new entries when cross-asset rolling correlation is high."""

    def __init__(self, corr_tracker: RollingDailyCorr, config: CorrConfig):
        self.tracker = corr_tracker
        self.config = config
        self.blocked_count = 0
        self.block_log: List[Dict] = []
        self._blocks_by_position: Dict[str, int] = defaultdict(int)

    def on_position_closed(self, symbol: str):
        self._blocks_by_position.pop(symbol, None)

    def should_block(self, symbol: str, side: int,
                     open_positions: Dict[str, int]) -> bool:
        if not self.config.enabled:
            return False
        if len(self.tracker.symbols) < 2:
            return False
        if not open_positions:
            return False

        for other_sym, other_side in open_positions.items():
            if other_sym == symbol:
                continue
            if self.config.same_side_only and side != other_side:
                continue
            corr = self.tracker.pairwise_corr(
                symbol, other_sym, self.config.window_days
            )
            if corr is None:
                continue
            if abs(corr) >= self.config.threshold:
                if self.config.max_block > 0 and self._blocks_by_position[other_sym] >= self.config.max_block:
                    continue
                self.blocked_count += 1
                self._blocks_by_position[other_sym] += 1
                entry = {
                    'symbol': symbol, 'side': side,
                    'blocked_by': other_sym, 'other_side': other_side,
                    'corr': round(corr, 4), 'thresh': self.config.threshold,
                    'blocker_count': self._blocks_by_position[other_sym],
                    'blocker_cap': self.config.max_block,
                }
                self.block_log.append(entry)
                log.info("[V5_CORR_BLOCK] blocked %s side=%+d due to corr(%s,%s)=%.3f >= %.2f (%d/%d blocks by %s)",
                         symbol, side, symbol, other_sym, corr, self.config.threshold,
                         self._blocks_by_position[other_sym], self.config.max_block, other_sym)
                return True
        return False


def compute_overlap_ratio(trade_bars: Dict[str, List[Tuple[int, int]]],
                          total_bars: int) -> float:
    if total_bars <= 0 or len(trade_bars) < 2:
        return 0.0

    occupied = np.zeros(total_bars, dtype=np.int32)
    for sym, spans in trade_bars.items():
        for start, end in spans:
            s = max(0, start)
            e = min(total_bars, end)
            if s < e:
                occupied[s:e] += 1

    overlap_bars = int(np.sum(occupied >= 2))
    return overlap_bars / total_bars


def build_fold_corr_report(fold_id: int, window_train: str, window_test: str,
                           corr_tracker: RollingDailyCorr,
                           overlap_ratio: float,
                           blocker: Optional[CorrBlocker] = None) -> Dict:
    mat, syms = corr_tracker.correlation_matrix(max_days=corr_tracker.window_days)

    upper = []
    max_abs = 0.0
    max_pair = ("", "")
    n = len(syms)
    for i in range(n):
        for j in range(i + 1, n):
            v = mat[i, j]
            if not np.isnan(v):
                upper.append(abs(v))
                if abs(v) > max_abs:
                    max_abs = abs(v)
                    max_pair = (syms[i], syms[j])

    mean_abs = float(np.mean(upper)) if upper else 0.0

    report = {
        'fold_id': fold_id,
        'window_train': window_train,
        'window_test': window_test,
        'symbols': syms,
        'corr_matrix': [[round(float(v), 4) if not np.isnan(v) else None
                          for v in row] for row in mat],
        'mean_abs_corr': round(mean_abs, 4),
        'max_abs_corr': round(max_abs, 4),
        'max_abs_corr_pair': list(max_pair),
        'overlap_ratio': round(overlap_ratio, 4),
        'daily_r_series': corr_tracker.get_daily_r_compact(),
    }
    if blocker is not None:
        report['corr_blocked_trades'] = blocker.blocked_count
        report['corr_block_log'] = blocker.block_log[:50]

    return report


def log_corr_report(report: Dict, fold_id: int):
    log.info("=" * 60)
    log.info("[V5_CORR] Fold %d — Cross-Asset Correlation Report", fold_id)
    log.info("-" * 60)

    syms = report['symbols']
    mat = report['corr_matrix']
    header = f"{'':>12}" + "".join(f"{s:>12}" for s in syms)
    log.info(header)
    for i, s in enumerate(syms):
        row_str = f"{s:>12}"
        for j in range(len(syms)):
            v = mat[i][j]
            if v is None:
                row_str += f"{'N/A':>12}"
            else:
                row_str += f"{v:>12.4f}"
        log.info(row_str)

    log.info("-" * 60)
    log.info("[V5_CORR] mean_abs_corr=%.4f  max_abs_corr=%.4f (%s/%s)",
             report['mean_abs_corr'], report['max_abs_corr'],
             report['max_abs_corr_pair'][0], report['max_abs_corr_pair'][1])
    log.info("[V5_CORR] overlap_ratio=%.4f", report['overlap_ratio'])
    if 'corr_blocked_trades' in report:
        log.info("[V5_CORR] blocked_trades=%d", report['corr_blocked_trades'])
    log.info("=" * 60)


def save_corr_report(report: Dict, fold_id: int, output_dir: str = "checkpoints"):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"v5_corr_report_fold_{fold_id}.json"
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log.info("[V5_CORR] Report saved to %s", out)
