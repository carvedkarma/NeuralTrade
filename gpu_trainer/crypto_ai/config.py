from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "MATICUSDT",
]


@dataclass(slots=True)
class SystemConfig:
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    interval: str = "1h"
    history_days: int = 540
    sequence_length: int = 96
    train_split: float = 0.70
    val_split: float = 0.15
    label_threshold: float = 0.0015

    batch_size: int = 128
    epochs: int = 35
    learning_rate: float = 3e-4
    weight_decay: float = 5e-5
    mse_loss_weight: float = 0.35
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    random_seed: int = 42

    d_model: int = 192
    nhead_time: int = 6
    nhead_symbol: int = 4
    num_time_layers: int = 3
    num_symbol_layers: int = 2
    dropout: float = 0.10

    risk_max_gross_exposure: float = 1.0
    risk_max_symbol_weight: float = 0.25
    risk_top_k: int = 3
    trading_fee_bps: float = 5.0
    slippage_bps: float = 3.0
    min_signal_strength: float = 0.03

    data_dir: Path = Path("crypto_ai_data")
    artifact_dir: Path = Path("crypto_ai_runs")
    reports_dir: Path = Path("crypto_ai_reports")

    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 2

    self_train_cycles: int = 3
    self_train_sleep_seconds: int = 120

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["artifact_dir"] = str(self.artifact_dir)
        payload["reports_dir"] = str(self.reports_dir)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SystemConfig":
        data = dict(raw)
        if "data_dir" in data:
            data["data_dir"] = Path(data["data_dir"])
        if "artifact_dir" in data:
            data["artifact_dir"] = Path(data["artifact_dir"])
        if "reports_dir" in data:
            data["reports_dir"] = Path(data["reports_dir"])
        return cls(**data)
