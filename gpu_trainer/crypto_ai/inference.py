from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import SystemConfig
from .model import MultiSymbolTransformer


@dataclass(slots=True)
class LoadedModel:
    model: MultiSymbolTransformer
    config: SystemConfig
    symbols: list[str]
    feature_names: list[str]
    normalizer_mean: np.ndarray
    normalizer_std: np.ndarray
    train_end_idx: int
    val_end_idx: int


class InferenceDataset(Dataset):
    def __init__(self, features: np.ndarray, sequence_length: int, start_idx: int, end_idx: int):
        self.features = features
        self.sequence_length = sequence_length
        self.start_idx = max(start_idx, sequence_length)
        self.end_idx = end_idx
        if self.end_idx <= self.start_idx:
            raise ValueError(
                f"Invalid inference range start={self.start_idx}, end={self.end_idx}, "
                f"sequence_length={self.sequence_length}"
            )

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        t = self.start_idx + idx
        x = self.features[t - self.sequence_length : t]
        return torch.from_numpy(x).float(), t


def load_model(checkpoint_path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = SystemConfig.from_dict(checkpoint["config"])
    symbols = list(checkpoint["symbols"])
    feature_names = list(checkpoint["feature_names"])

    model = MultiSymbolTransformer(
        num_symbols=len(symbols),
        num_features=len(feature_names),
        d_model=cfg.d_model,
        nhead_time=cfg.nhead_time,
        nhead_symbol=cfg.nhead_symbol,
        num_time_layers=cfg.num_time_layers,
        num_symbol_layers=cfg.num_symbol_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedModel(
        model=model,
        config=cfg,
        symbols=symbols,
        feature_names=feature_names,
        normalizer_mean=np.asarray(checkpoint["normalizer_mean"], dtype=np.float32),
        normalizer_std=np.asarray(checkpoint["normalizer_std"], dtype=np.float32),
        train_end_idx=int(checkpoint["train_end_idx"]),
        val_end_idx=int(checkpoint["val_end_idx"]),
    )


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / (std + 1e-8)).astype(np.float32)


def predict_range(
    loaded: LoadedModel,
    features_norm: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = InferenceDataset(
        features=features_norm,
        sequence_length=loaded.config.sequence_length,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_t: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    with torch.no_grad():
        for x, t in loader:
            x = x.to(device, non_blocking=True)
            logits, pred_ret = loaded.model(x)
            all_t.append(t.numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_returns.append(pred_ret.detach().cpu().numpy())

    idx = np.concatenate(all_t, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    pred_returns = np.concatenate(all_returns, axis=0)
    return idx, logits, pred_returns
