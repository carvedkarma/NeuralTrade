from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import SystemConfig
from .data import DataBundle
from .model import MultiSymbolTransformer


@dataclass(slots=True)
class TrainArtifacts:
    run_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    train_end_idx: int
    val_end_idx: int
    best_val_loss: float
    device: str
    symbols: list[str]
    feature_names: list[str]


class MultiSymbolSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        class_labels: np.ndarray,
        horizon_returns: np.ndarray,
        sequence_length: int,
        start_idx: int,
        end_idx: int,
    ):
        if end_idx <= start_idx:
            raise ValueError("end_idx must be greater than start_idx")
        self.features = features
        self.class_labels = class_labels
        self.horizon_returns = horizon_returns
        self.sequence_length = sequence_length
        self.start_idx = max(start_idx, sequence_length)
        self.end_idx = end_idx
        if self.end_idx - self.start_idx <= 0:
            raise ValueError(
                f"Insufficient rows ({end_idx-start_idx}) for sequence_length={sequence_length}"
            )

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.start_idx + index
        x = self.features[t - self.sequence_length : t]  # [L, S, F]
        y_cls = self.class_labels[t]  # [S]
        y_ret = self.horizon_returns[t]  # [S]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y_cls).long(),
            torch.from_numpy(y_ret).float(),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_device(config: SystemConfig) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config.device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_feature_normalizer(
    train_slice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # train_slice shape: [T, S, F]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_normalizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (features - mean) / std
    return normalized.astype(np.float32)


def _build_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    # train_labels shape [T, S]
    flat = train_labels.reshape(-1)
    counts = np.bincount(flat, minlength=3).astype(np.float32)
    total = counts.sum()
    weights = total / (3.0 * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.5, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def _epoch_step(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    class_weights: torch.Tensor,
    mse_weight: float,
    amp_enabled: bool,
    grad_clip_norm: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_mse_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y_cls, y_ret in loader:
        x = x.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)
        y_ret = y_ret.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_cm = (
            torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda")
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_cm:
            logits, pred_ret = model(x)
            cls_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_cls.view(-1),
                weight=class_weights.to(device),
            )
            mse_loss = F.huber_loss(pred_ret, y_ret, delta=0.01)
            loss = cls_loss + mse_weight * mse_loss

        if is_train:
            if scaler is not None and amp_enabled and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_cls_loss += float(cls_loss.detach().cpu().item())
        total_mse_loss += float(mse_loss.detach().cpu().item())

        preds = torch.argmax(logits.detach(), dim=-1)
        total_correct += int((preds == y_cls).sum().item())
        total_samples += int(y_cls.numel())

    n_batches = max(len(loader), 1)
    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "mse_loss": total_mse_loss / n_batches,
        "acc": float(total_correct / max(total_samples, 1)),
    }


def train_model(
    config: SystemConfig,
    bundle: DataBundle,
    run_name: str = "run",
) -> TrainArtifacts:
    _seed_everything(config.random_seed)
    config.ensure_directories()

    n_total = bundle.features.shape[0]
    train_end_idx = int(n_total * config.train_split)
    val_end_idx = int(n_total * (config.train_split + config.val_split))
    val_end_idx = min(max(val_end_idx, train_end_idx + 1), n_total - 1)

    if train_end_idx <= config.sequence_length + 16:
        raise RuntimeError(
            f"Too little training data ({train_end_idx}) for sequence_length={config.sequence_length}"
        )

    train_slice = bundle.features[:train_end_idx]
    mean, std = _fit_feature_normalizer(train_slice)
    features_norm = _apply_normalizer(bundle.features, mean=mean, std=std)

    train_ds = MultiSymbolSequenceDataset(
        features=features_norm,
        class_labels=bundle.class_labels,
        horizon_returns=bundle.horizon_returns,
        sequence_length=config.sequence_length,
        start_idx=0,
        end_idx=train_end_idx,
    )
    val_ds = MultiSymbolSequenceDataset(
        features=features_norm,
        class_labels=bundle.class_labels,
        horizon_returns=bundle.horizon_returns,
        sequence_length=config.sequence_length,
        start_idx=train_end_idx,
        end_idx=val_end_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    device = _select_device(config)
    model = MultiSymbolTransformer(
        num_symbols=len(bundle.symbols),
        num_features=len(bundle.feature_names),
        d_model=config.d_model,
        nhead_time=config.nhead_time,
        nhead_symbol=config.nhead_symbol,
        num_time_layers=config.num_time_layers,
        num_symbol_layers=config.num_symbol_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    class_weights = _build_class_weights(bundle.class_labels[:train_end_idx])
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run_dir = config.artifact_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.json"

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _epoch_step(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            mse_weight=config.mse_loss_weight,
            amp_enabled=amp_enabled,
            grad_clip_norm=config.grad_clip_norm,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics = _epoch_step(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                class_weights=class_weights,
                mse_weight=config.mse_loss_weight,
                amp_enabled=False,
                grad_clip_norm=config.grad_clip_norm,
                scaler=None,
            )
        scheduler.step(val_metrics["loss"])

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_cls_loss": val_metrics["cls_loss"],
            "val_mse_loss": val_metrics["mse_loss"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        improved = val_metrics["loss"] < (best_val_loss - 1e-6)
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "symbols": bundle.symbols,
                    "feature_names": bundle.feature_names,
                    "train_end_idx": train_end_idx,
                    "val_end_idx": val_end_idx,
                    "best_val_loss": best_val_loss,
                    "normalizer_mean": mean,
                    "normalizer_std": std,
                    "timestamp": [str(ts) for ts in bundle.timestamps],
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    payload = {
        "run_name": run_name,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_end_idx": train_end_idx,
        "val_end_idx": val_end_idx,
        "history": history,
        "config": config.to_dict(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    return TrainArtifacts(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        train_end_idx=train_end_idx,
        val_end_idx=val_end_idx,
        best_val_loss=best_val_loss,
        device=str(device),
        symbols=bundle.symbols,
        feature_names=bundle.feature_names,
    )
