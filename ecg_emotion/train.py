import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .model import EmotionRec


def configure_model(model: EmotionRec) -> EmotionRec:
    """Freeze the backbone and enable gradients only for active trainable heads."""
    model.train()
    model.requires_grad_(False)
    for attr in model.trainable_attrs:
        target = getattr(model, attr)
        if isinstance(target, nn.Parameter):
            target.requires_grad = True
        else:
            for p in target.parameters():
                p.requires_grad = True
    return model


def collect_params(model: EmotionRec):
    """Collect parameters belonging to the active trainable components."""
    params, names = [], []
    for attr in model.trainable_attrs:
        target = getattr(model, attr)
        if isinstance(target, nn.Parameter):
            if target.requires_grad:
                params.append(target)
                names.append(attr)
        else:
            for n, p in target.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{attr}.{n}")
    return params, names


@dataclass
class EpochStats:
    emo_loss: float = 0.0
    emo_acc: float = 0.0
    layer_weights: np.ndarray = field(default_factory=lambda: np.array([]))


def train_one_epoch(
    model: EmotionRec,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: "torch.cuda.amp.GradScaler",
    max_grad_norm: float = 0.0,
) -> EpochStats:
    configure_model(model)

    emo_loss_sum = 0.0
    emo_correct = emo_total = 0

    for inputs, emo_lbl, _ in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        emo_lbl = emo_lbl.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            (emo_logits,) = model(inputs)
            loss = criterion(emo_logits, emo_lbl)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)  # must unscale before clipping under AMP
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], max_grad_norm
            )
        scaler.step(optimizer)
        scaler.update()

        emo_loss_sum += loss.item() * emo_lbl.size(0)
        emo_correct += (emo_logits.argmax(1) == emo_lbl).sum().item()
        emo_total += emo_lbl.size(0)

    # Global fusion weights exist only when learned, non-conditioned fusion is active.
    if hasattr(model, "layer_weights"):
        layer_weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
    else:
        layer_weights = np.array([])

    return EpochStats(
        emo_loss=emo_loss_sum / max(emo_total, 1),
        emo_acc=100 * emo_correct / max(emo_total, 1),
        layer_weights=layer_weights,
    )


@torch.no_grad()
def evaluate(
    model: EmotionRec, loader: DataLoader, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (accuracy_pct, weighted_f1, labels, preds). `labels` and
    `preds` are the full concatenated integer arrays over the loader
    (evaluation order) -- callers that only want the scalars can ignore
    the last two (e.g. `acc, f1, _, _ = evaluate(...)`); callers building
    a confusion matrix use them directly (see main.py)."""
    model.eval()
    correct = total = 0
    all_labels, all_preds = [], []

    for inputs, emo_lbl, _ in loader:
        inputs, emo_lbl = inputs.to(device), emo_lbl.to(device)
        (emo_logits,) = model(inputs)
        preds = emo_logits.argmax(1)

        correct += (preds == emo_lbl).sum().item()
        total += emo_lbl.size(0)
        all_labels.extend(emo_lbl.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    acc = 100 * correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="weighted") if total else 0.0
    return acc, f1, np.array(all_labels), np.array(all_preds)


def _trainable_state_dict(model: EmotionRec) -> dict:
    """Only the small trainable heads for THIS model's ablation
    configuration (model.trainable_attrs) -- the pretrained backbone is
    never modified, so there's nothing new to save for it, and an
    ablated-away module (e.g. layer_weights when use_nmoe=False) never
    gets saved either."""
    full = model.state_dict()
    keep_prefixes = tuple(
        attr if attr == "layer_weights" else f"{attr}."
        for attr in model.trainable_attrs
    )
    return {k: v for k, v in full.items() if k.startswith(keep_prefixes)}


def train_fold(
    model: EmotionRec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    fold_label: str,
    output_dir: Path,
    best_metric: str = "val_f1",
    use_amp: bool = True,
    early_stop_patience: int = 5,
    weight_decay: float = 0.0,
    max_grad_norm: float = 0.0,
) -> Dict:

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / f"best_{fold_label}.pt"

    criterion = nn.CrossEntropyLoss()

    configure_model(model)
    params, _ = collect_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    amp_active = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_active)

    history = {
        "train_emo_loss": [], "train_emo_acc": [],
        "val_acc": [], "val_f1": [],
        "layer_weights": [],
        "best_epoch": None, "best_val_acc": None, "best_val_f1": None,
        "best_ckpt_path": str(best_ckpt_path),
        "stopped_early_at": None,
    }

    best_score = -float("inf")
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            amp_active, scaler, max_grad_norm=max_grad_norm,
        )
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)  # labels/preds unused for per-epoch val
        epoch_time = time.time() - epoch_start

        history["train_emo_loss"].append(stats.emo_loss)
        history["train_emo_acc"].append(stats.emo_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["layer_weights"].append(stats.layer_weights)

        print(
            f"[{fold_label}] epoch {epoch + 1}/{num_epochs}  "
            f"emotion: loss={stats.emo_loss:.4f} acc={stats.emo_acc:.2f}%  "
            f"val: acc={val_acc:.2f}% f1={val_f1:.4f}  ({epoch_time:.1f}s)"
        )

        score = val_f1 if best_metric == "val_f1" else val_acc
        if score > best_score:
            try:
                torch.save(_trainable_state_dict(model), best_ckpt_path)
            except RuntimeError as e:
                print(f"[{fold_label}] WARNING: failed to save checkpoint at epoch {epoch + 1} "
                      f"({e}). Keeping previous best checkpoint (epoch {history['best_epoch']}) "
                      f"on disk instead.")
            else:
                best_score = score
                epochs_since_improvement = 0
                history["best_epoch"] = epoch + 1
                history["best_val_acc"] = val_acc
                history["best_val_f1"] = val_f1
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stop_patience:
                history["stopped_early_at"] = epoch + 1
                print(f"[{fold_label}] early stopping at epoch {epoch + 1} "
                      f"(no {best_metric} improvement for {early_stop_patience} epochs; "
                      f"best was epoch {history['best_epoch']})")
                break

    return history