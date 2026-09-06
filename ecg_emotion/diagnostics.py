"""
Sanity checks for the attention pooling and NMoE fusion gradients.

Call `gradient_report(model)` right after a `loss.backward()` call during
training to see whether each active component is actually receiving
gradient -- this is exactly the check that would catch a component
silently getting zero gradient (it would show e.g. `stft_projection: 0.0`
for every epoch, rather than that sitting silent and undetected).

Ablation-aware: only reports on the components that actually exist for
the model's current use_stft/use_nmoe configuration (see model.EmotionRec
and config.ABLATION_PRESETS). A component that was deliberately ablated
away (e.g. stft_projection when use_stft=False) is simply absent from the
report rather than reported as a spurious 0.0.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


def gradient_report(model: nn.Module) -> Dict[str, float]:
    """Mean absolute gradient for each of the model's currently-active
    trainable components. Call AFTER `loss.backward()`, BEFORE
    `optimizer.zero_grad()` -- gradients are cleared by the zero_grad()
    call, so this must run in between. A component reading 0.0 when you
    expected it to be trained is exactly the silent-bug signature to
    watch for."""
    groups: Dict[str, nn.Module] = {
        "attention_shared_mlp": model.attn_module.shared_mlp,
        "emotion_classifier": model.emotion_classifier,
    }
    if hasattr(model, "stft_proj"):
        groups["stft_projection"] = model.stft_proj
    if hasattr(model, "gate_predictor"):
        groups["input_gate"] = model.gate_predictor

    report = {}
    for name, module in groups.items():
        grads = [p.grad.detach().abs().mean() for p in module.parameters() if p.grad is not None]
        report[name] = float(torch.stack(grads).mean().cpu()) if grads else 0.0

    if hasattr(model, "layer_weights"):
        report["global_layer_weights"] = (
            float(model.layer_weights.grad.detach().abs().mean().cpu())
            if model.layer_weights.grad is not None else 0.0
        )

    return report


def assert_full_framework_active(model: nn.Module) -> None:
    """Sanity-check that the FULL framework (STFT branch + NMoE fusion +
    attention pooling) is active for this model instance -- catches an
    accidental use_stft=False / use_nmoe=False / use_attn_pooling=False
    when you meant to be running the "full" ablation cell rather than one
    of the ablated ones."""
    active = model.active_components()
    assert active["attention_pooling"], "Attention pooling is disabled."
    assert active["use_stft"], "STFT branch is disabled (use_stft=False)."
    assert active["use_nmoe"], "NMoE fusion is disabled (use_nmoe=False)."
    print("Full framework (STFT + NMoE + attention pooling) is active:", active)


# Backwards-compatible alias for the old FrFT-era name.
assert_full_tfa_active = assert_full_framework_active


def print_trainable_parameters(model: nn.Module) -> None:
    """List every currently-trainable parameter and its shape -- useful
    right after configure_model() to confirm exactly what's unfrozen
    before an optimizer step (will vary by ablation configuration --
    see model.trainable_attrs)."""
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:80s} {tuple(p.shape)}")