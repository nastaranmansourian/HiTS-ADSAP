import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import confusion_matrix
from ecg_emotion.config import ABLATION_PRESETS, Config, DatasetSpec, DATASET_PRESETS
from ecg_emotion.data import (
    ECGDataset,
    build_intra_subject_split,
    build_loso_split,
    build_loso_split_no_calibration,
    check_no_leakage,
    load_dataset,
    remap_subject_ids,
    train_val_split,
)
from ecg_emotion.model import EmotionRec
from ecg_emotion.seeding import make_generator, seed_worker, set_seed
from ecg_emotion.train import evaluate, train_fold
from torch.utils.data import DataLoader


def apply_compat_patches() -> None:
    import torch.nn.utils
    import torch.nn.utils.parametrizations
    import transformers.utils

    torch.nn.utils.parametrizations.weight_norm = torch.nn.utils.weight_norm
    transformers.utils.is_torch_tpu_available = lambda: False


def download_pretrained(cfg: Config) -> None:
    ckpt_dir = cfg.root_dir / cfg.ckpt_dir
    if ckpt_dir.is_dir():
        return
    hf_hub_download(
        repo_id=cfg.pretrained_repo_id,
        filename=cfg.pretrained_filename,
        local_dir=str(ckpt_dir),
    )
    hf_hub_download(
        repo_id=cfg.pretrained_repo_id,
        filename=cfg.pretrained_config_filename,
        local_dir=str(ckpt_dir),
    )


def load_pretrained_model(cfg: Config):
    from fairseq_signals.models import build_model_from_checkpoint

    return build_model_from_checkpoint(checkpoint_path=str(cfg.pretrained_checkpoint_path))


def make_loader(
    dataset: ECGDataset, cfg: Config, shuffle: bool, seed_offset: int, drop_last: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=make_generator(cfg.seed + seed_offset),
        pin_memory=cfg.pin_memory,
        drop_last=drop_last,
        persistent_workers=cfg.num_workers > 0,
    )


def plot_layer_weights(layer_weights: np.ndarray, out_path: Path) -> None:
    if layer_weights is None or len(layer_weights) == 0:
        # NMoE ablated (use_nmoe=False) -- no learned layer_weights to plot
        # for this run; skip silently rather than erroring on an empty array.
        return
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(layer_weights) + 1), layer_weights, marker="o", linestyle="-",
              label="Final Layer Weights")
    plt.xlabel("Layer Depth", fontsize=12)
    plt.ylabel(r"$\alpha_i$", fontsize=12)
    plt.title("Final Learned Weights of Encoder Layers After Training", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def emotion_class_names(cfg: Config, num_emotions: int) -> List[str]:
    """Best-effort human-readable class names for confusion-matrix axis
    labels. Falls back to plain integer labels for anything other than
    the two binary targets, since "quadrant"/"valence"/"arousal" (and
    WESAD/manD's own native class sets) don't have one fixed name
    mapping here."""
    if num_emotions == 2:
        if cfg.label_target == "valence_binary":
            return ["Low valence", "High valence"]
        if cfg.label_target == "arousal_binary":
            return ["Low arousal", "High arousal"]
    return [str(i) for i in range(num_emotions)]


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, class_names: List[str],
    out_path: Path, title: str, normalize: bool = True,
) -> np.ndarray:
    """Saves a confusion-matrix PNG (row-normalized by default, with raw
    counts still annotated) and returns the raw integer confusion matrix
    so callers can also dump it to JSON."""
    n = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums

    plt.figure(figsize=(1.3 * n + 3, 1.3 * n + 3))
    plt.imshow(cm_display, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.title(title, fontsize=11)
    plt.colorbar()
    tick_marks = np.arange(n)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm_display.max() / 2.0 if cm_display.size else 0.5
    for i in range(n):
        for j in range(n):
            text = f"{cm_display[i, j]:.2f}\n(n={int(cm[i, j])})" if normalize else str(int(cm[i, j]))
            plt.text(j, i, text, ha="center", va="center", fontsize=9,
                      color="white" if cm_display[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm


def _save_fold_predictions(out_path: Path, labels: np.ndarray, preds: np.ndarray) -> None:
    """Persist raw per-window labels/preds for a single fold, so an
    aggregated confusion matrix (across LOSO subjects, or across
    intra-subject k-folds) can be rebuilt even for folds that get
    skipped via --resume in a future session."""
    np.savez(out_path, labels=labels, preds=preds)


def _aggregate_and_plot_confusion_matrix(
    npz_paths: List[Path], class_names: List[str],
    out_path_png: Path, out_path_json: Path, title: str,
) -> None:
    all_labels, all_preds = [], []
    missing = []
    for p in npz_paths:
        if not p.is_file():
            missing.append(p.name)
            continue
        data = np.load(p)
        all_labels.append(data["labels"])
        all_preds.append(data["preds"])
    if missing:
        print(f"  [confusion matrix] WARNING: missing prediction file(s) {missing} -- "
              f"excluded from the aggregated confusion matrix (those folds predate this feature "
              f"or haven't been (re)trained yet; rerun them to include them).")
    if not all_labels:
        print(f"  [confusion matrix] no predictions available yet -- skipping {out_path_png.name}")
        return
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    cm = plot_confusion_matrix(labels, preds, class_names, out_path_png, title)
    with open(out_path_json, "w") as f:
        json.dump({"class_names": class_names, "counts": cm.tolist(), "n_samples": int(cm.sum())}, f, indent=2)
    print(f"  [confusion matrix] wrote {out_path_png} ({int(cm.sum())} samples)")


def plot_ablation_comparison(per_preset: Dict[str, Dict[str, float]], out_path: Path, title: str) -> None:
    """Bar chart of mean test accuracy (+/- std) per ablation preset,
    e.g. full / no_stft / no_nmoe / no_stft_no_nmoe."""
    names = list(per_preset.keys())
    acc_means = [per_preset[n]["acc_mean"] for n in names]
    acc_stds = [per_preset[n]["acc_std"] for n in names]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(names))
    plt.bar(x, acc_means, yerr=acc_stds, capsize=5)
    plt.xticks(x, names, rotation=20)
    plt.ylabel("Test accuracy (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_config_metadata(
    cfg: Config, spec: DatasetSpec, num_emotions: int,
    model_in_channels: int, model_lead_indices: List[int],
    protocol: str = "loso",
) -> Dict:
    return {
        "protocol": protocol,
        "dataset_format": spec.format,
        "calibration_split": cfg.calibration_split,
        "loso_use_calibration": cfg.loso_use_calibration,
        "intra_subject_n_splits": cfg.intra_subject_n_splits,
        "intra_subject_fold_index": cfg.intra_subject_fold_index,
        "val_split": cfg.val_split,
        "seed": cfg.seed,
        "fs": cfg.fs,
        "window_size_s": cfg.window_size_s,
        "overlap": cfg.overlap,
        "highpass_cutoff_hz": cfg.highpass_cutoff_hz,
        "use_attn_pooling": cfg.use_attn_pooling,
        "use_stft": cfg.use_stft,
        "use_nmoe": cfg.use_nmoe,
        "num_channels": cfg.num_channels,
        "in_channels": model_in_channels,
        "lead_indices": model_lead_indices,
        "hidden_dim": cfg.hidden_dim,
        "num_emotions": num_emotions,
        "label_target": cfg.label_target,
        "use_input_conditioned_gating": cfg.use_input_conditioned_gating,
    }


def _build_loso_split(cfg: Config, signals, labels, subject_ids, held_out):
    """Single dispatch point so both LOSO paths stay consistent with
    cfg.loso_use_calibration."""
    if cfg.loso_use_calibration:
        return build_loso_split(signals, labels, subject_ids, cfg, held_out)
    return build_loso_split_no_calibration(signals, labels, subject_ids, cfg, held_out)


def run_loso_for_dataset(cfg: Config, spec: DatasetSpec, device: torch.device) -> List[Dict]:
    dataset_name = spec.name or (spec.path.stem if spec.path.is_file() else spec.path.name)
    num_emotions = spec.num_emotions if spec.num_emotions is not None else cfg.num_emotions

    loader_kwargs = {"num_channels": cfg.num_channels} if spec.format == "dreamer" else {}
    signals, labels, subject_ids_raw, fs = load_dataset(
        str(spec.path), spec.format, target_fs=cfg.fs, label_target=cfg.label_target, **loader_kwargs
    )
    assert fs == cfg.fs, f"{dataset_name}: loader returned {fs} Hz but cfg.fs={cfg.fs}."

    subject_ids, _mapping = remap_subject_ids(subject_ids_raw)
    unique_subjects = sorted(set(subject_ids.tolist()))
    num_speakers = len(unique_subjects)

    # ---- Dataset-aware channel config -----------------------------
    # DREAMER is 2-channel and uses cfg.in_channels/cfg.lead_indices as
    # tuned. WESAD and manD are single-channel ECG -- force in_channels=1,
    # lead_indices=[0] regardless of what cfg has set (which is tuned
    # for DREAMER and would be wrong here).
    model_in_channels = cfg.in_channels if spec.format == "dreamer" else 1
    model_lead_indices = cfg.lead_indices if spec.format == "dreamer" else [0]

    print(f"\n### Dataset {dataset_name} ({spec.format}): "
          f"{num_speakers} subjects, {len(signals)} clips, {num_emotions} emotion classes, "
          f"attn_pooling={'on' if cfg.use_attn_pooling else 'off'}, "
          f"use_stft={'on' if cfg.use_stft else 'off'}, use_nmoe={'on' if cfg.use_nmoe else 'off'}, "
          f"channels={model_in_channels}, lead_indices={model_lead_indices}, "
          f"calibration_split={cfg.calibration_split if cfg.loso_use_calibration else 'DISABLED (strict LOSO)'} ###")

    dataset_output_dir = cfg.output_dir / dataset_name
    if cfg.experiment_tag:
        dataset_output_dir = dataset_output_dir / cfg.experiment_tag
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_output_dir / "run_config.json", "w") as f:
        json.dump(
            run_config_metadata(cfg, spec, num_emotions, model_in_channels, model_lead_indices, protocol="loso"),
            f, indent=2,
        )

    fold_results: List[Dict] = []
    results_path = dataset_output_dir / "test_results.json"

    existing_by_subject: Dict[int, Dict] = {}
    if cfg.resume and results_path.is_file():
        with open(results_path) as f:
            prior = json.load(f)
        for r in prior.get("per_fold", []):
            existing_by_subject[r["subject"]] = r
        print(f"  [resume] found {len(existing_by_subject)} previous result(s) in {results_path}")

    def _save_results_incrementally() -> None:
        if not fold_results:
            return
        accs = [r["test_acc"] for r in fold_results]
        f1s = [r["test_f1"] for r in fold_results]
        with open(results_path, "w") as f:
            json.dump({
                "per_fold": fold_results,
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0),
            }, f, indent=2)

    for fold_num, held_out in enumerate(unique_subjects, start=1):
        fold_label = f"subject{held_out}"
        ckpt_path = dataset_output_dir / f"best_{fold_label}.pt"

        # ---- Resume check: ONLY skip if a complete test result exists.
        # No partial-checkpoint recovery -- if test_acc isn't recorded
        # for this subject, retrain it from scratch (all num_epochs),
        # overwriting any leftover checkpoint. "No result" always means
        # "train fresh", never "reuse whatever epoch happened to be on disk."
        if cfg.resume and held_out in existing_by_subject:
            print(f"\n=== [{dataset_name}] LOSO fold {fold_num}/{num_speakers} "
                  f"(held-out subject {held_out}): SKIPPED, valid result already exists ===")
            fold_results.append(existing_by_subject[held_out])
            continue

        print(f"\n=== [{dataset_name}] LOSO fold {fold_num}/{num_speakers} "
              f"(held-out subject {held_out}) -- training from scratch ===")

        X_train_full, y_train_full, sid_train_full, X_test, y_test, sid_test = _build_loso_split(
            cfg, signals, labels, subject_ids, held_out
        )
        if len(X_test) == 0:
            print(f"  skipping subject {held_out}: no held-out test windows produced")
            continue

        X_tr, y_tr, sid_tr, X_val, y_val, sid_val = train_val_split(
            X_train_full, y_train_full, sid_train_full, cfg.val_split, cfg.seed
        )

        n_leak_train_test = check_no_leakage(X_train_full, X_test)
        n_leak_val_test = check_no_leakage(X_val, X_test)
        if n_leak_train_test or n_leak_val_test:
            raise RuntimeError(
                f"Leakage detected for {dataset_name} subject {held_out}: "
                f"train/test={n_leak_train_test}, val/test={n_leak_val_test}"
            )

        train_loader = make_loader(ECGDataset(X_tr, y_tr, sid_tr), cfg, shuffle=True, seed_offset=held_out, drop_last=True)
        val_loader = make_loader(ECGDataset(X_val, y_val, sid_val), cfg, shuffle=False, seed_offset=10_000 + held_out)
        test_loader = make_loader(ECGDataset(X_test, y_test, sid_test), cfg, shuffle=False, seed_offset=20_000 + held_out)

        pretrained_fold = load_pretrained_model(cfg)
        model = EmotionRec(
            pretrained_fold, num_emotions=num_emotions, hidden_dim=cfg.hidden_dim,
            use_attn_pooling=cfg.use_attn_pooling,
            use_input_conditioned_gating=cfg.use_input_conditioned_gating,
            in_channels=model_in_channels, lead_indices=model_lead_indices,
            use_stft=cfg.use_stft, use_nmoe=cfg.use_nmoe,
        ).to(device)

        history = train_fold(
            model, train_loader, val_loader, device,
            num_epochs=cfg.num_epochs, lr=cfg.lr,
            fold_label=fold_label, output_dir=dataset_output_dir, best_metric=cfg.best_metric,
            use_amp=cfg.use_amp, early_stop_patience=cfg.early_stop_patience,
            weight_decay=cfg.weight_decay,
        )

        model.load_state_dict(torch.load(history["best_ckpt_path"], map_location=device), strict=False)
        test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)
        _save_fold_predictions(dataset_output_dir / f"preds_{fold_label}.npz", test_labels, test_preds)

        print(
            f"  subject {held_out}: best_epoch={history['best_epoch']} "
            f"(val_acc={history['best_val_acc']:.2f}%, val_f1={history['best_val_f1']:.4f}) "
            f"-> test_acc={test_acc:.2f}%, test_f1={test_f1:.4f}"
        )

        plot_layer_weights(
            history["layer_weights"][history["best_epoch"] - 1],
            dataset_output_dir / f"layer_weights_{fold_label}.png",
        )

        fold_results.append({
            "dataset": dataset_name, "subject": held_out,
            "test_acc": test_acc, "test_f1": test_f1,
            "best_epoch": history["best_epoch"],
            "best_val_acc": history["best_val_acc"], "best_val_f1": history["best_val_f1"],
        })
        _save_results_incrementally()

        model.remove_hooks()
        del model, pretrained_fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if fold_results:
        accs = [r["test_acc"] for r in fold_results]
        f1s = [r["test_f1"] for r in fold_results]
        acc_mean = float(np.mean(accs))
        f1_mean = float(np.mean(f1s))
        acc_std = float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
        f1_std = float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)
        print(f"\n[{dataset_name}] LOSO mean test accuracy: {acc_mean:.2f}% (std {acc_std:.2f})")
        print(f"[{dataset_name}] LOSO mean test F1: {f1_mean:.4f} (std {f1_std:.4f})")

    # ---- one aggregated confusion matrix across all held-out subjects --
    class_names = emotion_class_names(cfg, num_emotions)
    npz_paths = [dataset_output_dir / f"preds_subject{s}.npz" for s in unique_subjects]
    _aggregate_and_plot_confusion_matrix(
        npz_paths, class_names,
        dataset_output_dir / "confusion_matrix.png",
        dataset_output_dir / "confusion_matrix.json",
        title=f"{dataset_name} -- LOSO (use_stft={cfg.use_stft}, use_nmoe={cfg.use_nmoe})",
    )

    return fold_results


def run_intra_subject_for_dataset(cfg: Config, spec: DatasetSpec, device: torch.device) -> List[Dict]:
    dataset_name = spec.name or (spec.path.stem if spec.path.is_file() else spec.path.name)
    num_emotions = spec.num_emotions if spec.num_emotions is not None else cfg.num_emotions
    fold_label = f"intra_subject_fold{cfg.intra_subject_fold_index}"

    dataset_output_dir = cfg.output_dir / dataset_name
    if cfg.experiment_tag:
        dataset_output_dir = dataset_output_dir / cfg.experiment_tag
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    results_path = dataset_output_dir / "test_results.json"

    # ---- Resume check: only trust a COMPLETE, valid test_results.json.
    # No partial/checkpoint recovery -- if results aren't there, retrain
    # this fold from scratch.
    if cfg.resume and results_path.is_file():
        try:
            with open(results_path) as f:
                prior = json.load(f)
            prior_folds = prior.get("per_fold", [])
            if prior_folds and prior_folds[0].get("fold_index") == cfg.intra_subject_fold_index:
                print(f"\n=== [{dataset_name}] intra-subject fold {cfg.intra_subject_fold_index}: "
                      f"SKIPPED, valid test_results.json already exists ===")
                return prior_folds
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  [resume] {results_path} exists but is invalid/incomplete ({e}) -- "
                  f"retraining fold {cfg.intra_subject_fold_index} from scratch.")

    loader_kwargs = {"num_channels": cfg.num_channels} if spec.format == "dreamer" else {}
    signals, labels, subject_ids_raw, fs = load_dataset(
        str(spec.path), spec.format, target_fs=cfg.fs, label_target=cfg.label_target, **loader_kwargs
    )
    assert fs == cfg.fs, f"{dataset_name}: loader returned {fs} Hz but cfg.fs={cfg.fs}."

    subject_ids, _mapping = remap_subject_ids(subject_ids_raw)
    unique_subjects = sorted(set(subject_ids.tolist()))
    num_speakers = len(unique_subjects)

    # ---- Dataset-aware channel config (see run_loso_for_dataset) -----
    model_in_channels = cfg.in_channels if spec.format == "dreamer" else 1
    model_lead_indices = cfg.lead_indices if spec.format == "dreamer" else [0]

    print(f"\n### Dataset {dataset_name} ({spec.format}), INTRA-SUBJECT protocol: "
          f"{num_speakers} subjects, {len(signals)} clips, {num_emotions} emotion classes, "
          f"attn_pooling={'on' if cfg.use_attn_pooling else 'off'}, "
          f"use_stft={'on' if cfg.use_stft else 'off'}, use_nmoe={'on' if cfg.use_nmoe else 'off'}, "
          f"channels={model_in_channels}, lead_indices={model_lead_indices}, "
          f"n_splits={cfg.intra_subject_n_splits}, fold_index={cfg.intra_subject_fold_index} ###")

    with open(dataset_output_dir / "run_config.json", "w") as f:
        json.dump(
            run_config_metadata(cfg, spec, num_emotions, model_in_channels, model_lead_indices, protocol="intra_subject"),
            f, indent=2,
        )

    X_train_full, y_train_full, sid_train_full, X_test, y_test, sid_test = build_intra_subject_split(
        signals, labels, subject_ids, cfg
    )

    X_tr, y_tr, sid_tr, X_val, y_val, sid_val = train_val_split(
        X_train_full, y_train_full, sid_train_full, cfg.val_split, cfg.seed
    )

    n_leak_train_test = check_no_leakage(X_train_full, X_test)
    n_leak_val_test = check_no_leakage(X_val, X_test)
    if n_leak_train_test or n_leak_val_test:
        raise RuntimeError(
            f"Leakage detected for {dataset_name} (intra-subject protocol): "
            f"train/test={n_leak_train_test}, val/test={n_leak_val_test}"
        )

    train_loader = make_loader(ECGDataset(X_tr, y_tr, sid_tr), cfg, shuffle=True, seed_offset=1, drop_last=True)
    val_loader = make_loader(ECGDataset(X_val, y_val, sid_val), cfg, shuffle=False, seed_offset=2)
    test_loader = make_loader(ECGDataset(X_test, y_test, sid_test), cfg, shuffle=False, seed_offset=3)

    pretrained = load_pretrained_model(cfg)
    model = EmotionRec(
        pretrained, num_emotions=num_emotions, hidden_dim=cfg.hidden_dim,
        use_attn_pooling=cfg.use_attn_pooling,
        use_input_conditioned_gating=cfg.use_input_conditioned_gating,
        in_channels=model_in_channels, lead_indices=model_lead_indices,
        use_stft=cfg.use_stft, use_nmoe=cfg.use_nmoe,
    ).to(device)

    history = train_fold(
        model, train_loader, val_loader, device,
        num_epochs=cfg.num_epochs, lr=cfg.lr,
        fold_label=fold_label,
        output_dir=dataset_output_dir, best_metric=cfg.best_metric,
        use_amp=cfg.use_amp, early_stop_patience=cfg.early_stop_patience,
        weight_decay=cfg.weight_decay,
    )

    model.load_state_dict(torch.load(history["best_ckpt_path"], map_location=device), strict=False)
    test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)
    _save_fold_predictions(dataset_output_dir / f"preds_{fold_label}.npz", test_labels, test_preds)
    print(
        f"[{dataset_name}] intra-subject fold {cfg.intra_subject_fold_index}: "
        f"best_epoch={history['best_epoch']} "
        f"(val_acc={history['best_val_acc']:.2f}%, val_f1={history['best_val_f1']:.4f}) "
        f"-> test_acc={test_acc:.2f}%, test_f1={test_f1:.4f}"
    )

    plot_layer_weights(
        history["layer_weights"][history["best_epoch"] - 1],
        dataset_output_dir / f"layer_weights_{fold_label}.png",
    )

    with open(dataset_output_dir / f"weights_{fold_label}.json", "w") as f:
        json.dump({
            "layer_weights": [w.tolist() for w in history["layer_weights"]],
        }, f, indent=2)

    class_names = emotion_class_names(cfg, num_emotions)
    cm = plot_confusion_matrix(
        test_labels, test_preds, class_names,
        dataset_output_dir / f"confusion_matrix_{fold_label}.png",
        title=f"{dataset_name} -- intra-subject fold {cfg.intra_subject_fold_index} "
              f"(use_stft={cfg.use_stft}, use_nmoe={cfg.use_nmoe})",
    )
    with open(dataset_output_dir / f"confusion_matrix_{fold_label}.json", "w") as f:
        json.dump({"class_names": class_names, "counts": cm.tolist(), "n_samples": int(cm.sum())}, f, indent=2)

    result = {
        "dataset": dataset_name, "protocol": "intra_subject",
        "fold_index": cfg.intra_subject_fold_index,
        "test_acc": test_acc, "test_f1": test_f1,
        "best_epoch": history["best_epoch"],
        "best_val_acc": history["best_val_acc"], "best_val_f1": history["best_val_f1"],
    }

    with open(dataset_output_dir / "test_results.json", "w") as f:
        json.dump({
            "per_fold": [result],
            "acc_mean": test_acc, "acc_std": 0.0,
            "f1_mean": test_f1, "f1_std": 0.0,
        }, f, indent=2)

    model.remove_hooks()
    del model, pretrained
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [result]


def run_intra_subject_kfold_for_dataset(cfg: Config, spec: DatasetSpec, device: torch.device) -> List[Dict]:
    dataset_name = spec.name or (spec.path.stem if spec.path.is_file() else spec.path.name)
    num_emotions = spec.num_emotions if spec.num_emotions is not None else cfg.num_emotions
    base_tag = cfg.experiment_tag
    original_fold_index = cfg.intra_subject_fold_index

    print(f"\n### Running FULL {cfg.intra_subject_n_splits}-fold intra-subject CV "
          f"on {dataset_name} -- {cfg.intra_subject_n_splits} separate training runs ###")

    fold_results: List[Dict] = []
    try:
        for fold_index in range(cfg.intra_subject_n_splits):
            cfg.intra_subject_fold_index = fold_index
            cfg.experiment_tag = f"{base_tag}/kfold{fold_index}" if base_tag else f"kfold{fold_index}"
            print(f"\n--- Fold {fold_index + 1}/{cfg.intra_subject_n_splits} ---")
            result = run_intra_subject_for_dataset(cfg, spec, device)
            fold_results.extend(result)
    finally:
        cfg.intra_subject_fold_index = original_fold_index
        cfg.experiment_tag = base_tag

    accs = [r["test_acc"] for r in fold_results]
    f1s = [r["test_f1"] for r in fold_results]
    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)

    print(f"\n[{dataset_name}] {cfg.intra_subject_n_splits}-fold mean test accuracy: "
          f"{acc_mean:.2f}% (std {acc_std:.2f})")
    print(f"[{dataset_name}] {cfg.intra_subject_n_splits}-fold mean test F1: "
          f"{f1_mean:.4f} (std {f1_std:.4f})")

    dataset_output_dir = cfg.output_dir / dataset_name
    if base_tag:
        dataset_output_dir = dataset_output_dir / base_tag
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_output_dir / "kfold_results.json", "w") as f:
        json.dump({
            "per_fold": fold_results,
            "acc_mean": acc_mean, "acc_std": acc_std,
            "f1_mean": f1_mean, "f1_std": f1_std,
        }, f, indent=2)

    # ---- one aggregated confusion matrix across all k folds -----------
    class_names = emotion_class_names(cfg, num_emotions)
    npz_paths = []
    for fold_index in range(cfg.intra_subject_n_splits):
        fold_tag = f"{base_tag}/kfold{fold_index}" if base_tag else f"kfold{fold_index}"
        npz_paths.append(cfg.output_dir / dataset_name / fold_tag / f"preds_intra_subject_fold{fold_index}.npz")
    _aggregate_and_plot_confusion_matrix(
        npz_paths, class_names,
        dataset_output_dir / "confusion_matrix.png",
        dataset_output_dir / "confusion_matrix.json",
        title=f"{dataset_name} -- {cfg.intra_subject_n_splits}-fold intra-subject "
              f"(use_stft={cfg.use_stft}, use_nmoe={cfg.use_nmoe})",
    )

    return fold_results


def main(cfg: Config = None, dataset: Optional[str] = None, protocol: str = "loso") -> Dict[str, List[Dict]]:
    cfg = cfg or Config()

    valid_protocols = ("loso", "intra_subject", "intra_subject_kfold")
    if protocol not in valid_protocols:
        raise ValueError(f"Unknown protocol '{protocol}'. Must be one of {valid_protocols}.")

    if dataset is not None:
        if dataset not in DATASET_PRESETS:
            raise ValueError(f"Unknown dataset preset '{dataset}'. Available: {sorted(DATASET_PRESETS.keys())}.")
        cfg.dataset_specs = list(DATASET_PRESETS[dataset])

    set_seed(cfg.seed)
    apply_compat_patches()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    else:
        print("Using device: cpu (no CUDA GPU detected -- training will be significantly slower)")

    download_pretrained(cfg)

    run_fn = {
        "loso": run_loso_for_dataset,
        "intra_subject": run_intra_subject_for_dataset,
        "intra_subject_kfold": run_intra_subject_kfold_for_dataset,
    }[protocol]

    all_results: Dict[str, List[Dict]] = {}
    for spec in cfg.dataset_specs:
        if not spec.path.exists():
            print(f"Skipping missing dataset path: {spec.path}")
            continue
        dataset_name = spec.name or (spec.path.stem if spec.path.is_file() else spec.path.name)
        all_results[dataset_name] = run_fn(cfg, spec, device)

    return all_results


def run_ablation_study(
    cfg: Config = None,
    dataset: Optional[str] = None,
    protocol: str = "loso",
    presets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run the same protocol once per architecture-ablation preset in
    config.ABLATION_PRESETS (or a chosen subset via `presets`) -- e.g.
    "full" (STFT + NMoE), "no_stft", "no_nmoe", "no_stft_no_nmoe" -- each
    under its own experiment_tag so results never collide or overwrite
    each other, then writes one comparison summary
    (<output_dir>/<tag>/ablation_comparison.json, plus a bar-chart PNG
    per dataset) with mean/std test accuracy and F1 per preset.

    Returns {preset_name: {dataset_name: [fold_result, ...]}}.
    """
    cfg = cfg or Config()
    presets = presets or list(ABLATION_PRESETS.keys())
    unknown = [p for p in presets if p not in ABLATION_PRESETS]
    if unknown:
        raise ValueError(f"Unknown ablation preset(s) {unknown}. Available: {sorted(ABLATION_PRESETS.keys())}.")

    base_tag = cfg.experiment_tag
    all_preset_results: Dict[str, Dict[str, List[Dict]]] = {}

    try:
        for preset_name in presets:
            cfg.apply_ablation_preset(preset_name)
            cfg.experiment_tag = f"{base_tag}/ablation_{preset_name}" if base_tag else f"ablation_{preset_name}"
            print(f"\n{'=' * 70}\nABLATION CELL: {preset_name}  "
                  f"(use_stft={cfg.use_stft}, use_nmoe={cfg.use_nmoe})\n{'=' * 70}")
            all_preset_results[preset_name] = main(cfg=cfg, dataset=dataset, protocol=protocol)
    finally:
        cfg.experiment_tag = base_tag

    # ---- comparison summary across presets, per dataset -------------
    comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
    for preset_name, per_dataset in all_preset_results.items():
        for dataset_name, fold_results in per_dataset.items():
            if not fold_results:
                continue
            accs = [r["test_acc"] for r in fold_results]
            f1s = [r["test_f1"] for r in fold_results]
            comparison.setdefault(dataset_name, {})[preset_name] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0),
                "n_folds": len(fold_results),
            }

    summary_dir = cfg.output_dir / (base_tag or "ablation_study")
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "ablation_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    for dataset_name, per_preset in comparison.items():
        plot_ablation_comparison(
            per_preset, summary_dir / f"ablation_comparison_{dataset_name}.png",
            title=f"Ablation study: {dataset_name}",
        )

    print(f"\nAblation study summary written to {summary_dir / 'ablation_comparison.json'}")
    return all_preset_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None, choices=sorted(DATASET_PRESETS.keys()))
    parser.add_argument("--protocol", type=str, default="loso",
                         choices=["loso", "intra_subject", "intra_subject_kfold"])
    parser.add_argument("--ablation", type=str, default=None, choices=sorted(ABLATION_PRESETS.keys()),
                         help="Run a single architecture-ablation preset instead of the cfg default "
                              "(sets use_stft/use_nmoe together).")
    parser.add_argument("--ablation-study", action="store_true",
                         help="Run every preset in --ablation-presets (default: all of them) back to "
                              "back and write a comparison summary + bar chart.")
    parser.add_argument("--ablation-presets", type=str, default=None,
                         help="Comma-separated subset of ablation presets to use with --ablation-study "
                              f"(default: all of {sorted(ABLATION_PRESETS.keys())}).")
    args = parser.parse_args()

    if args.ablation_study:
        presets = args.ablation_presets.split(",") if args.ablation_presets else None
        run_ablation_study(dataset=args.dataset, protocol=args.protocol, presets=presets)
    else:
        cfg = Config()
        if args.ablation:
            cfg.apply_ablation_preset(args.ablation)
        main(cfg=cfg, dataset=args.dataset, protocol=args.protocol)