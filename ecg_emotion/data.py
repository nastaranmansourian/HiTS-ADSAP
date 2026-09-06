from math import gcd
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pickle
from pathlib import Path
import scipy.io
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

from .config import Config


# --------------------------------------------------------------------- #
# Resampling -- shared by every loader so all datasets end up at the
# same sampling rate (cfg.fs) regardless of their native rate. Works on
# both 1D (T,) and multi-channel (C, T) signals via `axis`.
# --------------------------------------------------------------------- #
def resample_to(signal: np.ndarray, orig_fs: int, target_fs: int, axis: int = -1) -> np.ndarray:
    """Polyphase resampling (with automatic anti-aliasing filtering --
    important for ECG, since naive decimation can distort the QRS
    complex). No-op if orig_fs == target_fs. `axis` is the TIME axis --
    for a (C, T) multi-channel signal, leave axis=-1 (default)."""
    if orig_fs == target_fs:
        return signal
    g = gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g
    return resample_poly(signal, up, down, axis=axis)


# --------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------- #
def highpass_filter(signal: np.ndarray, cutoff: float = 0.5, fs: int = 256, axis: int = -1) -> np.ndarray:
    """`axis` is the TIME axis -- for a (C, T) multi-channel signal,
    leave axis=-1 (default) so each channel is filtered independently
    along time."""
    b, a = butter(1, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, signal, axis=axis)


# --------------------------------------------------------------------- #
# Dataset loaders (pluggable -- register new formats in DATASET_LOADERS)
# --------------------------------------------------------------------- #
def load_dreamer_signals(
    data_path: str, target_fs: int = 256, label_target: str = "quadrant",
    num_channels: int = 2,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
    """
    `num_channels`: how many of DREAMER's ECG columns to keep, 1 or 2.
    1 reproduces the original single-lead (channel index 1) behavior;
    2 keeps both leads, returning each signal as a (2, T) array instead
    of a (T,) array. Every downstream consumer (resample_to,
    highpass_filter, _window, the split builders, ECGDataset) accepts
    both shapes transparently via the `axis`/`.T` handling added for
    multi-channel support.
    """

    if label_target not in ("quadrant", "valence", "arousal", "arousal_binary", "valence_binary"):
        raise ValueError(
            f"label_target must be 'quadrant', 'valence', 'arousal', 'arousal_binary', "
            f"or 'valence_binary', got {label_target!r}"
        )
    if num_channels not in (1, 2):
        raise ValueError(f"num_channels must be 1 or 2, got {num_channels!r}")

    mat_data = scipy.io.loadmat(data_path, struct_as_record=False, squeeze_me=True)
    dreamer = mat_data["DREAMER"]

    num_participants = dreamer.noOfSubjects
    num_videos = dreamer.noOfVideoSequences
    native_fs = int(dreamer.ECG_SamplingRate)

    ecg_signals, relabeled_emotion, subject_ids = [], [], []

    for i in range(num_participants):
        participant = dreamer.Data[i]
        arousal_scores = participant.ScoreArousal
        valence_scores = participant.ScoreValence

        for clip_idx in range(num_videos):
            raw = participant.ECG.stimuli[clip_idx]  # (T, 2) -- time-major, 2 columns

            if num_channels == 1:
                stimuli_signal = raw[:, 1]  # (T,) -- original single-lead behavior
                total_len_sec = len(stimuli_signal) / native_fs
            else:
                stimuli_signal = raw[:, :2].T  # (2, T) -- channel-first
                total_len_sec = stimuli_signal.shape[-1] / native_fs

            keep_len_sec = int(min(300, (total_len_sec // 10) * 10))
            if keep_len_sec < 10:
                continue

            keep_len_samples = keep_len_sec * native_fs

            if num_channels == 1:
                segment = stimuli_signal[-keep_len_samples:]
            else:
                segment = stimuli_signal[:, -keep_len_samples:]

            v = int(round(valence_scores[clip_idx]))
            a = int(round(arousal_scores[clip_idx]))
            # DREAMER's scale is 1-5; clip defensively in case of any
            # out-of-range rounding artifacts in the source file.
            v = min(max(v, 1), 5)
            a = min(max(a, 1), 5)

            if label_target == "quadrant":
                if v in (1, 2) and a in (1, 2):
                    label = 0
                elif v in (1, 2) and a in (3, 4, 5):
                    label = 1
                elif v in (3, 4, 5) and a in (1, 2):
                    label = 2
                elif v in (3, 4, 5) and a in (3, 4, 5):
                    label = 3
                else:
                    continue
            elif label_target == "valence":
                label = v - 1  # 0..4
            elif label_target == "arousal":
                label = a - 1  # 0..4
            elif label_target == "arousal_binary":
                # raw arousal {1,2,3} -> 0 ; {4,5} -> 1
                label = 0 if a in (1, 2) else 1
            else:  # "valence_binary"
                # raw valence {1,2,3} -> 0 ; {4,5} -> 1
                label = 0 if v in (1, 2) else 1

            segment = resample_to(segment, native_fs, target_fs)
            ecg_signals.append(segment)
            relabeled_emotion.append(label)
            subject_ids.append(i + 1)

    return ecg_signals, np.array(relabeled_emotion), np.array(subject_ids), target_fs


def load_wesad_signals(
    dir_path: str, target_fs: int = 256, native_fs: int = 700, label_target: str = "quadrant",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
    """
    Load WESAD chest-ECG from a directory of per-subject `S{n}_ecg.pkl`
    files, each a dict: {"subject": str, "ecg": (N,) array, "label": (N,)
    int array}. Label codes follow the WESAD standard:
        1 = baseline, 2 = stress, 3 = amusement, 4 = meditation
    (0, 5, 6, 7 = transient/other conditions, discarded).

    Contiguous runs of the same valid label are treated as one "clip"
    (analogous to DREAMER's per-video-clip segments), then each clip is
    resampled from `native_fs` (WESAD chest-ECG: 700 Hz) to `target_fs`.

    Remapped to 4 emotion classes (0-indexed, in WESAD's own label order):
        0: baseline, 1: stress, 2: amusement, 3: meditation

    `label_target` is accepted but IGNORED here (WESAD's classes are fixed
    by its own protocol, no valence/arousal equivalent) -- present only so
    load_dataset() can call every loader with the same keyword arguments.
    WESAD is single-channel chest-ECG only, so no num_channels option here.

    Returns
    -------
    ecg_signals : list of 1D arrays (variable length), at target_fs
    relabeled_emotion : (N,) array in {0, 1, 2, 3}
    subject_ids : (N,) array (integer, parsed from the "subject" field)
    fs : target_fs
    """
    import pickle
    from pathlib import Path

    label_map = {1: 0, 2: 1, 3: 2, 4: 3}  # baseline, stress, amusement, meditation
    valid_codes = set(label_map.keys())

    ecg_signals, relabeled_emotion, subject_ids = [], [], []

    for pkl_path in sorted(Path(dir_path).glob("S*_ecg.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        ecg = np.asarray(d["ecg"]).reshape(-1)
        label = np.asarray(d["label"]).reshape(-1)
        subject_id = int(d["subject"])

        if len(ecg) != len(label):
            raise ValueError(f"{pkl_path}: ecg and label length mismatch "
                              f"({len(ecg)} vs {len(label)})")

        # Contiguous same-label runs (vectorized: find where the label
        # value changes, slice between boundaries).
        change_points = np.where(np.diff(label.astype(np.int64)) != 0)[0] + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [len(label)]))

        for s, e in zip(starts, ends):
            code = int(label[s])
            if code not in valid_codes:
                continue
            segment = ecg[s:e]
            if len(segment) < native_fs:  # shorter than 1 second, skip
                continue
            segment = resample_to(segment, native_fs, target_fs)
            ecg_signals.append(segment)
            relabeled_emotion.append(label_map[code])
            subject_ids.append(subject_id)

    return ecg_signals, np.array(relabeled_emotion), np.array(subject_ids), target_fs


def load_mand_signals(
    root: str,
    target_fs: int = 256,
    label_target: str = "emotion",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
    """
    Loads manD ECG data from the per-subject .pkl files produced by
    extract_mand_ecg.py:
        {"subject": "P1", "states": {"Anger": {"ecg": np.ndarray, "fs": 256.0}, ...}}

    Each (subject, state) pair becomes one clip -- same convention as
    load_dreamer_signals/load_wesad_signals. Not every subject has all
    5 states; whatever states were successfully extracted are used.
    """
    root = Path(root)
    pkl_files = sorted(root.glob("P*_ecg.pkl"))
    if not pkl_files:
        raise RuntimeError(f"No manD .pkl files found under {root} (expected P*_ecg.pkl)")

    label_map = {"Anger": 0, "NoEmotion": 1, "Fear": 2, "Sadness": 3, "Surprise": 4}

    signals: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[int] = []

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        subject_num = int(data["subject"][1:])  # "P1" -> 1

        for state_name, state_data in data["states"].items():
            if state_name not in label_map:
                print(f"  [warn] {data['subject']}: unknown state '{state_name}', skipping")
                continue

            ecg = np.asarray(state_data["ecg"], dtype=np.float64).reshape(-1)
            fs = state_data["fs"]
            if fs != target_fs:
                ecg = resample_to(ecg, orig_fs=int(fs), target_fs=target_fs)

            signals.append(ecg)
            labels.append(label_map[state_name])
            subject_ids.append(subject_num)

    return signals, np.array(labels), np.array(subject_ids), target_fs

DATASET_LOADERS = {
    "dreamer": load_dreamer_signals,
    "wesad": load_wesad_signals,
    "mand": load_mand_signals,
}


def load_dataset(
    path: str, fmt: str = "dreamer", target_fs: int = 256, label_target: str = "quadrant",
    **loader_kwargs,
):
    if fmt not in DATASET_LOADERS:
        raise NotImplementedError(
            f"No loader registered for format '{fmt}'. Add one to "
            f"data.DATASET_LOADERS following the (signals, labels, "
            f"subject_ids, fs) interface of load_dreamer_signals."
        )
    return DATASET_LOADERS[fmt](path, target_fs=target_fs, label_target=label_target, **loader_kwargs)


# --------------------------------------------------------------------- #
# Subject ID handling
# --------------------------------------------------------------------- #
def remap_subject_ids(subject_ids: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Map arbitrary subject IDs to contiguous 0..N-1 IDs. Run this ONCE
    per dataset, before the LOSO loop, so every fold uses the same
    mapping (and so `num_speakers` == the resulting ID range)."""
    unique_ids = sorted(set(int(s) for s in subject_ids.tolist()))
    mapping = {old: new for new, old in enumerate(unique_ids)}
    remapped = np.array([mapping[int(s)] for s in subject_ids], dtype=np.int64)
    return remapped, mapping


# --------------------------------------------------------------------- #
# Windowing -- works on BOTH 1D (T,) and multi-channel (C, T) signals.
# --------------------------------------------------------------------- #
def _window(signal: np.ndarray, segment_length: int, step: int) -> List[np.ndarray]:
    if signal.ndim == 1:
        return [
            signal[i : i + segment_length]
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
    # (C, T) -- slice the TIME axis (last axis), keep channel axis intact
    return [
        signal[:, i : i + segment_length]
        for i in range(0, signal.shape[-1] - segment_length + 1, step)
    ]


def _fit_transform_scaler(scaler: MinMaxScaler, signal: np.ndarray) -> np.ndarray:
    """Fit `scaler` on `signal` and return the transformed signal, same
    shape as input. Handles both 1D (T,) and multi-channel (C, T)
    signals -- for (C, T), each channel is scaled independently (fit
    treats channels as separate features via `.T` -> (T, C))."""
    if signal.ndim == 1:
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return scaler.fit_transform(signal.T).T


def _transform_scaler(scaler: MinMaxScaler, signal: np.ndarray) -> np.ndarray:
    """Apply an ALREADY-FIT `scaler` to `signal`. Same shape handling as
    _fit_transform_scaler."""
    if signal.ndim == 1:
        return scaler.transform(signal.reshape(-1, 1)).flatten()
    return scaler.transform(signal.T).T


# --------------------------------------------------------------------- #
# LOSO + calibration split for ONE held-out subject
# --------------------------------------------------------------------- #
def build_loso_split(
    signals: List[np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg: Config,
    held_out_subject: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    fs = cfg.fs
    segment_length = cfg.segment_length
    step = int(segment_length * (1 - cfg.overlap))

    X_train, y_train, sid_train = [], [], []
    X_test, y_test, sid_test = [], [], []

    for signal, label, subject_id in zip(signals, labels, subject_ids):
        signal = highpass_filter(signal, cutoff=cfg.highpass_cutoff_hz, fs=fs)

        if subject_id == held_out_subject:
            time_axis = -1  # works for both (T,) and (C, T)
            split_idx = int(signal.shape[time_axis] * cfg.calibration_split)
            if signal.ndim == 1:
                signal_calib, signal_holdout = signal[:split_idx], signal[split_idx:]
            else:
                signal_calib, signal_holdout = signal[:, :split_idx], signal[:, split_idx:]

            scaler = MinMaxScaler(feature_range=(-1, 1))
            norm_calib = _fit_transform_scaler(scaler, signal_calib)  # fit on calibration portion ONLY
            norm_holdout = _transform_scaler(scaler, signal_holdout)

            segs_calib = _window(norm_calib, segment_length, step)
            segs_holdout = _window(norm_holdout, segment_length, step)

            X_train.extend(segs_calib)
            y_train.extend([label] * len(segs_calib))
            sid_train.extend([subject_id] * len(segs_calib))

            X_test.extend(segs_holdout)
            y_test.extend([label] * len(segs_holdout))
            sid_test.extend([subject_id] * len(segs_holdout))
        else:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            norm = _fit_transform_scaler(scaler, signal)
            segs = _window(norm, segment_length, step)

            X_train.extend(segs)
            y_train.extend([label] * len(segs))
            sid_train.extend([subject_id] * len(segs))

    return (
        np.array(X_train), np.array(y_train), np.array(sid_train),
        np.array(X_test), np.array(y_test), np.array(sid_test),
    )


def build_loso_split_no_calibration(
    signals: List[np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg: Config,
    held_out_subject: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Strict LOSO: the held-out subject contributes ZERO data to fitting
    any normalization statistics. A single MinMaxScaler is fit on the
    POOLED training subjects' signals only, then applied as-is to the
    held-out subject's entire signal (all of it becomes test data --
    there is no per-subject calibration split).

    This is a stricter, more conservative protocol than
    build_loso_split: it tests whether the model generalizes to a
    subject's ECG using only population-level (not subject-specific)
    normalization statistics.
    """
    fs = cfg.fs
    segment_length = cfg.segment_length
    step = int(segment_length * (1 - cfg.overlap))

    # ---- Pass 1: highpass-filter everything, split by subject -------
    train_signals, train_labels_list = [], []
    test_signals, test_labels_list = [], []

    for signal, label, subject_id in zip(signals, labels, subject_ids):
        signal = highpass_filter(signal, cutoff=cfg.highpass_cutoff_hz, fs=fs)
        if subject_id == held_out_subject:
            test_signals.append(signal)
            test_labels_list.append(label)
        else:
            train_signals.append(signal)
            train_labels_list.append(label)

    if not train_signals:
        raise RuntimeError("No training subjects available to fit the pooled scaler.")

    # ---- Fit ONE scaler on pooled training signals only --------------
    concat_axis = 0 if train_signals[0].ndim == 1 else 1
    train_concat_for_fit = np.concatenate(train_signals, axis=concat_axis)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    _fit_transform_scaler(scaler, train_concat_for_fit)  # fit only; discard output here

    # ---- Apply that scaler to train subjects' own signals -----------
    X_train, y_train, sid_train = [], [], []
    for signal, label in zip(train_signals, train_labels_list):
        norm = _transform_scaler(scaler, signal)
        segs = _window(norm, segment_length, step)
        X_train.extend(segs)
        y_train.extend([label] * len(segs))
        sid_train.extend([-1] * len(segs))  # subject id not meaningful when pooled pre-fit; adjust if you track it elsewhere

    # ---- Apply the SAME scaler to the held-out subject's FULL signal -
    X_test, y_test, sid_test = [], [], []
    for signal, label in zip(test_signals, test_labels_list):
        norm = _transform_scaler(scaler, signal)  # held-out subject NEVER fits anything
        segs = _window(norm, segment_length, step)
        X_test.extend(segs)
        y_test.extend([label] * len(segs))
        sid_test.extend([held_out_subject] * len(segs))

    return (
        np.array(X_train), np.array(y_train), np.array(sid_train),
        np.array(X_test), np.array(y_test), np.array(sid_test),
    )


def build_intra_subject_split(
    signals: List[np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Intra-subject (subject-dependent) split: NO held-out subject. EVERY
    subject's own signal is divided into `cfg.intra_subject_n_splits`
    equal contiguous chunks (along the TIME axis -- works for both 1D
    and multi-channel (C, T) signals); the chunk at
    `cfg.intra_subject_fold_index` is held out as that subject's test
    portion, and every OTHER chunk is pooled into that subject's train
    portion. All subjects' train portions are pooled into one training
    set and all subjects' test portions into one test set. A single
    model is meant to be trained once on the pooled result (see
    main.run_intra_subject_for_dataset), not one model per subject.

    fold_index = n_splits - 1 reproduces the ORIGINAL fixed split exactly
    (last 1/n_splits of each subject's signal held out, everything before
    it pooled as train).

    Every test subject is therefore also present in training (on
    different portions of their own signal) -- this is the
    "subject-mixed" style evaluation, as opposed to LOSO's cross-subject
    generalization test. Leakage-safe the same way as build_loso_split:
    each chunk boundary is computed on the raw signal BEFORE windowing (so
    no window straddles a chunk boundary), and each subject's
    MinMaxScaler is fit ONLY on that subject's pooled train chunks, then
    applied to every chunk (train and test) for that subject.

    When the held-out chunk sits strictly between two train chunks (i.e.
    fold_index is neither 0 nor n_splits - 1), those two train chunks are
    windowed SEPARATELY rather than concatenated -- concatenating them
    first would create fabricated windows that splice together samples
    that were never adjacent in the original recording, once the held-out
    gap is removed.

    Returns X_train, y_train, sid_train, X_test, y_test, sid_test (each
    pooled across all subjects).
    """
    fs = cfg.fs
    segment_length = cfg.segment_length
    step = int(segment_length * (1 - cfg.overlap))
    n_splits = cfg.intra_subject_n_splits
    fold_index = cfg.intra_subject_fold_index

    X_train, y_train, sid_train = [], [], []
    X_test, y_test, sid_test = [], [], []

    for signal, label, subject_id in zip(signals, labels, subject_ids):
        signal = highpass_filter(signal, cutoff=cfg.highpass_cutoff_hz, fs=fs)

        time_len = signal.shape[-1]  # works for both (T,) and (C, T)
        boundaries = np.linspace(0, time_len, n_splits + 1).astype(int)
        test_start, test_end = boundaries[fold_index], boundaries[fold_index + 1]

        if signal.ndim == 1:
            signal_test = signal[test_start:test_end]
            train_pieces = []
            if test_start > 0:
                train_pieces.append(signal[:test_start])
            if test_end < time_len:
                train_pieces.append(signal[test_end:])
            concat_axis = 0
        else:
            signal_test = signal[:, test_start:test_end]
            train_pieces = []
            if test_start > 0:
                train_pieces.append(signal[:, :test_start])
            if test_end < time_len:
                train_pieces.append(signal[:, test_end:])
            concat_axis = 1

        if signal_test.shape[-1] == 0 or not train_pieces:
            # Degenerate case (e.g. a subject's signal too short to
            # produce a nonzero chunk at this boundary) -- skip this
            # subject for this fold rather than corrupt the split.
            continue

        # Fit the scaler on THIS subject's pooled train chunks only, then
        # apply it to both the test chunk and every train chunk.
        train_concat_for_fit = np.concatenate(train_pieces, axis=concat_axis)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        _fit_transform_scaler(scaler, train_concat_for_fit)  # fit only; discard the transformed output here

        norm_test = _transform_scaler(scaler, signal_test)
        segs_test = _window(norm_test, segment_length, step)

        segs_train = []
        for piece in train_pieces:
            norm_piece = _transform_scaler(scaler, piece)
            segs_train.extend(_window(norm_piece, segment_length, step))

        X_train.extend(segs_train)
        y_train.extend([label] * len(segs_train))
        sid_train.extend([subject_id] * len(segs_train))

        X_test.extend(segs_test)
        y_test.extend([label] * len(segs_test))
        sid_test.extend([subject_id] * len(segs_test))

    return (
        np.array(X_train), np.array(y_train), np.array(sid_train),
        np.array(X_test), np.array(y_test), np.array(sid_test),
    )


def train_val_split(
    X_train: np.ndarray, y_train: np.ndarray, sid_train: np.ndarray,
    val_fraction: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random split of the training pool, used ONLY to pick the
    best training epoch. Independent of (and never overlapping with) the
    held-out subject's test windows, since it only ever operates on the
    already-built training pool."""
    idx = np.arange(len(X_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=y_train
    )
    return (
        X_train[train_idx], y_train[train_idx], sid_train[train_idx],
        X_train[val_idx], y_train[val_idx], sid_train[val_idx],
    )


def check_no_leakage(X_a: np.ndarray, X_b: np.ndarray) -> int:
    """Count exact-duplicate windows shared between two window arrays.
    Should always be 0 -- assert on this after building every split.
    Works for both 1D-window arrays (N, T) and multi-channel-window
    arrays (N, C, T), since .tobytes() flattens either shape
    consistently."""
    if len(X_a) == 0 or len(X_b) == 0:
        return 0
    set_a = {x.tobytes() for x in X_a}
    set_b = {x.tobytes() for x in X_b}
    return len(set_a & set_b)


# --------------------------------------------------------------------- #
# PyTorch Dataset
# --------------------------------------------------------------------- #
class ECGDataset(Dataset):
    """(signal, emotion_label, subject_id) triples as tensors. `X` can be
    either (N, T) [single-channel] or (N, C, T) [multi-channel] -- both
    pass through torch.tensor() unchanged, and the model's forward()
    handles either shape via its `in_channels` setting."""

    def __init__(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_ids[idx]