from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetSpec:

    path: Path
    format: str
    num_emotions: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.path = Path(self.path)

DATASET_PRESETS: Dict[str, List[DatasetSpec]] = {
    "dreamer": [
        DatasetSpec(path=Path("DREAMER.mat"), format="dreamer"),
    ],
    "wesad": [
        DatasetSpec(path=Path("wesad/"), format="wesad", num_emotions=4),
    ],
    "mand": [
        DatasetSpec(
            path=Path(r"mand"),
            format="mand",
            num_emotions=5,
        )
    ],
    "all": [
        DatasetSpec(path=Path("DREAMER.mat"), format="dreamer"),
        DatasetSpec(path=Path("wesad/"), format="wesad", num_emotions=4),
        DatasetSpec(path=Path("mand/"), format="mand", num_emotions=5),
    ],
}

# Architecture ablations used in the paper.
# `use_nmoe` is retained as an internal compatibility name for learned hierarchical fusion.
ABLATION_PRESETS: Dict[str, Dict[str, bool]] = {
    "full": {"use_stft": True, "use_nmoe": True},
    "no_stft": {"use_stft": False, "use_nmoe": True},
    "no_nmoe": {"use_stft": True, "use_nmoe": False},
    "no_stft_no_nmoe": {"use_stft": False, "use_nmoe": False},
}


@dataclass
class Config:
    # Pretrained ECG foundation model
    root_dir: Path = Path(".")
    ckpt_dir: Path = Path("ckpts")
    pretrained_repo_id: str = "wanglab/ecg-fm-preprint"
    pretrained_filename: str = "mimic_iv_ecg_physionet_pretrained.pt"
    pretrained_config_filename: str = "mimic_iv_ecg_physionet_pretrained.yaml"

    dataset_specs: List[DatasetSpec] = field(
        default_factory=lambda: list(DATASET_PRESETS["dreamer"])
    )

    output_dir: Path = Path("outputs")
    seed: int = 42

    # Signal preprocessing
    fs: int = 256
    window_size_s: int = 10
    overlap: float = 0.0
    highpass_cutoff_hz: float = 0.5

    # Evaluation protocol
    calibration_split: float = 0.5
    loso_use_calibration: bool = False   # False = strict LOSO, no subject-specific calibration
    val_split: float = 0.1
    intra_subject_test_split: float = 0.1
    intra_subject_n_splits: int = 10
    intra_subject_fold_index: int = 0

    # Data loading
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    # How many DREAMER ECG lead(s) to load. 1 = original single-lead
    # behavior; 2 = both leads. Only DREAMER's loader accepts this.
    # Must equal `in_channels` below.
    num_channels: int = 2

    # Model
    num_emotions: int = 5
    label_target: str = "arousal"
    hidden_dim: int = 768
    use_attn_pooling: bool = True
    use_input_conditioned_gating: bool = False
    # Architecture switches
    # Both True = full framework. Set individually, or via
    # ABLATION_PRESETS[name] / apply_ablation_preset() below.
    use_stft: bool = False
    use_nmoe: bool = True
    # Number of ECG leads YOUR data provides -- must equal num_channels
    # above. The pretrained backbone's first conv layer is NEVER
    # replaced/reinitialized regardless of this value -- see
    # model.EmotionRec's docstring for how in_channels < native_channels
    # is handled (zero-padding into lead_indices, not architecture
    # change).
    in_channels: int = 2
    # Where your `in_channels` map into the pretrained model's native
    # channel slots. None -> defaults to [0, 1, ..., in_channels-1] with
    # a warning (see EmotionRec's docstring) -- set explicitly once
    # you've confirmed the pretrained model's real lead order and
    # DREAMER's actual lead identity.
    # lead_indices: Optional[List[int]] = None
    lead_indices: List[int] = field(default_factory=lambda: [0, 5])

    # Experiment organization
    experiment_tag: str = ""

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 100
    early_stop_patience: int = 30

    # Runtime
    use_amp: bool = True
    cudnn_benchmark: bool = False

    # Checkpointing
    best_metric: str = "val_f1"

    resume: bool = True

    @property
    def segment_length(self) -> int:
        return self.fs * self.window_size_s

    @property
    def pretrained_checkpoint_path(self) -> Path:
        return self.root_dir / self.ckpt_dir / self.pretrained_filename

    def apply_ablation_preset(self, name: str) -> "Config":
        """Set use_stft/use_nmoe from ABLATION_PRESETS[name] in place.
        Returns self for chaining, e.g. `Config().apply_ablation_preset('no_stft')`."""
        if name not in ABLATION_PRESETS:
            raise ValueError(f"Unknown ablation preset {name!r}. Available: {sorted(ABLATION_PRESETS.keys())}.")
        preset = ABLATION_PRESETS[name]
        self.use_stft = preset["use_stft"]
        self.use_nmoe = preset["use_nmoe"]
        return self

    def __post_init__(self):
        if not (0.0 < self.calibration_split < 1.0):
            raise ValueError("calibration_split must be in (0, 1).")
        if not (0.0 < self.val_split < 1.0):
            raise ValueError("val_split must be in (0, 1).")
        if not (0.0 < self.intra_subject_test_split < 1.0):
            raise ValueError("intra_subject_test_split must be in (0, 1).")
        if self.intra_subject_n_splits < 2:
            raise ValueError("intra_subject_n_splits must be at least 2.")
        if not (0 <= self.intra_subject_fold_index < self.intra_subject_n_splits):
            raise ValueError(
                f"intra_subject_fold_index must be in [0, {self.intra_subject_n_splits - 1}], "
                f"got {self.intra_subject_fold_index}."
            )
        if self.label_target not in ("quadrant", "valence", "arousal", "arousal_binary", "valence_binary"):
            raise ValueError(f"label_target invalid, got {self.label_target!r}.")
        if self.num_channels != self.in_channels:
            raise ValueError(
                f"num_channels ({self.num_channels}) must equal in_channels ({self.in_channels})."
            )
        if not self.use_stft and not self.use_nmoe:
            # Still a valid, meaningful ablation cell (uniform average over
            # encoder layers only) -- just flagged so it's not mistaken for
            # a bug when acc drops the most in this row.
            print(
                "[Config] NOTE: use_stft=False and use_nmoe=False -- fusion is a fixed "
                "uniform average over encoder layers only (no STFT branch, no learned mixture)."
            )
        if not self.use_nmoe and self.use_input_conditioned_gating:
            print(
                "[Config] NOTE: use_input_conditioned_gating is ignored because use_nmoe=False "
                "(no learned fusion of any kind is applied when NMoE is disabled)."
            )