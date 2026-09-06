# HiTS-AMAP

Code accompanying the manuscript on ECG-based emotion recognition with target-subject calibration.

HiTS-AMAP combines hierarchical representations from a frozen ECG foundation model with an STFT-based spectral representation. The representations are adaptively fused and summarized using average-max attention pooling (AMAP) before classification.

> **Review version.** This repository is intended for private peer review. Please do not redistribute the code or datasets. A public version can be released after acceptance/publication.

## Repository structure

```text
HiTS-AMAP/
├── main.py                  # Experiment entry point
├── requirements.txt
└── ecg_emotion/
    ├── config.py            # Dataset, model, and training settings
    ├── data.py              # Dataset loading and preprocessing
    ├── model.py             # HiTS-AMAP model
    ├── train.py             # Training and evaluation loops
    ├── diagnostics.py       # Representation diagnostics
    └── seeding.py           # Reproducibility utilities
```

Raw datasets, pretrained checkpoints, and experiment outputs are intentionally excluded from the repository.

## Supported datasets

The current code supports the three datasets used in the study:

- **DREAMER** — arousal and valence recognition, including binary and five-class settings.
- **WESAD** — four classes: neutral, stress, amusement, and meditation.
- **manD 1.0** — five emotion states: anger, neutral/no-emotion, fear, sadness, and surprise.

Dataset locations are defined in `ecg_emotion/config.py` and can be changed as needed.

Expected default layout:

```text
HiTS-AMAP/
├── DREAMER.mat
├── wesad/
│   ├── S2_ecg.pkl
│   └── ...
└── mand/
    ├── P1_ecg.pkl
    └── ...
```

For WESAD, each subject file is expected to contain the subject identifier, ECG signal, and sample-level labels. For manD, the loader expects one preprocessed ECG pickle per participant with the structure documented in `load_mand_signals()` in `ecg_emotion/data.py`.

The raw datasets are not redistributed here and should be obtained from their original sources in accordance with their licenses/terms of use.

## Environment

The experiments were developed with Python and PyTorch. Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

The ECG foundation model uses `fairseq_signals`, which is installed separately from source. Follow the upstream `fairseq-signals` installation instructions for the version compatible with your environment.

At the first run, the pretrained ECG foundation-model checkpoint and configuration are downloaded from the repository specified by `pretrained_repo_id` in `ecg_emotion/config.py` and stored under `ckpts/`.

## Running experiments

The full architecture is selected with the `full` ablation preset:

```bash
python main.py --dataset dreamer --protocol loso --ablation full
```

Available dataset presets are `dreamer`, `wesad`, `mand`, and `all`. Available protocols are:

- `loso` — leave-one-subject-out evaluation;
- `intra_subject` — one selected intra-subject fold;
- `intra_subject_kfold` — repeated intra-subject cross-validation.

For example:

```bash
python main.py --dataset wesad --protocol intra_subject_kfold --ablation full
python main.py --dataset mand --protocol loso --ablation full
```

### DREAMER tasks

The DREAMER target is controlled by `Config.label_target`. Supported values include `arousal`, `valence`, `arousal_binary`, `valence_binary`, and `quadrant`.

Example for five-class valence recognition:

```python
from ecg_emotion.config import Config
from main import main

cfg = Config(label_target="valence", num_emotions=5)
cfg.apply_ablation_preset("full")
main(cfg=cfg, dataset="dreamer", protocol="loso")
```

### Target-subject calibration

Strict LOSO uses no samples from the held-out participant:

```python
cfg.loso_use_calibration = False
```

For calibrated evaluation, enable calibration and set the fraction of the target participant used for calibration. For example, 10% calibration is configured as:

```python
from ecg_emotion.config import Config
from main import main

cfg = Config(loso_use_calibration=True, calibration_split=0.10)
cfg.apply_ablation_preset("full")
main(cfg=cfg, dataset="dreamer", protocol="loso")
```

The study evaluates 0%, 10%, 20%, and 50% target-subject calibration. For the 0% condition, use strict LOSO (`loso_use_calibration=False`).

## Ablation settings

The architecture switches used for the ablation experiments are defined in `ABLATION_PRESETS`:

- `full` — hierarchical temporal fusion + STFT spectral representation;
- `no_stft` — temporal representations only;
- `no_nmoe` — STFT retained without learned hierarchical expert weighting;
- `no_stft_no_nmoe` — temporal-only baseline without learned expert weighting.

All presets can be evaluated sequentially with:

```bash
python main.py --dataset dreamer --protocol loso --ablation-study
```

`use_nmoe` is retained as an internal variable name for compatibility with the experiment code. In the manuscript, this component is described as the learned adaptive/hierarchical expert-fusion mechanism.

## Outputs and reproducibility

Experiment outputs are written to `outputs/`. Depending on the protocol, the code stores metrics, predictions, confusion matrices, learned layer/expert weights, and run configuration information.

Random seeds are set through `ecg_emotion/seeding.py`. Dataset splitting is performed at the subject level for LOSO experiments to keep the held-out participant separate from the source-subject training data.

## Code availability

This private version is provided for peer review. The public repository URL and formal citation can be added after acceptance/publication.
