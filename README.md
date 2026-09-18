# EEG Motor-Imagery Classifier

End-to-end classification of PhysioNet **EEG Motor Movement/Imagery** recordings
([eegmmidb 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)) into five classes:

| ID | Class | Meaning |
|---:|---|---|
| 0 | `rest` | T0 (inter-trial rest) |
| 1 | `real_hands` | Executed left/right/both-fist movement |
| 2 | `real_feet` | Executed both-feet movement |
| 3 | `imagined_hands` | Imagined left/right/both-fist movement |
| 4 | `imagined_feet` | Imagined both-feet movement |

The pipeline compares **CSP + LDA/SVM** against a compact **EEGNet-style CNN**, under
**within-subject** and **leave-one-subject-out (cross-subject)** splits.

## Dataset

- 64-channel EEG, 160 Hz, international 10-10 montage (BCI2000).
- 14 runs per subject; this project uses motor runs **3–14** (baselines 1–2 skipped).
- Event codes: `T0` rest; `T1`/`T2` mean left vs right fist **or** both fists vs both
  feet depending on the run (see `config.py`).
- Default subject list: **S001–S020** (configurable).

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Python 3.11+ (developed on 3.12). CPU is enough for the classical models; a GPU
speeds up EEGNet leave-one-subject-out.

## Reproduce

```bash
# Phase 1 — download EDFs with wfdb, bandpass 8–30 Hz, epoch, save .npz
python3 data/download.py --subjects 1-20 --runs 3-14
python3 data/preprocess.py --subjects 1-20 --runs 3-14

# Phase 2 — CSP + LDA / SVM
python3 models/csp_baseline.py

# Phase 3 — EEGNet
python3 models/train_cnn.py --epochs 15

# Phase 4 — plots + interpretation
python3 models/visualize.py
```

Outputs land in `results/`:

- `classical_metrics.json`, `cnn_metrics.json`, `comparison.json`
- `cm_*.png` confusion matrices (one per method × split)
- `accuracy_comparison.png`
- `REPORT.md` — which approach generalizes and why it matters for BCI

## Methodology

1. **Preprocessing.** EDF load via MNE, 10-10 montage, FIR bandpass **8–30 Hz**
   (mu/beta motor rhythms), 0–4 s epochs locked to annotations.
2. **Classical.** MNE `CSP` (6 log-variance components, Ledoit–Wolf covariance)
   → StandardScaler → LDA (shrinkage) or RBF-SVM (`class_weight=balanced`).
3. **Deep learning.** Compact EEGNet (temporal conv + depthwise spatial conv +
   separable conv) on z-scored epochs, weighted cross-entropy, early stopping.
4. **Splits.**
   - *Within-subject:* stratified 80/20 split **inside each subject**, predictions
     pooled for the reported accuracy.
   - *Cross-subject:* leave-one-subject-out. This is the number that matters for
     calibration-free BCI.

Expect a large **generalization gap**: CSP spatial filters and even CNNs overfit
individual mu-rhythm topography. See `results/REPORT.md` after a full run.

## Repo layout

```
config.py              # paths, class map, filter/epoch settings
data/download.py       # wfdb.dl_files EDF downloader
data/preprocess.py     # MNE filter + epoch → data/processed/epochs.npz
models/csp_baseline.py
models/eegnet.py
models/train_cnn.py
models/visualize.py
notebooks/01_exploration.ipynb
results/
```

Raw EDF files and the processed `.npz` are git-ignored (hundreds of MB).
