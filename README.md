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
- Default subject list: **S001–S020**.
- After 8–30 Hz filtering, 0–4 s epoching, and rest downsampling: **4,959 epochs**,
  64 channels, 641 samples. Class counts: rest 1359, real_hands 1345, real_feet 455,
  imagined_hands 1351, imagined_feet 449. Majority-class baseline ≈ **27%**; chance = **20%**.

## Headline results (S001–S020)

| Method | Within-subject acc | LOSO acc | Gap | Train time (within / LOSO) |
|---|---:|---:|---:|---:|
| CSP + LDA | **0.473** | 0.298 | 0.175 | 35 s / 9.1 min |
| CSP + SVM | 0.477 | 0.217 | 0.260 | 32 s / 8.7 min |
| EEGNet | 0.289 | **0.322** | −0.033 | 43 s / 15.6 min |

**Takeaway:** CSP+LDA is the right tool when you can calibrate on the same user.
EEGNet is the only method that *improves* when trained on other people (it is
starved of data in the per-subject split) and it slightly wins LOSO, but 32% on
a 5-class problem is still far from a plug-and-play BCI. See `results/REPORT.md`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Python 3.11+ (developed on 3.12). CPU is enough; a GPU speeds up EEGNet LOSO.

## Reproduce

```bash
# Phase 1 — download EDFs with wfdb, bandpass 8–30 Hz, epoch, save .npz
python3 data/download.py --subjects 1-20 --runs 3-14
python3 data/preprocess.py --subjects 1-20 --runs 3-14

# Phase 2 — CSP + LDA / SVM
python3 models/csp_baseline.py

# Phase 3 — EEGNet (decimate=2 → 80 Hz input, valid for an 8–30 Hz band)
python3 models/train_cnn.py --epochs 10 --patience 4 --decimate 2

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
   (mu/beta motor rhythms), 0–4 s epochs locked to annotations. Rest (`T0`) is
   downsampled per subject to the size of the largest motor class so accuracy
   is not a rest-vs-everything majority detector (`--keep-all-rest` to disable).
2. **Classical.** MNE `CSP` (6 log-variance components, Ledoit–Wolf covariance)
   → StandardScaler → LDA (shrinkage) or RBF-SVM (`class_weight=balanced`).
3. **Deep learning.** Compact EEGNet (temporal conv + depthwise spatial conv +
   separable conv) on z-scored epochs, weighted cross-entropy, early stopping.
4. **Splits.**
   - *Within-subject:* stratified 80/20 split **inside each subject**, predictions
     pooled for the reported accuracy.
   - *Cross-subject:* leave-one-subject-out. This is the number that matters for
     calibration-free BCI.

## Repo layout

```
config.py              # paths, class map, filter/epoch settings
data/download.py       # wfdb.dl_files EDF downloader (parallel)
data/preprocess.py     # MNE filter + epoch → data/processed/epochs.npz
models/csp_baseline.py
models/eegnet.py
models/train_cnn.py
models/visualize.py
notebooks/01_exploration.ipynb
results/
```

Raw EDF files and the processed `.npz` are git-ignored (hundreds of MB).
