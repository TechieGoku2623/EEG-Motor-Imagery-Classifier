# EEG motor-imagery results

Five-class problem: `rest`, `real_hands`, `real_feet`, `imagined_hands`, `imagined_feet`.
Chance accuracy is **20%**.

## Accuracy summary

| Method | Within acc | Within F1 | LOSO acc | LOSO F1 | Gap (acc) | Time within / LOSO |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 0.473 | 0.406 | 0.298 | 0.191 | 0.175 | 35s / 546s |
| SVM | 0.477 | 0.461 | 0.217 | 0.200 | 0.260 | 32s / 525s |
| EEGNet | 0.289 | 0.264 | 0.322 | 0.274 | -0.033 | 43s / 938s |

## Why within-subject and cross-subject differ

Motor-imagery EEG is highly **subject-specific**. Sensorimotor-rhythm (mu/beta, 8–30 Hz) topography, SNR, and even the preferred imagery strategy vary across people. CSP finds spatial filters that maximize class variance *on the training subjects*; those filters overfit individual anatomy and montage coupling, so leave-one-subject-out accuracy drops sharply versus a within-subject train/test split.

A compact CNN can learn slightly more invariant spectro-spatial patterns, but with only 20 subjects and no domain-alignment it still fails to close the generalization gap. That gap is the central obstacle for real-world BCI: calibration-free systems must work on unseen users, yet most published within-subject numbers overstate field performance.

On this run, **EEGNet** had the strongest cross-subject accuracy (0.322). That LOSO number is still far below within-subject CSP+LDA (~0.47), which is why clinical/consumer BCIs still rely on per-user calibration or transfer-learning / Riemannian alignment rather than a single off-the-shelf model.

EEGNet's within-subject score can land *below* its LOSO score: each subject only has ~250 epochs, which is too little to train a CNN, whereas LOSO trains on the other 19 subjects (~4.7k epochs). CSP+LDA is the opposite — it shines with a few minutes of calibration data from the same person and collapses across people.

## Class-level patterns

- **Rest vs movement** is usually the easiest contrast (mu-rhythm amplitude).
- **Hands vs feet** is spatially distinct (lateral vs midline motor cortex) but feet trials are scarce (only T2 in three runs), so those classes are often absorbed into the corresponding hands class.
- **Real vs imagined** of the same limb is the hardest pair: imagery is a weaker, noisier version of the same rhythm, so models often confuse those classes with each other and with rest.

Confusion matrices for every method/split are saved under `results/`.
