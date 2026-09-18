# EEG motor-imagery results

Five classes: rest, real_hands, real_feet, imagined_hands, imagined_feet.
Chance is 20%. After I downsampled rest, majority is about 27%.
See README.md for how I set the experiment up.

## Accuracy summary

| Method | Within acc | Within F1 | LOSO acc | LOSO F1 | Gap (acc) | Time within / LOSO |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 0.473 | 0.406 | 0.298 | 0.191 | 0.175 | 35s / 546s |
| SVM | 0.477 | 0.461 | 0.217 | 0.200 | 0.260 | 32s / 525s |
| EEGNet | 0.289 | 0.264 | 0.322 | 0.274 | -0.033 | 43s / 938s |

## Within-subject vs LOSO

EEG looks pretty different across people (mu/beta pattern, noise, how they imagine the movement). CSP fits spatial filters on whoever is in the training set, so it does well if I test on the same person and worse if I leave them out.

On this run **EEGNet** had the best LOSO accuracy (0.322). That's still a lot lower than within-subject CSP (~0.47). EEGNet can beat its own within-subject score on LOSO because each person only has ~250 epochs, while LOSO trains on the other 19 people.

## Per class

- Rest vs movement is the easier split.
- Feet have fewer trials (only T2 on three runs) so they often get predicted as hands.
- Real vs imagined of the same limb is messy; imagery looks like a weaker version of the same rhythm.

Confusion matrices are saved next to this file.
