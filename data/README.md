Raw PhysioNet EDF files land in `raw/` (git-ignored). Processed `epochs.npz` lands in `processed/`.

```bash
python3 data/download.py --subjects 1-20 --runs 3-14
python3 data/preprocess.py --subjects 1-20 --runs 3-14
```
