EDFs go in `raw/` (gitignored). After preprocess, `processed/epochs.npz`.

```bash
python3 data/download.py --subjects 1-20 --runs 3-14
python3 data/preprocess.py --subjects 1-20 --runs 3-14
```
