# Contributing

PRs are welcome. Things that would help:

- more subjects in `config.py`
- alignment methods for LOSO
- GPU training options

Don't commit EDFs or `data/processed/*.npz`.

```bash
python3 data/preprocess.py --subjects 1-2 --runs 3-4
python3 models/csp_baseline.py
```
