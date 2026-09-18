"""Phase 2: CSP + LDA / SVM with within-subject and LOSO splits."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from models.utils import load_epochs, metrics_dict, write_json


def make_pipeline(clf_name: str) -> Pipeline:
    csp = CSP(
        n_components=config.CSP_COMPONENTS,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
    )
    if clf_name == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif clf_name == "svm":
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    else:
        raise ValueError(clf_name)
    return Pipeline(
        [
            ("csp", csp),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def within_subject(X, y, subjects, clf_name: str) -> dict:
    y_true_all, y_pred_all = [], []
    per_subject = {}
    t0 = time.perf_counter()
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.WITHIN_TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    for subj in np.unique(subjects):
        mask = subjects == subj
        Xs, ys = X[mask], y[mask]
        # Skip subjects that don't have at least 2 samples of every class in both splits
        counts = np.bincount(ys, minlength=len(config.CLASS_NAMES))
        if np.any(counts < 5):
            print(f"  skip subject {subj}: class counts {counts.tolist()}")
            continue
        train_idx, test_idx = next(splitter.split(Xs, ys))
        pipe = make_pipeline(clf_name)
        pipe.fit(Xs[train_idx], ys[train_idx])
        pred = pipe.predict(Xs[test_idx])
        acc = float((pred == ys[test_idx]).mean())
        per_subject[str(int(subj))] = acc
        y_true_all.append(ys[test_idx])
        y_pred_all.append(pred)
    elapsed = time.perf_counter() - t0
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    out = metrics_dict(y_true, y_pred, elapsed)
    out["mean_subject_accuracy"] = float(np.mean(list(per_subject.values())))
    out["per_subject_accuracy"] = per_subject
    return out


def leave_one_subject_out(X, y, subjects, clf_name: str) -> dict:
    y_true_all, y_pred_all = [], []
    per_subject = {}
    t0 = time.perf_counter()
    uniq = np.unique(subjects)
    for held in tqdm(uniq, desc=f"LOSO {clf_name}"):
        train = subjects != held
        test = subjects == held
        pipe = make_pipeline(clf_name)
        pipe.fit(X[train], y[train])
        pred = pipe.predict(X[test])
        acc = float((pred == y[test]).mean())
        per_subject[str(int(held))] = acc
        y_true_all.append(y[test])
        y_pred_all.append(pred)
    elapsed = time.perf_counter() - t0
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    out = metrics_dict(y_true, y_pred, elapsed)
    out["mean_subject_accuracy"] = float(np.mean(list(per_subject.values())))
    out["per_subject_accuracy"] = per_subject
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=config.DATA_PROCESSED / "epochs.npz")
    parser.add_argument("--out", type=Path, default=config.RESULTS / "classical_metrics.json")
    args = parser.parse_args()

    ds = load_epochs(args.data)
    X, y, subjects = ds["X"], ds["y"], ds["subjects"]
    print(
        f"Loaded {len(y)} epochs, X={X.shape}, "
        f"subjects={len(np.unique(subjects))}, classes={np.bincount(y).tolist()}"
    )

    results = {"class_names": config.CLASS_NAMES, "methods": {}}
    for clf_name in ("lda", "svm"):
        print(f"\n=== {clf_name.upper()} within-subject ===")
        within = within_subject(X, y, subjects, clf_name)
        print(
            f"  pooled acc={within['accuracy']:.3f}  "
            f"mean-subject acc={within['mean_subject_accuracy']:.3f}  "
            f"time={within['train_seconds']:.1f}s"
        )
        print(f"\n=== {clf_name.upper()} leave-one-subject-out ===")
        loso = leave_one_subject_out(X, y, subjects, clf_name)
        print(
            f"  pooled acc={loso['accuracy']:.3f}  "
            f"mean-subject acc={loso['mean_subject_accuracy']:.3f}  "
            f"time={loso['train_seconds']:.1f}s"
        )
        results["methods"][clf_name] = {"within_subject": within, "cross_subject": loso}

    write_json(args.out, results)
    print("\nPhase 2 summary")
    for name, block in results["methods"].items():
        w = block["within_subject"]["accuracy"]
        c = block["cross_subject"]["accuracy"]
        print(f"  {name}: within={w:.3f}  LOSO={c:.3f}  gap={w-c:.3f}")


if __name__ == "__main__":
    main()
