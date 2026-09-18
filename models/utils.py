"""Shared dataset loading and evaluation helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def load_epochs(path: Path | None = None) -> dict:
    path = path or (config.DATA_PROCESSED / "epochs.npz")
    data = np.load(path)
    return {
        "X": data["X"],
        "y": data["y"],
        "subjects": data["subjects"],
        "runs": data["runs"],
    }


def metrics_dict(y_true, y_pred, elapsed_s: float | None = None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(config.CLASS_NAMES)))
        ).tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(config.CLASS_NAMES))),
            target_names=config.CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "n_samples": int(len(y_true)),
    }
    if elapsed_s is not None:
        out["train_seconds"] = float(elapsed_s)
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")
