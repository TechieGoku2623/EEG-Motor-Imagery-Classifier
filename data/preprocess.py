"""Load EDFs with MNE, bandpass-filter, epoch, and save numpy arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.download import parse_subjects


mne.set_log_level("ERROR")


def label_for_event(run: int, description: str) -> str | None:
    """Map (run, T0/T1/T2) to a 5-class name."""
    desc = description.strip()
    if desc == "T0":
        return "rest"
    if run in config.REAL_HAND_LR_RUNS:
        if desc in {"T1", "T2"}:
            return "real_hands"
    elif run in config.IMAG_HAND_LR_RUNS:
        if desc in {"T1", "T2"}:
            return "imagined_hands"
    elif run in config.REAL_HANDS_FEET_RUNS:
        if desc == "T1":
            return "real_hands"
        if desc == "T2":
            return "real_feet"
    elif run in config.IMAG_HANDS_FEET_RUNS:
        if desc == "T1":
            return "imagined_hands"
        if desc == "T2":
            return "imagined_feet"
    return None


def find_edf(raw_dir: Path, subject: int, run: int) -> Path | None:
    name = f"S{subject:03d}R{run:02d}.edf"
    candidates = [
        raw_dir / f"S{subject:03d}" / name,
        raw_dir / name,
        raw_dir / "eegmmidb" / f"S{subject:03d}" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(raw_dir.rglob(name))
    return matches[0] if matches else None


def load_and_filter(edf_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None)
    eegbci.standardize(raw)
    raw.pick("eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    raw.filter(
        l_freq=config.BANDPASS[0],
        h_freq=config.BANDPASS[1],
        fir_design="firwin",
        skip_by_annotation="edge",
    )
    return raw


def epoch_run(raw: mne.io.BaseRaw, run: int) -> tuple[np.ndarray, np.ndarray] | None:
    events, event_id = mne.events_from_annotations(raw, event_id=None)
    if events.size == 0:
        return None

    # Map annotation descriptions to class ids we care about
    keep_id_to_class: dict[int, int] = {}
    for desc, eid in event_id.items():
        cls_name = label_for_event(run, desc)
        if cls_name is not None:
            keep_id_to_class[eid] = config.CLASS_TO_ID[cls_name]

    mask = np.isin(events[:, 2], list(keep_id_to_class.keys()))
    events = events[mask]
    if len(events) == 0:
        return None

    # Remap event codes to class ids so epochs carry the 5-class labels
    remapped = events.copy()
    for eid, cid in keep_id_to_class.items():
        remapped[events[:, 2] == eid, 2] = cid

    epochs = mne.Epochs(
        raw,
        remapped,
        event_id={name: i for name, i in config.CLASS_TO_ID.items()},
        tmin=config.EPOCH_TMIN,
        tmax=config.EPOCH_TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        on_missing="ignore",
    )
    if len(epochs) == 0:
        return None
    data = epochs.get_data(units="uV")  # (n_epochs, n_ch, n_times)
    labels = epochs.events[:, 2]
    return data.astype(np.float32), labels.astype(np.int64)


def preprocess_subject(raw_dir: Path, subject: int, runs: list[int]) -> dict | None:
    X_parts, y_parts, run_parts = [], [], []
    for run in runs:
        edf = find_edf(raw_dir, subject, run)
        if edf is None:
            print(f"  missing S{subject:03d} R{run:02d}, skipping")
            continue
        try:
            raw = load_and_filter(edf)
            result = epoch_run(raw, run)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed S{subject:03d} R{run:02d}: {exc}")
            continue
        if result is None:
            continue
        data, labels = result
        X_parts.append(data)
        y_parts.append(labels)
        run_parts.append(np.full(len(labels), run, dtype=np.int16))
        del raw

    if not X_parts:
        return None
    data = {
        "X": np.concatenate(X_parts, axis=0),
        "y": np.concatenate(y_parts, axis=0),
        "runs": np.concatenate(run_parts, axis=0),
        "subject": np.full(sum(len(p) for p in y_parts), subject, dtype=np.int16),
    }
    return data


def balance_rest(part: dict, rng: np.random.Generator) -> dict:
    """Downsample rest so it matches the largest non-rest class (per subject).

    T0 appears on every trial, so rest is otherwise ~50% of epochs and raw
    accuracy is dominated by the majority class.
    """
    y = part["y"]
    rest_id = config.CLASS_TO_ID["rest"]
    rest_idx = np.where(y == rest_id)[0]
    other_counts = [
        int(np.sum(y == i))
        for i in range(len(config.CLASS_NAMES))
        if i != rest_id
    ]
    n_keep = min(len(rest_idx), max(other_counts) if other_counts else 0)
    keep_rest = rng.choice(rest_idx, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([keep_rest, np.where(y != rest_id)[0]]))
    return {k: v[keep] for k, v in part.items()}


def save_dataset(out_path: Path, parts: list[dict], extra_meta: dict | None = None) -> dict:
    X = np.concatenate([p["X"] for p in parts], axis=0)
    y = np.concatenate([p["y"] for p in parts], axis=0)
    subjects = np.concatenate([p["subject"] for p in parts], axis=0)
    runs = np.concatenate([p["runs"] for p in parts], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, subjects=subjects, runs=runs)

    counts = {name: int(np.sum(y == i)) for i, name in enumerate(config.CLASS_NAMES)}
    meta = {
        "n_epochs": int(len(y)),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "n_subjects": int(len(np.unique(subjects))),
        "subjects": [int(s) for s in np.unique(subjects)],
        "class_names": config.CLASS_NAMES,
        "class_counts": counts,
        "bandpass_hz": list(config.BANDPASS),
        "tmin": config.EPOCH_TMIN,
        "tmax": config.EPOCH_TMAX,
        "sfreq": config.SFREQ,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EEGMMIDB EDFs into epochs")
    parser.add_argument("--subjects", default="1-20")
    parser.add_argument("--runs", default="3-14")
    parser.add_argument("--raw-dir", type=Path, default=config.DATA_RAW)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.DATA_PROCESSED / "epochs.npz",
    )
    parser.add_argument(
        "--keep-all-rest",
        action="store_true",
        help="Do not downsample T0/rest epochs (accuracy will be majority-class dominated)",
    )
    args = parser.parse_args()
    subjects = parse_subjects(args.subjects)
    runs = parse_subjects(args.runs)
    rng = np.random.default_rng(config.RANDOM_STATE)

    parts = []
    for subject in tqdm(subjects, desc="subjects"):
        result = preprocess_subject(args.raw_dir, subject, runs)
        if result is None:
            print(f"No epochs for subject {subject}")
            continue
        n_raw = len(result["y"])
        if not args.keep_all_rest:
            result = balance_rest(result, rng)
        parts.append(result)
        print(
            f"S{subject:03d}: {len(result['y'])} epochs "
            f"(from {n_raw}), shape={result['X'].shape[1:]}"
        )

    if not parts:
        raise SystemExit("No epochs were created. Did download succeed?")

    meta = save_dataset(
        args.out,
        parts,
        extra_meta={"rest_balanced": not args.keep_all_rest},
    )
    print("Saved", args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
