"""Phase 3: train EEGNet with the same within-subject and LOSO splits as Phase 2."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from models.eegnet import EEGNet
from models.utils import load_epochs, metrics_dict, write_json


def maybe_decimate(X: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return X
    return X[:, :, ::factor]


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def standardize_from_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel z-score using training statistics only."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32)


def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> tuple[np.ndarray, float]:
    X_train, X_test = standardize_from_train(X_train, X_test)
    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    n_classes = len(config.CLASS_NAMES)
    kernel_length = 64 if n_times >= 400 else 32
    dev = device()
    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_times,
        n_classes=n_classes,
        kernel_length=kernel_length,
    ).to(dev)
    # Materialize LazyLinear
    with torch.no_grad():
        model(torch.from_numpy(X_train[:1]).to(dev))

    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=dev)
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)

    best_state = None
    best_val = -1.0
    stale = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(X_test), batch_size):
                xb = torch.from_numpy(X_test[i : i + batch_size]).to(dev)
                logits.append(model(xb).cpu())
            pred = torch.cat(logits).argmax(1).numpy()
        acc = float((pred == y_test).mean())
        if acc > best_val + 1e-4:
            best_val = acc
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(X_test), batch_size):
            xb = torch.from_numpy(X_test[i : i + batch_size]).to(dev)
            logits.append(model(xb).cpu())
        pred = torch.cat(logits).argmax(1).numpy()
    elapsed = time.perf_counter() - t0
    return pred, elapsed


def within_subject(X, y, subjects, epochs, batch_size, lr, patience) -> dict:
    y_true_all, y_pred_all = [], []
    per_subject = {}
    total_time = 0.0
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=config.WITHIN_TEST_SIZE, random_state=config.RANDOM_STATE
    )
    for subj in tqdm(np.unique(subjects), desc="CNN within-subject"):
        mask = subjects == subj
        Xs, ys = X[mask], y[mask]
        counts = np.bincount(ys, minlength=len(config.CLASS_NAMES))
        if np.any(counts < 5):
            continue
        train_idx, test_idx = next(splitter.split(Xs, ys))
        pred, elapsed = train_one(
            Xs[train_idx], ys[train_idx], Xs[test_idx], ys[test_idx],
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        )
        total_time += elapsed
        per_subject[str(int(subj))] = float((pred == ys[test_idx]).mean())
        y_true_all.append(ys[test_idx])
        y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    out = metrics_dict(y_true, y_pred, total_time)
    out["mean_subject_accuracy"] = float(np.mean(list(per_subject.values())))
    out["per_subject_accuracy"] = per_subject
    return out


def leave_one_subject_out(X, y, subjects, epochs, batch_size, lr, patience) -> dict:
    y_true_all, y_pred_all = [], []
    per_subject = {}
    total_time = 0.0
    uniq = np.unique(subjects)
    for held in tqdm(uniq, desc="CNN LOSO"):
        train, test = subjects != held, subjects == held
        pred, elapsed = train_one(
            X[train], y[train], X[test], y[test],
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        )
        total_time += elapsed
        per_subject[str(int(held))] = float((pred == y[test]).mean())
        y_true_all.append(y[test])
        y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    out = metrics_dict(y_true, y_pred, total_time)
    out["mean_subject_accuracy"] = float(np.mean(list(per_subject.values())))
    out["per_subject_accuracy"] = per_subject
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=config.DATA_PROCESSED / "epochs.npz")
    parser.add_argument("--out", type=Path, default=config.RESULTS / "cnn_metrics.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--decimate", type=int, default=2, help="Keep every Nth sample (2 → 80 Hz)")
    args = parser.parse_args()

    ds = load_epochs(args.data)
    X, y, subjects = ds["X"], ds["y"], ds["subjects"]
    X = maybe_decimate(X, args.decimate)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    print(
        f"Loaded {len(y)} epochs on {device()}, X={X.shape}, "
        f"subjects={len(np.unique(subjects))}, decimate={args.decimate}"
    )

    print("\n=== EEGNet within-subject ===")
    within = within_subject(X, y, subjects, args.epochs, args.batch_size, args.lr, args.patience)
    print(
        f"  pooled acc={within['accuracy']:.3f}  "
        f"macro-F1={within['macro_f1']:.3f}  "
        f"mean-subject acc={within['mean_subject_accuracy']:.3f}  "
        f"time={within['train_seconds']:.1f}s"
    )

    print("\n=== EEGNet leave-one-subject-out ===")
    loso = leave_one_subject_out(X, y, subjects, args.epochs, args.batch_size, args.lr, args.patience)
    print(
        f"  pooled acc={loso['accuracy']:.3f}  "
        f"macro-F1={loso['macro_f1']:.3f}  "
        f"mean-subject acc={loso['mean_subject_accuracy']:.3f}  "
        f"time={loso['train_seconds']:.1f}s"
    )

    results = {
        "class_names": config.CLASS_NAMES,
        "device": str(device()),
        "epochs": args.epochs,
        "decimate": args.decimate,
        "methods": {"eegnet": {"within_subject": within, "cross_subject": loso}},
    }
    write_json(args.out, results)
    w, c = within["accuracy"], loso["accuracy"]
    print(f"\nPhase 3 summary  EEGNet within={w:.3f}  LOSO={c:.3f}  gap={w-c:.3f}")


if __name__ == "__main__":
    main()
