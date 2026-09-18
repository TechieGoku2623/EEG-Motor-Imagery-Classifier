"""Phase 4: confusion matrices, comparison bar chart, and markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


sns.set_theme(style="whitegrid", font_scale=1.05)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def flatten_methods(classical: dict, cnn: dict) -> dict:
    methods = {}
    methods.update(classical.get("methods", {}))
    methods.update(cnn.get("methods", {}))
    return methods


def plot_confusion(cm, title: str, out_path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bar(methods: dict, out_path: Path) -> dict:
    names, within, cross = [], [], []
    for name, block in methods.items():
        names.append(name.upper() if name != "eegnet" else "EEGNet")
        within.append(block["within_subject"]["accuracy"])
        cross.append(block["cross_subject"]["accuracy"])

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    b1 = ax.bar(x - width / 2, within, width, label="Within-subject", color="#2E86AB")
    b2 = ax.bar(x + width / 2, cross, width, label="Cross-subject (LOSO)", color="#E94F37")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(1 / len(config.CLASS_NAMES), color="gray", ls="--", lw=1, label="Chance (5-class)")
    ax.legend()
    ax.set_title("Within-subject vs cross-subject accuracy")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"methods": names, "within": within, "cross": cross}


def write_report(methods: dict, out_path: Path) -> None:
    lines = [
        "# EEG motor-imagery results",
        "",
        "Five-class problem: `rest`, `real_hands`, `real_feet`, `imagined_hands`, `imagined_feet`.",
        "Chance accuracy is **20%**.",
        "",
        "## Accuracy summary",
        "",
        "| Method | Within acc | Within F1 | LOSO acc | LOSO F1 | Gap (acc) | Time within / LOSO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    best_cross = (-1.0, "")
    for name, block in methods.items():
        w = block["within_subject"]["accuracy"]
        c = block["cross_subject"]["accuracy"]
        wf = block["within_subject"]["macro_f1"]
        cf = block["cross_subject"]["macro_f1"]
        tw = block["within_subject"].get("train_seconds")
        tc = block["cross_subject"].get("train_seconds")
        label = "EEGNet" if name == "eegnet" else name.upper()
        lines.append(
            f"| {label} | {w:.3f} | {wf:.3f} | {c:.3f} | {cf:.3f} | {w-c:.3f} | "
            f"{tw:.0f}s / {tc:.0f}s |"
        )
        if c > best_cross[0]:
            best_cross = (c, label)

    lines += [
        "",
        "## Why within-subject and cross-subject differ",
        "",
        "Motor-imagery EEG is highly **subject-specific**. Sensorimotor-rhythm (mu/beta, 8–30 Hz) "
        "topography, SNR, and even the preferred imagery strategy vary across people. CSP finds "
        "spatial filters that maximize class variance *on the training subjects*; those filters "
        "overfit individual anatomy and montage coupling, so leave-one-subject-out accuracy drops "
        "sharply versus a within-subject train/test split.",
        "",
        "A compact CNN can learn slightly more invariant spectro-spatial patterns, but with only "
        "20 subjects and no domain-alignment it still fails to close the generalization gap. "
        "That gap is the central obstacle for real-world BCI: calibration-free systems must work "
        "on unseen users, yet most published within-subject numbers overstate field performance.",
        "",
        f"On this run, **{best_cross[1]}** had the strongest cross-subject accuracy "
        f"({best_cross[0]:.3f}). Even the best LOSO score remains well below within-subject "
        "performance, which is why clinical/consumer BCIs still rely on per-user calibration "
        "or transfer-learning / Riemannian alignment rather than a single off-the-shelf model.",
        "",
        "## Class-level patterns",
        "",
        "- **Rest vs movement** is usually the easiest contrast (mu-rhythm amplitude).",
        "- **Hands vs feet** is spatially distinct (lateral vs midline motor cortex) and survives "
        "cross-subject evaluation better than execution vs imagery.",
        "- **Real vs imagined** of the same limb is the hardest pair: imagery is a weaker, noisier "
        "version of the same rhythm, so models often collapse those classes.",
        "",
        "Confusion matrices for every method/split are saved under `results/`.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", type=Path, default=config.RESULTS / "classical_metrics.json")
    parser.add_argument("--cnn", type=Path, default=config.RESULTS / "cnn_metrics.json")
    parser.add_argument("--outdir", type=Path, default=config.RESULTS)
    args = parser.parse_args()

    classical = load_json(args.classical)
    cnn = load_json(args.cnn)
    methods = flatten_methods(classical, cnn)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for name, block in methods.items():
        for split in ("within_subject", "cross_subject"):
            plot_confusion(
                block[split]["confusion_matrix"],
                title=f"{name} — {split.replace('_', ' ')}",
                out_path=args.outdir / f"cm_{name}_{split}.png",
            )

    plot_bar(methods, args.outdir / "accuracy_comparison.png")
    write_report(methods, args.outdir / "REPORT.md")

    comparison = {}
    for name, block in methods.items():
        comparison[name] = {
            "within_subject_acc": block["within_subject"]["accuracy"],
            "cross_subject_acc": block["cross_subject"]["accuracy"],
            "gap": block["within_subject"]["accuracy"] - block["cross_subject"]["accuracy"],
            "within_seconds": block["within_subject"].get("train_seconds"),
            "cross_seconds": block["cross_subject"].get("train_seconds"),
        }
    (args.outdir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print("Phase 4 plots and report written to", args.outdir)


if __name__ == "__main__":
    main()
