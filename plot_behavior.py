#!/usr/bin/env python3
"""Generate the behavioral figure (manuscript Fig. 1) from a completed run.

Uses only the philosophical subset of a combined run: prompts whose item id
does not start with "mod_". The modality-set behavior appears as a table in
the manuscript and is available in the axes analysis output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent
LABELS = ["coherent", "contradiction", "paradox", "underdetermined"]
DISPLAY_LABELS = ["Coherent", "Contradiction", "Paradox", "Underdetermined"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "figure")
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13, 1.08, label, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top",
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(center - half), float(center + half)


def main() -> int:
    args = parse_args()
    configure_style()
    with (args.run_dir / "records.jsonl").open(encoding="utf-8") as handle:
        records: list[dict[str, Any]] = [
            json.loads(line) for line in handle if line.strip()
        ]
    records = [r for r in records if not r["item_id"].startswith("mod_")]
    if not records:
        raise ValueError("No philosophical prompts found in this run.")

    expected = [r["expected_label"] for r in records]
    predicted = [r["predicted_label"] for r in records]
    cm = confusion_matrix(expected, predicted, labels=LABELS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 2.85), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )
    image = ax_a.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax_a.set_xticks(range(4), DISPLAY_LABELS, rotation=28, ha="right")
    ax_a.set_yticks(range(4), DISPLAY_LABELS)
    ax_a.set_xlabel("Model prediction")
    ax_a.set_ylabel("Expected label")
    for row in range(4):
        for col in range(4):
            value = cm_norm[row, col]
            color = "white" if value > 0.55 else "#222222"
            ax_a.text(
                col, row, f"{cm[row, col]}\n{value:.0%}",
                ha="center", va="center", color=color, fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.04)
    colorbar.set_label("Row proportion")
    panel_label(ax_a, "A")

    subsets = [
        ("Overall", records),
        ("Canonical", [r for r in records if r["variant_type"] == "canonical"]),
        ("Transformation", [r for r in records if r["variant_type"] == "transformation"]),
        ("Control", [r for r in records if r["variant_type"] == "control"]),
    ]
    values, lower, upper = [], [], []
    for _, subset in subsets:
        successes = sum(bool(r["is_correct"]) for r in subset)
        value = successes / len(subset)
        low, high = wilson_interval(successes, len(subset))
        values.append(value)
        lower.append(value - low)
        upper.append(high - value)
    x = np.arange(len(subsets))
    ax_b.bar(x, values, width=0.64, color="#4C78A8", edgecolor="none")
    ax_b.errorbar(
        x, values, yerr=np.array([lower, upper]),
        fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2.5,
    )
    for xi, value, up, (_, subset) in zip(x, values, upper, subsets):
        ax_b.text(
            xi,
            value + up + 0.025,
            f"{value:.0%}\n(n={len(subset)})",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_b.axhline(0.25, color="#777777", linestyle="--", linewidth=0.9, label="4-way chance")
    ax_b.set_xticks(x, [name for name, _ in subsets], rotation=20, ha="right")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Exact-label accuracy")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(frameon=False, loc="upper right")
    panel_label(ax_b, "B")

    fig.subplots_adjust(wspace=0.42, bottom=0.22)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure1_behavior.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "figure1_behavior.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure1_behavior.[pdf,png] to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
