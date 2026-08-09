#!/usr/bin/env python3
"""Generate publication-style figures from a completed activation + SAE run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path.resolve())


FIGURE_DIR = PROJECT_DIR / "figure"
LABELS = ["coherent", "contradiction", "paradox", "underdetermined"]
DISPLAY_LABELS = ["Coherent", "Contradiction", "Paradox", "Underdetermined"]
COLORS = {
    "coherent": "#0072B2",
    "contradiction": "#D55E00",
    "paradox": "#CC79A7",
    "underdetermined": "#009E73",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready project figures.")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Completed result directory; defaults to the latest full run with SAE output.",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def latest_complete_run() -> Path:
    candidates: list[tuple[float, Path]] = []
    for run in (PROJECT_DIR / "results").glob("gemma3_4b_*"):
        records = run / "records.jsonl"
        activations = run / "activations.npz"
        sae_files = list(run.glob("sae/*/feature_activations.npz"))
        if not (records.is_file() and activations.is_file() and sae_files):
            continue
        count = sum(1 for line in records.open(encoding="utf-8") if line.strip())
        if count == 85:
            candidates.append((run.stat().st_mtime, run))
    if not candidates:
        raise FileNotFoundError("No complete 85-prompt run with SAE output was found.")
    return max(candidates)[1].resolve()


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(center - half), float(center + half)


def make_behavior_figure(
    records: list[dict[str, Any]], output_dir: Path, dpi: int
) -> dict[str, Any]:
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
                col,
                row,
                f"{cm[row, col]}\n{value:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
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
        x,
        values,
        yerr=np.array([lower, upper]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
    )
    for xi, value, (_, subset) in zip(x, values, subsets):
        ax_b.text(xi, value + 0.055, f"{value:.0%}\n(n={len(subset)})", ha="center", fontsize=7)
    ax_b.axhline(0.25, color="#777777", linestyle="--", linewidth=0.9, label="4-way chance")
    ax_b.set_xticks(x, [name for name, _ in subsets], rotation=20, ha="right")
    ax_b.set_ylim(0, 0.87)
    ax_b.set_ylabel("Exact-label accuracy")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(frameon=False, loc="upper right")
    panel_label(ax_b, "B")

    fig.subplots_adjust(wspace=0.42, bottom=0.22)
    save_figure(fig, output_dir, "figure1_behavior", dpi)
    return {
        "four_way_accuracy": float(np.mean([r["is_correct"] for r in records])),
        "confusion_matrix": cm.tolist(),
        "predicted_counts": dict(Counter(predicted)),
    }


def probe_curve(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv = GroupKFold(n_splits=5)
    fold_scores = np.empty((activations.shape[1], 5), dtype=np.float32)
    for column in range(activations.shape[1]):
        x = activations[:, column, :]
        for fold, (train, test) in enumerate(cv.split(x, labels, groups)):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    C=0.1,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=0,
                ),
            )
            classifier.fit(x[train], labels[train])
            fold_scores[column, fold] = balanced_accuracy_score(
                labels[test], classifier.predict(x[test])
            )
    mean = fold_scores.mean(axis=1)
    sem = fold_scores.std(axis=1, ddof=1) / np.sqrt(fold_scores.shape[1])
    return mean, sem, fold_scores


def token_length_baseline(
    records: list[dict[str, Any]], labels: np.ndarray, groups: np.ndarray
) -> float:
    x = np.array([[r["input_token_count"]] for r in records], dtype=np.float32)
    scores = []
    for train, test in GroupKFold(n_splits=5).split(x, labels, groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=0),
        )
        classifier.fit(x[train], labels[train])
        scores.append(balanced_accuracy_score(labels[test], classifier.predict(x[test])))
    return float(np.mean(scores))


def make_probe_figure(
    records: list[dict[str, Any]],
    activations: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    groups = np.array([r["item_id"] for r in records])
    expected_binary = np.array([r["expected_label"] != "coherent" for r in records], dtype=int)
    predicted_binary = np.array([r["predicted_label"] != "coherent" for r in records], dtype=int)
    expected_mean, expected_sem, expected_folds = probe_curve(
        activations, expected_binary, groups
    )
    predicted_mean, predicted_sem, predicted_folds = probe_curve(
        activations, predicted_binary, groups
    )
    length_baseline = token_length_baseline(records, expected_binary, groups)

    # Column 0 is the embedding output; columns 1..34 are transformer layers 0..33.
    layer_axis = np.arange(-1, activations.shape[1] - 1)
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    ax.plot(
        layer_axis,
        expected_mean,
        color="#0072B2",
        marker="o",
        markersize=2.8,
        label="Expected coherence",
    )
    ax.fill_between(
        layer_axis,
        expected_mean - expected_sem,
        expected_mean + expected_sem,
        color="#0072B2",
        alpha=0.17,
        linewidth=0,
    )
    ax.plot(
        layer_axis,
        predicted_mean,
        color="#D55E00",
        marker="s",
        markersize=2.5,
        label="Model's coherence judgment",
    )
    ax.fill_between(
        layer_axis,
        predicted_mean - predicted_sem,
        predicted_mean + predicted_sem,
        color="#D55E00",
        alpha=0.15,
        linewidth=0,
    )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=0.8, label="Chance")
    ax.axhline(
        length_baseline,
        color="#777777",
        linestyle=":",
        linewidth=1.0,
        label=f"Token-count baseline ({length_baseline:.2f})",
    )
    ax.axvline(12, color="#999999", linestyle="-.", linewidth=0.9, label="SAE layer (12)")
    ax.set_xlim(-1.5, 33.5)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks([-1, 4, 8, 12, 16, 20, 24, 28, 33])
    ax.set_xticklabels(["Emb.", "4", "8", "12", "16", "20", "24", "28", "33"])
    ax.set_xlabel("Representation depth")
    ax.set_ylabel("Group-held-out balanced accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    expected_best = int(np.argmax(expected_mean))
    predicted_best = int(np.argmax(predicted_mean))
    for column, mean, color in [
        (expected_best, expected_mean, "#0072B2"),
        (predicted_best, predicted_mean, "#D55E00"),
    ]:
        ax.annotate(
            f"{mean[column]:.2f}",
            (layer_axis[column], mean[column]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=7,
        )
    fig.tight_layout()
    save_figure(fig, output_dir, "figure2_layerwise_probe", dpi)

    np.savez_compressed(
        output_dir / "figure2_probe_data.npz",
        layer_axis=layer_axis,
        expected_fold_scores=expected_folds,
        predicted_fold_scores=predicted_folds,
        token_length_baseline=length_baseline,
    )
    return {
        "expected_best_layer": int(layer_axis[expected_best]),
        "expected_best_balanced_accuracy": float(expected_mean[expected_best]),
        "predicted_best_layer": int(layer_axis[predicted_best]),
        "predicted_best_balanced_accuracy": float(predicted_mean[predicted_best]),
        "token_length_baseline": length_baseline,
    }


def make_pca_figure(
    records: list[dict[str, Any]],
    activations: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    layer_columns = [1, 13, 19, 34]
    layer_names = ["Layer 0", "Layer 12", "Layer 18", "Layer 33"]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.35))
    explained: dict[str, list[float]] = {}

    for panel_index, (ax, column, layer_name) in enumerate(
        zip(axes, layer_columns, layer_names)
    ):
        pca = PCA(n_components=2, random_state=0)
        coordinates = pca.fit_transform(activations[:, column, :])
        explained[layer_name] = pca.explained_variance_ratio_.tolist()
        for label in LABELS:
            indices = [i for i, r in enumerate(records) if r["expected_label"] == label]
            correct = [i for i in indices if records[i]["is_correct"]]
            incorrect = [i for i in indices if not records[i]["is_correct"]]
            if correct:
                ax.scatter(
                    coordinates[correct, 0],
                    coordinates[correct, 1],
                    s=15,
                    color=COLORS[label],
                    alpha=0.78,
                    linewidth=0.35,
                    edgecolor="white",
                )
            if incorrect:
                ax.scatter(
                    coordinates[incorrect, 0],
                    coordinates[incorrect, 1],
                    s=19,
                    color=COLORS[label],
                    marker="x",
                    alpha=0.85,
                    linewidth=0.8,
                )
        variance = 100 * pca.explained_variance_ratio_.sum()
        ax.set_title(f"{layer_name}\nPC1+PC2: {variance:.1f}%")
        ax.set_xlabel("PC1")
        if panel_index == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
        panel_label(ax, chr(ord("A") + panel_index))

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS[label],
            markeredgecolor="none",
            markersize=5,
            label=display,
        )
        for label, display in zip(LABELS, DISPLAY_LABELS)
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="#555555", linestyle="none", markersize=4, label="Correct"),
        Line2D([0], [0], marker="x", color="#555555", linestyle="none", markersize=5, label="Incorrect"),
    ]
    fig.legend(
        handles=class_handles + status_handles,
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.25)
    save_figure(fig, output_dir, "figure3_representation_pca", dpi)
    return {"explained_variance_ratio": explained}


def select_sae_features(
    feature_acts: np.ndarray, records: list[dict[str, Any]], per_label: int = 2
) -> tuple[list[int], dict[int, str], np.ndarray, np.ndarray]:
    labels = np.array([r["expected_label"] for r in records])
    logged = np.log1p(feature_acts.astype(np.float32))
    class_means = np.stack([logged[labels == label].mean(axis=0) for label in LABELS])
    prevalence = np.stack([(feature_acts[labels == label] > 0).mean(axis=0) for label in LABELS])
    selected: list[int] = []
    selected_for: dict[int, str] = {}

    feature_mean = class_means.mean(axis=0)
    feature_std = class_means.std(axis=0)
    specificity = (class_means - feature_mean) / (feature_std + 1e-6)
    winning_class = class_means.argmax(axis=0)

    for label_index, label in enumerate(LABELS):
        eligible = (
            (prevalence[label_index] >= 0.05)
            & (winning_class == label_index)
            & np.isfinite(specificity[label_index])
        )
        candidates = np.flatnonzero(eligible)
        candidates = candidates[np.argsort(specificity[label_index, candidates])[::-1]]
        added = 0
        for feature_id in candidates:
            feature_id = int(feature_id)
            if feature_id in selected:
                continue
            selected.append(feature_id)
            selected_for[feature_id] = label
            added += 1
            if added == per_label:
                break
    return selected, selected_for, class_means, prevalence


def make_sae_figure(
    records: list[dict[str, Any]],
    feature_acts: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    selected, selected_for, class_means, prevalence = select_sae_features(
        feature_acts, records
    )
    mean_matrix = class_means[:, selected].T
    row_mean = mean_matrix.mean(axis=1, keepdims=True)
    row_std = mean_matrix.std(axis=1, keepdims=True)
    standardized = (mean_matrix - row_mean) / (row_std + 1e-6)
    prevalence_matrix = prevalence[:, selected].T
    row_labels = [f"f{feature_id} ({selected_for[feature_id][:4]}.)" for feature_id in selected]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 4.25), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    image_a = ax_a.imshow(standardized, cmap="RdBu_r", vmin=-1.75, vmax=1.75, aspect="auto")
    ax_a.set_xticks(range(4), DISPLAY_LABELS, rotation=28, ha="right")
    ax_a.set_yticks(range(len(selected)), row_labels)
    ax_a.set_title("Relative mean activation")
    ax_a.set_xlabel("Expected label")
    ax_a.set_ylabel("SAE feature (selection class)")
    for row in range(len(selected)):
        for col in range(4):
            value = standardized[row, col]
            ax_a.text(
                col,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if abs(value) > 1.05 else "#222222",
            )
    fig.colorbar(image_a, ax=ax_a, fraction=0.046, pad=0.04)
    panel_label(ax_a, "A")

    image_b = ax_b.imshow(prevalence_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_b.set_xticks(range(4), DISPLAY_LABELS, rotation=28, ha="right")
    ax_b.set_yticks([])
    ax_b.set_title("Activation prevalence")
    ax_b.set_xlabel("Expected label")
    for row in range(len(selected)):
        for col in range(4):
            value = prevalence_matrix[row, col]
            ax_b.text(
                col,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if value < 0.28 or value > 0.72 else "#222222",
            )
    fig.colorbar(image_b, ax=ax_b, fraction=0.046, pad=0.04)
    panel_label(ax_b, "B")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.16, wspace=0.26)
    save_figure(fig, output_dir, "figure4_sae_features", dpi)
    return {
        "selected_feature_ids": selected,
        "selection_labels": {str(key): value for key, value in selected_for.items()},
        "mean_active_features_per_prompt": float((feature_acts > 0).sum(axis=1).mean()),
        "selection_note": "Features were selected and displayed on the same exploratory dataset.",
    }


def write_captions(output_dir: Path, stats: dict[str, Any]) -> None:
    text = f"""# Figure captions

## Figure 1. Behavioral classification of philosophical stimuli

**(A)** Row-normalized confusion matrix for Gemma 3 4B IT on 85 stimuli. Each
cell reports count and row percentage. **(B)** Exact four-way classification
accuracy overall and by stimulus form; error bars are Wilson 95% confidence
intervals. The model achieved {stats['behavior']['four_way_accuracy']:.1%}
overall accuracy and showed a strong tendency to use the `paradox` label.

## Figure 2. Layer-wise decodability of coherence

Five-fold group-held-out linear probes distinguish coherent from noncoherent
stimuli. All variants derived from one seed item were assigned to the same fold.
Blue uses the expected binary label; orange uses the model's eventual binary
judgment. Shading denotes the standard error across folds. The dotted baseline
uses input token count alone. Expected coherence peaked at transformer layer
{stats['probe']['expected_best_layer']} (balanced accuracy
{stats['probe']['expected_best_balanced_accuracy']:.2f}); the model's own
judgment peaked at layer {stats['probe']['predicted_best_layer']}
({stats['probe']['predicted_best_balanced_accuracy']:.2f}).

## Figure 3. Evolution of residual-stream geometry

Two-dimensional PCA of the final prompt-token representation at four depths.
Color denotes the expected label; circles and crosses denote correct and
incorrect four-way model classifications. PCA is fit independently at each
layer and is descriptive rather than a test of separability.

## Figure 4. Label-selective Gemma Scope 2 SAE features

Exploratory sparse-feature analysis at transformer layer 12 using the official
16k-feature Gemma Scope 2 residual-stream SAE. Two candidate features were
selected per expected class. **(A)** Mean log-activation, standardized within
each feature across classes. **(B)** Fraction of prompts on which each feature
was active. Feature selection and visualization use the same small dataset;
these features are correlational candidates, not established semantic detectors.
"""
    (output_dir / "captions.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_style()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else latest_complete_run()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(run_dir / "records.jsonl")
    with np.load(run_dir / "activations.npz") as loaded:
        activations = loaded["activations"].astype(np.float32)
    sae_paths = sorted(run_dir.glob("sae/*/feature_activations.npz"))
    if not sae_paths:
        raise FileNotFoundError(f"No SAE feature file found under {run_dir}")
    with np.load(sae_paths[0]) as loaded:
        feature_acts = loaded["features"].astype(np.float32)

    if not (len(records) == len(activations) == len(feature_acts)):
        raise ValueError("Record, residual activation, and SAE feature row counts differ.")

    print(f"Run        : {run_dir}")
    print(f"Prompts    : {len(records)}")
    print(f"Activations: {activations.shape}")
    print(f"SAE        : {feature_acts.shape}")
    print(f"Output     : {output_dir}")

    stats = {
        "source_run": portable_path(run_dir),
        "behavior": make_behavior_figure(records, output_dir, args.dpi),
        "probe": make_probe_figure(records, activations, output_dir, args.dpi),
        "pca": make_pca_figure(records, activations, output_dir, args.dpi),
        "sae": make_sae_figure(records, feature_acts, output_dir, args.dpi),
    }
    with (output_dir / "figure_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    write_captions(output_dir, stats)
    print("Generated four figures as PDF and 400-dpi PNG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
