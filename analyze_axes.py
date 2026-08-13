#!/usr/bin/env python3
"""Probe geometry of truth, improbability, anomaly, and impossibility axes.

Reads a combined extraction run (philosophical families + modality contrast
set) and asks whether the direction that separates impossible statements from
possible ones is the same direction that separates false statements from true
ones, anomalous from ordinary ones, and incoherent philosophical stimuli from
their coherent controls.

All probe evaluations hold out whole stimulus families. Outputs are written to
<run_dir>/axes/ and figures to figure/.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_DIR / "figure"
DEFAULT_RUN = PROJECT_DIR / "results" / "gemma3_4b_combined_20260812"
LEGEND_KW = {
    "frameon": False,
    "fontsize": 6.5,
    "handlelength": 1.4,
    "labelspacing": 0.3,
    "borderaxespad": 0.2,
}
SEED = 20260812
MODALITIES = ("true", "false", "improbable", "anomalous", "impossible")
POSSIBLE = ("true", "false", "improbable")

AXES = {
    # name: (positive class selector, negative class selector, dataset)
    "impossibility": ("impossible", POSSIBLE),
    "truth": ("false", ("true",)),
    "anomaly": ("anomalous", POSSIBLE),
}

TEST_CONTRASTS = {
    "false_vs_true": ("false", ("true",)),
    "impossible_vs_true": ("impossible", ("true",)),
    "impossible_vs_false": ("impossible", ("false",)),
    "impossible_vs_possible": ("impossible", POSSIBLE),
    "anomalous_vs_true": ("anomalous", ("true",)),
    "impossible_vs_anomalous": ("impossible", ("anomalous",)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate the figure from an existing axes_results.json without recomputing.",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_probe() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED
        ),
    )


def modality_of(record: dict[str, Any]) -> str | None:
    item_id = record["item_id"]
    if not item_id.startswith("mod_"):
        return None
    return item_id.rsplit("_", 1)[1]


def family_of(record: dict[str, Any]) -> str:
    item_id = record["item_id"]
    if item_id.startswith("mod_"):
        return item_id.split("_")[1]
    return item_id


def held_out_axis_curves(
    states: np.ndarray,
    records: list[dict[str, Any]],
    mod_index: dict[str, np.ndarray],
    families: np.ndarray,
) -> dict[str, Any]:
    """Group-held-out balanced accuracy per layer for each axis, plus transfer AUCs."""
    n_layers = states.shape[1]
    curves: dict[str, list[float]] = {name: [] for name in AXES}
    curve_se: dict[str, list[float]] = {name: [] for name in AXES}
    transfer: dict[str, dict[str, list[float]]] = {
        name: {target: [] for target in TEST_CONTRASTS} for name in AXES
    }

    for layer in tqdm(range(n_layers), unit="layer", desc="axis probes"):
        for axis_name, (pos, neg) in AXES.items():
            pos_idx = mod_index[pos]
            neg_idx = np.concatenate([mod_index[m] for m in neg])
            idx = np.concatenate([pos_idx, neg_idx])
            y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
            groups = families[idx]
            x = states[idx, layer, :]

            fold_scores: list[float] = []
            target_scores: dict[str, list[float]] = {t: [] for t in TEST_CONTRASTS}
            splitter = GroupKFold(n_splits=5)
            for train_rows, test_rows in splitter.split(x, y, groups):
                probe = make_probe()
                probe.fit(x[train_rows], y[train_rows])
                fold_scores.append(
                    balanced_accuracy_score(
                        y[test_rows], probe.predict(x[test_rows])
                    )
                )
                held_families = set(families[idx][test_rows])
                for target, (tpos, tneg) in TEST_CONTRASTS.items():
                    tpos_idx = [
                        i
                        for i in mod_index[tpos]
                        if families[i] in held_families
                    ]
                    tneg_idx = [
                        i
                        for m in tneg
                        for i in mod_index[m]
                        if families[i] in held_families
                    ]
                    if not tpos_idx or not tneg_idx:
                        continue
                    all_idx = np.array(tpos_idx + tneg_idx)
                    ty = np.concatenate(
                        [np.ones(len(tpos_idx)), np.zeros(len(tneg_idx))]
                    )
                    scores = probe.decision_function(states[all_idx, layer, :])
                    target_scores[target].append(roc_auc_score(ty, scores))

            curves[axis_name].append(float(np.mean(fold_scores)))
            curve_se[axis_name].append(
                float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
            )
            for target in TEST_CONTRASTS:
                transfer[axis_name][target].append(
                    float(np.mean(target_scores[target]))
                    if target_scores[target]
                    else float("nan")
                )

    return {"balanced_accuracy": curves, "standard_error": curve_se, "transfer_auc": transfer}


def cross_dataset_transfer(
    states: np.ndarray,
    mod_index: dict[str, np.ndarray],
    phil_idx: np.ndarray,
    phil_y: np.ndarray,
) -> dict[str, list[float]]:
    """Train on one dataset, test on the other; disjoint stimuli by construction."""
    n_layers = states.shape[1]
    pos_idx = mod_index["impossible"]
    neg_idx = np.concatenate([mod_index[m] for m in POSSIBLE])
    mod_idx = np.concatenate([pos_idx, neg_idx])
    mod_y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

    out: dict[str, list[float]] = {
        "modality_to_philosophical_auc": [],
        "philosophical_to_modality_auc": [],
    }
    for layer in tqdm(range(n_layers), unit="layer", desc="cross-dataset"):
        probe = make_probe()
        probe.fit(states[mod_idx, layer, :], mod_y)
        scores = probe.decision_function(states[phil_idx, layer, :])
        out["modality_to_philosophical_auc"].append(
            float(roc_auc_score(phil_y, scores))
        )

        probe = make_probe()
        probe.fit(states[phil_idx, layer, :], phil_y)
        scores = probe.decision_function(states[mod_idx, layer, :])
        out["philosophical_to_modality_auc"].append(
            float(roc_auc_score(mod_y, scores))
        )
    return out


def axis_cosines(
    states: np.ndarray, mod_index: dict[str, np.ndarray]
) -> dict[str, list[float]]:
    """Cosine similarity between full-data axis directions at every layer."""
    n_layers = states.shape[1]
    pairs = [("truth", "impossibility"), ("anomaly", "impossibility"), ("truth", "anomaly")]
    out: dict[str, list[float]] = {f"{a}·{b}": [] for a, b in pairs}
    for layer in tqdm(range(n_layers), unit="layer", desc="axis cosines"):
        weights: dict[str, np.ndarray] = {}
        for axis_name, (pos, neg) in AXES.items():
            pos_idx = mod_index[pos]
            neg_idx = np.concatenate([mod_index[m] for m in neg])
            idx = np.concatenate([pos_idx, neg_idx])
            y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
            probe = make_probe()
            probe.fit(states[idx, layer, :], y)
            weights[axis_name] = probe.named_steps["logisticregression"].coef_[0]
        for a, b in pairs:
            wa, wb = weights[a], weights[b]
            out[f"{a}·{b}"].append(
                float(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
            )
    return out


def surface_baselines(
    records: list[dict[str, Any]],
    mod_index: dict[str, np.ndarray],
    families: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Token-count and TF-IDF probes for the two contrasts that carry the story."""
    contrasts = {
        "impossible_vs_possible": ("impossible", POSSIBLE),
        "impossible_vs_false": ("impossible", ("false",)),
    }
    statements = np.array([r["statement"] for r in records], dtype=object)
    token_counts = np.array([[r["input_token_count"]] for r in records], dtype=float)

    out: dict[str, dict[str, float]] = {}
    for cname, (pos, neg) in contrasts.items():
        pos_idx = mod_index[pos]
        neg_idx = np.concatenate([mod_index[m] for m in neg])
        idx = np.concatenate([pos_idx, neg_idx])
        y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
        groups = families[idx]
        scores: dict[str, list[float]] = {"token_count": [], "tfidf_word": [], "tfidf_char": []}
        splitter = GroupKFold(n_splits=5)
        for train_rows, test_rows in splitter.split(idx, y, groups):
            probe = make_probe()
            probe.fit(token_counts[idx][train_rows], y[train_rows])
            scores["token_count"].append(
                balanced_accuracy_score(
                    y[test_rows], probe.predict(token_counts[idx][test_rows])
                )
            )
            for key, vectorizer in (
                ("tfidf_word", TfidfVectorizer(ngram_range=(1, 2))),
                (
                    "tfidf_char",
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
                ),
            ):
                train_x = vectorizer.fit_transform(statements[idx][train_rows])
                test_x = vectorizer.transform(statements[idx][test_rows])
                clf = LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED
                )
                clf.fit(train_x, y[train_rows])
                scores[key].append(
                    balanced_accuracy_score(y[test_rows], clf.predict(test_x))
                )
        out[cname] = {key: float(np.mean(vals)) for key, vals in scores.items()}
    return out


def within_family_permutation(
    states: np.ndarray,
    mod_index: dict[str, np.ndarray],
    families: np.ndarray,
    layer: int,
    pos: str,
    neg: tuple[str, ...],
    n_permutations: int,
) -> float:
    """Permutation P for a group-held-out probe, shuffling labels within families."""
    pos_idx = mod_index[pos]
    neg_idx = np.concatenate([mod_index[m] for m in neg])
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    groups = families[idx]
    x = states[idx, layer, :]

    def cv_score(labels: np.ndarray) -> float:
        fold_scores = []
        splitter = GroupKFold(n_splits=5)
        for train_rows, test_rows in splitter.split(x, labels, groups):
            probe = make_probe()
            probe.fit(x[train_rows], labels[train_rows])
            fold_scores.append(
                balanced_accuracy_score(labels[test_rows], probe.predict(x[test_rows]))
            )
        return float(np.mean(fold_scores))

    observed = cv_score(y)
    rng = np.random.default_rng(SEED)
    exceed = 0
    for _ in tqdm(range(n_permutations), unit="perm", desc=f"permutation L{layer}"):
        permuted = y.copy()
        for family in np.unique(groups):
            rows = np.flatnonzero(groups == family)
            permuted[rows] = rng.permutation(permuted[rows])
        if cv_score(permuted) >= observed:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)


def behavior_table(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = {m: Counter() for m in MODALITIES}
    for record in records:
        modality = modality_of(record)
        if modality is not None:
            table[modality][record.get("predicted_label") or "unparsed"] += 1
    return {m: dict(c) for m, c in table.items()}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_figure(
    curves: dict[str, Any],
    cross: dict[str, list[float]],
    cosines: dict[str, list[float]],
    peak_layer: int,
    output_dir: Path,
    dpi: int,
) -> None:
    configure_style()
    n_layers = len(curves["balanced_accuracy"]["impossibility"])
    depth = np.arange(n_layers)
    tick_positions = [0, 4, 8, 12, 16, 20, 24, 28, 34]
    tick_labels = ["Emb."] + [str(t - 1) for t in tick_positions[1:]]

    fig, axes_row = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    ax = axes_row[0]
    colors = {"impossibility": "#0072B2", "truth": "#D55E00", "anomaly": "#009E73"}
    labels = {
        "impossibility": "Impossible vs. possible",
        "truth": "False vs. true",
        "anomaly": "Anomalous vs. ordinary",
    }
    for axis_name, color in colors.items():
        mean = np.array(curves["balanced_accuracy"][axis_name])
        se = np.array(curves["standard_error"][axis_name])
        ax.plot(depth, mean, color=color, label=labels[axis_name])
        ax.fill_between(depth, mean - se, mean + se, color=color, alpha=0.18, lw=0)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Representation depth")
    ax.set_ylabel("Held-out balanced accuracy")
    ax.legend(loc="lower right", **LEGEND_KW)
    ax.set_title("a", loc="left", fontweight="bold")

    ax = axes_row[1]
    transfer = curves["transfer_auc"]["truth"]
    ax.plot(
        depth,
        transfer["false_vs_true"],
        color="#D55E00",
        label="False vs. true (in-axis)",
    )
    ax.plot(
        depth,
        transfer["impossible_vs_true"],
        color="#0072B2",
        label="Impossible vs. true",
    )
    ax.plot(
        depth,
        transfer["impossible_vs_false"],
        color="#CC79A7",
        label="Impossible vs. false",
    )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Representation depth")
    ax.set_ylabel("Transfer AUC of truth probe")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.66), **LEGEND_KW)
    ax.set_title("b", loc="left", fontweight="bold")

    ax = axes_row[2]
    ax.plot(
        depth,
        cross["modality_to_philosophical_auc"],
        color="#0072B2",
        label="Modality → philosophical",
    )
    ax.plot(
        depth,
        cross["philosophical_to_modality_auc"],
        color="#D55E00",
        label="Philosophical → modality",
    )
    ax.plot(
        depth,
        cosines["truth·impossibility"],
        color="#CC79A7",
        ls=":",
        label="cos(truth, impossibility)",
    )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0.0, color="gray", ls="-", lw=0.5)
    ax.axvline(peak_layer, color="gray", ls="-.", lw=0.8)
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel("Representation depth")
    ax.set_ylabel("Cross-dataset AUC / cosine")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.45), **LEGEND_KW)
    ax.set_title("c", loc="left", fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"figure5_axes.{suffix}", dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if args.replot_only:
        with (run_dir / "axes" / "axes_results.json").open(encoding="utf-8") as handle:
            results = json.load(handle)
        plot_figure(
            results["curves"],
            results["cross_dataset"],
            results["axis_cosines"],
            results["peak_layer_impossibility"],
            FIGURE_DIR,
            args.dpi,
        )
        print(f"Replotted figure from {run_dir / 'axes' / 'axes_results.json'}")
        return 0
    records = read_records(run_dir / "records.jsonl")
    with np.load(run_dir / "activations.npz") as loaded:
        states = loaded["activations"].astype(np.float32)
    if len(records) != states.shape[0]:
        raise ValueError("records.jsonl and activations.npz row counts differ.")

    families = np.array([family_of(r) for r in records], dtype=object)
    modalities = [modality_of(r) for r in records]
    mod_index = {
        m: np.array([i for i, mm in enumerate(modalities) if mm == m])
        for m in MODALITIES
    }
    phil_idx = np.array([i for i, mm in enumerate(modalities) if mm is None])
    phil_y = np.array(
        [0 if records[i]["expected_label"] == "coherent" else 1 for i in phil_idx]
    )
    print(f"Run                 : {run_dir}")
    print(f"Prompts             : {len(records)} "
          f"(philosophical {len(phil_idx)}, modality {sum(len(v) for v in mod_index.values())})")

    curves = held_out_axis_curves(states, records, mod_index, families)
    cross = cross_dataset_transfer(states, mod_index, phil_idx, phil_y)
    cosines = axis_cosines(states, mod_index)
    baselines = surface_baselines(records, mod_index, families)
    behavior = behavior_table(records)

    imp_curve = np.array(curves["balanced_accuracy"]["impossibility"])
    peak_layer = int(imp_curve.argmax())
    print(f"Impossibility peak  : depth {peak_layer} "
          f"(balanced accuracy {imp_curve[peak_layer]:.3f})")

    p_impossible = within_family_permutation(
        states, mod_index, families, peak_layer, "impossible", POSSIBLE,
        args.permutations,
    )
    n_layers = states.shape[1]

    results = {
        "run_dir": str(run_dir),
        "peak_layer_impossibility": peak_layer,
        "peak_balanced_accuracy_impossibility": float(imp_curve[peak_layer]),
        "permutation_p_impossibility_at_peak": p_impossible,
        "bonferroni_p_over_layers": min(1.0, p_impossible * n_layers),
        "curves": curves,
        "cross_dataset": cross,
        "axis_cosines": cosines,
        "surface_baselines": baselines,
        "behavior_by_modality": behavior,
    }
    output_dir = run_dir / "axes"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "axes_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    plot_figure(curves, cross, cosines, peak_layer, FIGURE_DIR, args.dpi)

    print(f"Permutation P (peak): {p_impossible:.4f} "
          f"(Bonferroni x{n_layers}: {min(1.0, p_impossible * n_layers):.4f})")
    print(f"Saved               : {output_dir / 'axes_results.json'}")
    print(f"Figure              : {FIGURE_DIR / 'figure5_axes.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
