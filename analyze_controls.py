#!/usr/bin/env python3
"""Run compact lexical, permutation, and residual-norm controls for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "results" / "reference_run"


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "figure" / "control_stats.json")
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classifier(c: float = 0.1):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            C=c,
            max_iter=3000,
            solver="lbfgs",
            random_state=0,
        ),
    )


def grouped_scores(model, x, y: np.ndarray, groups: np.ndarray) -> list[float]:
    scores: list[float] = []
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        model.fit(x[train], y[train])
        scores.append(float(balanced_accuracy_score(y[test], model.predict(x[test]))))
    return scores


def tfidf_score(
    texts: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
) -> dict:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=True,
        sublinear_tf=True,
    )
    model = make_pipeline(
        vectorizer,
        LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=3000, random_state=0
        ),
    )
    scores = grouped_scores(model, texts, y, groups)
    return {"mean_balanced_accuracy": float(np.mean(scores)), "fold_scores": scores}


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    records = read_records(run_dir / "records.jsonl")
    with np.load(run_dir / "activations.npz") as loaded:
        activations = loaded["activations"].astype(np.float32)

    texts = np.asarray([record["statement"] for record in records])
    labels = np.asarray(
        [record["expected_label"] != "coherent" for record in records], dtype=int
    )
    groups = np.asarray([record["item_id"] for record in records])

    # Column zero is the embedding output, so transformer layer 18 is column 19.
    peak_layer = 18
    peak_column = peak_layer + 1
    observed, permutation_scores, raw_p = permutation_test_score(
        classifier(),
        activations[:, peak_column, :],
        labels,
        groups=groups,
        cv=GroupKFold(n_splits=5),
        scoring="balanced_accuracy",
        n_permutations=args.permutations,
        n_jobs=args.jobs,
        random_state=0,
    )
    # sklearn restricts label permutations to samples sharing a group identifier.
    # This preserves one coherent and four noncoherent examples in every family.
    bonferroni_p = min(1.0, float(raw_p) * activations.shape[1])

    norm_controls: dict[str, dict] = {}
    for layer in (12, 18):
        norms = np.linalg.norm(activations[:, layer + 1, :], axis=1)[:, None]
        scores = grouped_scores(classifier(c=1.0), norms, labels, groups)
        norm_controls[str(layer)] = {
            "mean_balanced_accuracy": float(np.mean(scores)),
            "fold_scores": scores,
        }

    layer12 = activations[:, 13, :].astype(np.float64)
    layer12_norm = np.linalg.norm(layer12, axis=1)
    raw_pca = PCA(n_components=2).fit(layer12)
    raw_coordinates = raw_pca.transform(layer12)
    unit_layer12 = layer12 / np.maximum(layer12_norm[:, None], 1e-12)
    unit_pca = PCA(n_components=2).fit(unit_layer12)

    selected_feature_ids = [477, 514, 941, 4660, 753, 1105, 386, 1223]
    sae_path = next(run_dir.glob("sae/*/feature_activations.npz"))
    with np.load(sae_path) as loaded:
        feature_activations = loaded["features"].astype(np.float64)
    feature_norm_correlations = {
        str(feature_id): float(
            np.corrcoef(
                np.log1p(feature_activations[:, feature_id]), layer12_norm
            )[0, 1]
        )
        for feature_id in selected_feature_ids
    }

    results = {
        "source_run": portable_path(run_dir),
        "grouping": "five-fold GroupKFold over 17 stimulus families",
        "tfidf": {
            "word_1_2gram": tfidf_score(
                texts, labels, groups, analyzer="word", ngram_range=(1, 2)
            ),
            "character_3_5gram": tfidf_score(
                texts, labels, groups, analyzer="char_wb", ngram_range=(3, 5)
            ),
        },
        "restricted_permutation": {
            "layer": peak_layer,
            "observed_balanced_accuracy": float(observed),
            "n_permutations": args.permutations,
            "raw_p_value": float(raw_p),
            "bonferroni_layers": int(activations.shape[1]),
            "bonferroni_p_value": bonferroni_p,
            "null_mean": float(np.mean(permutation_scores)),
            "null_max": float(np.max(permutation_scores)),
            "note": "Labels were shuffled only within each five-example family.",
        },
        "residual_norm_baseline": norm_controls,
        "layer12_pca_norm_diagnostic": {
            "raw_pc1_variance": float(raw_pca.explained_variance_ratio_[0]),
            "pc1_norm_pearson_r": float(
                np.corrcoef(raw_coordinates[:, 0], layer12_norm)[0, 1]
            ),
            "unit_normalized_pc1_variance": float(
                unit_pca.explained_variance_ratio_[0]
            ),
        },
        "selected_sae_feature_norm_correlations": {
            "pearson_r": feature_norm_correlations,
            "maximum_absolute_r": float(
                max(abs(value) for value in feature_norm_correlations.values())
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
