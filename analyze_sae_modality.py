#!/usr/bin/env python3
"""Rank SAE features by modality contrasts (impossible vs. false / possible).

Reads feature_activations.npz produced by analyze_sae.py for a combined run
and reports candidate features whose activation separates necessary falsehood
from contingent falsehood and from ordinary possible statements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "results" / "gemma3_4b_combined_20260812"
POSSIBLE = ("true", "false", "improbable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--sae-id", default="layer_15_width_16k_l0_small")
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def modality_of(record: dict[str, Any]) -> str | None:
    item_id = record["item_id"]
    if not item_id.startswith("mod_"):
        return None
    return item_id.rsplit("_", 1)[1]


def contrast(
    values: np.ndarray,
    feats: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    top_k: int,
) -> list[dict[str, float]]:
    pos, neg = values[pos_idx], values[neg_idx]
    effect = (pos.mean(0) - neg.mean(0)) / (
        np.sqrt((pos.var(0) + neg.var(0)) / 2.0) + 1e-6
    )
    pos_prev = (feats[pos_idx] > 0).mean(0)
    neg_prev = (feats[neg_idx] > 0).mean(0)
    eligible = np.flatnonzero(pos_prev >= 0.2)
    order = eligible[np.argsort(effect[eligible])[::-1]][:top_k]
    return [
        {
            "feature_id": int(f),
            "effect_size": float(effect[f]),
            "pos_prevalence": float(pos_prev[f]),
            "neg_prevalence": float(neg_prev[f]),
            "pos_mean_log1p": float(pos[:, f].mean()),
            "neg_mean_log1p": float(neg[:, f].mean()),
        }
        for f in order
    ]


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    sae_dir = run_dir / "sae" / args.sae_id
    with np.load(sae_dir / "feature_activations.npz") as loaded:
        feats = loaded["features"].astype(np.float32)

    modalities = [modality_of(r) for r in records]
    idx = {
        m: np.array([i for i, mm in enumerate(modalities) if mm == m])
        for m in ("true", "false", "improbable", "anomalous", "impossible")
    }
    phil_noncoherent = np.array(
        [
            i
            for i, (m, r) in enumerate(zip(modalities, records))
            if m is None and r["expected_label"] != "coherent"
        ]
    )
    values = np.log1p(feats)

    results = {
        "sae_id": args.sae_id,
        "impossible_vs_false": contrast(
            values, feats, idx["impossible"], idx["false"], args.top_k
        ),
        "impossible_vs_possible": contrast(
            values,
            feats,
            idx["impossible"],
            np.concatenate([idx[m] for m in POSSIBLE]),
            args.top_k,
        ),
        "false_vs_true": contrast(
            values, feats, idx["false"], idx["true"], args.top_k
        ),
    }

    # For the strongest impossible-selective features, report where else they fire.
    top_ids = [row["feature_id"] for row in results["impossible_vs_possible"][:6]]
    profile = {}
    for f in top_ids:
        profile[f] = {
            m: float((feats[idx[m], f] > 0).mean()) for m in idx
        }
        profile[f]["philosophical_noncoherent"] = float(
            (feats[phil_noncoherent, f] > 0).mean()
        )
    results["top_impossible_feature_prevalence"] = profile

    out_path = sae_dir / "modality_feature_contrasts.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    for name in ("impossible_vs_false", "impossible_vs_possible", "false_vs_true"):
        print(f"\n== {name} ==")
        for row in results[name][:8]:
            print(
                f"  f{row['feature_id']:>6d}  d={row['effect_size']:+.2f}  "
                f"prev {row['pos_prevalence']:.2f} vs {row['neg_prevalence']:.2f}"
            )
    print("\n== prevalence profile of top impossible-selective features ==")
    for f, prof in results["top_impossible_feature_prevalence"].items():
        print(f"  f{f}: " + "  ".join(f"{k}={v:.2f}" for k, v in prof.items()))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
