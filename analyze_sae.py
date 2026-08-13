#!/usr/bin/env python3
"""Apply an official Gemma Scope 2 SAE to extracted prompt activations.

This script does not load or modify the language model. It reads a completed
activation run, downloads/loads one pretrained SAE, encodes the matching
residual-stream layer, and ranks sparse features by label selectivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RELEASE = "gemma-scope-2-4b-it-res-all"
DEFAULT_SAE_ID = "layer_12_width_16k_l0_small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode Gemma activations with a pretrained Gemma Scope 2 SAE."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Extraction result directory. Defaults to the latest full run.",
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Transformer layer. By default it is parsed from --sae-id.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow analysis of an interrupted/smoke-test run with fewer than 85 prompts.",
    )
    return parser.parse_args()


def latest_full_run() -> Path:
    results_dir = PROJECT_DIR / "results"
    candidates: list[tuple[float, Path]] = []
    for path in results_dir.glob("gemma3_4b_*"):
        records_path = path / "records.jsonl"
        activations_path = path / "activations.npz"
        if not (records_path.is_file() and activations_path.is_file()):
            continue
        count = sum(1 for line in records_path.open(encoding="utf-8") if line.strip())
        if count == 85:
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(
            "No full 85-prompt run found. Pass a run directory explicitly, or run "
            "python extract_activations.py first."
        )
    return max(candidates)[1].resolve()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def parse_layer(sae_id: str, explicit_layer: int | None) -> int:
    if explicit_layer is not None:
        return explicit_layer
    match = re.search(r"(?:^|[/_])layer_(\d+)(?:_|$)", sae_id)
    if not match:
        raise ValueError("Could not parse a layer from --sae-id; pass --layer explicitly.")
    return int(match.group(1))


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_activations(
    sae: SAE,
    activations: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    sae_dtype = next(sae.parameters()).dtype
    for start in tqdm(range(0, len(activations), batch_size), unit="batch"):
        batch = torch.from_numpy(activations[start : start + batch_size]).to(
            device=device, dtype=sae_dtype
        )
        with torch.inference_mode():
            encoded = sae.encode(batch)
        rows.append(encoded.detach().float().cpu().numpy().astype(np.float16))
    return np.concatenate(rows, axis=0)


def contrast_rows(
    feature_acts: np.ndarray,
    labels: list[str | None],
    label_source: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank features using log-activation standardized mean differences."""
    values = np.log1p(feature_acts.astype(np.float32))
    rows: list[dict[str, Any]] = []
    available = sorted({label for label in labels if label is not None})

    for target_label in available:
        target_mask = np.array([label == target_label for label in labels])
        if target_label == "coherent":
            comparison_mask = np.array(
                [label is not None and label != "coherent" for label in labels]
            )
            comparison_label = "noncoherent"
        else:
            comparison_mask = np.array([label == "coherent" for label in labels])
            comparison_label = "coherent"

        if target_mask.sum() < 2 or comparison_mask.sum() < 2:
            continue

        target = values[target_mask]
        comparison = values[comparison_mask]
        target_mean = target.mean(axis=0)
        comparison_mean = comparison.mean(axis=0)
        pooled_scale = np.sqrt(
            (target.var(axis=0) + comparison.var(axis=0)) / 2.0
        )
        effect = (target_mean - comparison_mean) / (pooled_scale + 1e-6)

        target_prev = (feature_acts[target_mask] > 0).mean(axis=0)
        comparison_prev = (feature_acts[comparison_mask] > 0).mean(axis=0)
        eligible = (target_prev > 0) & np.isfinite(effect)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            continue
        order = indices[np.argsort(effect[indices])[::-1]][:top_k]

        for rank, feature_id in enumerate(order, start=1):
            rows.append(
                {
                    "label_source": label_source,
                    "target_label": target_label,
                    "comparison_label": comparison_label,
                    "rank": rank,
                    "feature_id": int(feature_id),
                    "effect_size": float(effect[feature_id]),
                    "target_mean_log1p": float(target_mean[feature_id]),
                    "comparison_mean_log1p": float(comparison_mean[feature_id]),
                    "target_prevalence": float(target_prev[feature_id]),
                    "comparison_prevalence": float(comparison_prev[feature_id]),
                    "target_n": int(target_mask.sum()),
                    "comparison_n": int(comparison_mask.sum()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else latest_full_run()
    records = read_records(run_dir / "records.jsonl")
    with np.load(run_dir / "activations.npz") as loaded:
        residual_activations = loaded["activations"]

    if len(records) != residual_activations.shape[0]:
        raise ValueError("records.jsonl and activations.npz have different row counts.")
    if len(records) < 85 and not args.allow_partial:
        raise ValueError(
            f"This run contains only {len(records)} prompts. Use a full run, or pass "
            "--allow-partial for a mechanical smoke test."
        )

    layer = parse_layer(args.sae_id, args.layer)
    # hidden_states[0] is the embedding output; layer L resid_post is at L + 1.
    activation_column = layer + 1
    if activation_column >= residual_activations.shape[1]:
        raise ValueError(
            f"Layer {layer} maps to activation column {activation_column}, but the "
            f"array has only {residual_activations.shape[1]} columns."
        )

    layer_activations = residual_activations[:, activation_column, :]
    device = choose_device(args.device)
    print(f"Run             : {run_dir}")
    print(f"SAE release     : {args.release}")
    print(f"SAE id          : {args.sae_id}")
    print(f"Transformer layer: {layer} (activation column {activation_column})")
    print(f"Input shape     : {layer_activations.shape}")
    print(f"Device          : {device}")
    print("The SAE may be downloaded to the Hugging Face cache on first use.")

    sae, sae_cfg, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.release,
        sae_id=args.sae_id,
    )
    sae.eval()
    sae.to(device)

    expected_d_in = int(getattr(sae.cfg, "d_in", layer_activations.shape[1]))
    if layer_activations.shape[1] != expected_d_in:
        raise ValueError(
            f"Activation width {layer_activations.shape[1]} does not match SAE d_in "
            f"{expected_d_in}. This SAE does not match the selected model/layer."
        )

    feature_acts = encode_activations(
        sae, layer_activations, device=device, batch_size=args.batch_size
    )
    output_dir = run_dir / "sae" / args.sae_id.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "feature_activations.npz", features=feature_acts)

    expected_labels = [record.get("expected_label") for record in records]
    predicted_labels = [record.get("predicted_label") for record in records]
    rankings = contrast_rows(
        feature_acts, expected_labels, label_source="expected_label", top_k=args.top_k
    )
    rankings.extend(
        contrast_rows(
            feature_acts,
            predicted_labels,
            label_source="predicted_label",
            top_k=args.top_k,
        )
    )
    write_csv(output_dir / "top_features.csv", rankings)

    active_counts = (feature_acts > 0).sum(axis=1)
    metadata = {
        "source_run": str(run_dir),
        "release": args.release,
        "sae_id": args.sae_id,
        "transformer_layer": layer,
        "activation_column": activation_column,
        "input_shape": list(layer_activations.shape),
        "feature_shape": list(feature_acts.shape),
        "mean_active_features_per_prompt": float(active_counts.mean()),
        "median_active_features_per_prompt": float(np.median(active_counts)),
        "sae_config": sae_cfg,
        "sparsity_shape": (
            list(sparsity.shape)
            if sparsity is not None and hasattr(sparsity, "shape")
            else None
        ),
        "interpretation_warning": (
            "Ranked features are correlational candidates from a small labeled set, "
            "not proven paradox or coherence detectors."
        ),
    }
    with (output_dir / "sae_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)

    print(f"Feature shape   : {feature_acts.shape}")
    print(f"Mean active     : {active_counts.mean():.1f} features/prompt")
    print(f"Ranked contrasts: {len(rankings)} rows")
    print(f"Saved to        : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
