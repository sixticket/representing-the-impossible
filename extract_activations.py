#!/usr/bin/env python3
"""Ask Gemma 3 the seed questions and extract layer-wise hidden states.

The model can be loaded by Hugging Face model ID or from a local snapshot.
For each stimulus, the script stores the hidden state at the final prompt token,
immediately before answer generation. Row i in activations.npz corresponds to
the record whose activation_index is i in records.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = PROJECT_DIR / "data" / "questions_philosophical.json"
DEFAULT_MODEL = "google/gemma-3-4b-it"
LABELS = ("coherent", "contradiction", "paradox", "underdetermined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run philosophical stimuli through Gemma 3 and save activations."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID, cache directory, or resolved snapshot directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable downloads; use this with an existing local model cache or snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/gemma3_4b_<timestamp>.",
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N stimuli; useful for a smoke test.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def resolve_snapshot(path: Path) -> Path:
    """Resolve a Hugging Face cache root to a concrete local snapshot."""
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path

    ref = path / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        candidate = path / "snapshots" / revision
        if (candidate / "config.json").is_file():
            return candidate.resolve()

    snapshots = path / "snapshots"
    candidates = sorted(
        (p for p in snapshots.glob("*") if (p / "config.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError(f"No complete local model snapshot found under: {path}")


def resolve_model_source(value: str) -> str | Path:
    """Return a model ID unchanged, or resolve a local cache path to a snapshot."""
    candidate = Path(value).expanduser()
    if candidate.exists():
        return resolve_snapshot(candidate)
    return value


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


def choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[requested]
    if device.type in {"mps", "cuda"}:
        return torch.bfloat16
    return torch.float32


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a repository-relative path when the file lives in this project."""
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path.resolve())


def load_stimuli(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    template = dataset["prompt_template"]
    stimuli: list[dict[str, Any]] = []

    for item in dataset["items"]:
        variants = [("canonical", 0, item["statement_en"], item["expected_label"])]
        variants.extend(
            ("transformation", index, text, item["expected_label"])
            for index, text in enumerate(item.get("transformations", []), start=1)
        )
        if "coherent_control" in item:
            variants.append(
                (
                    "control",
                    0,
                    item["coherent_control"],
                    item.get("control_expected_label", "coherent"),
                )
            )

        for variant_type, variant_index, statement, expected_label in variants:
            stimuli.append(
                {
                    "item_id": item["id"],
                    "name": item["name"],
                    "category": item["category"],
                    "variant_type": variant_type,
                    "variant_index": variant_index,
                    "statement": statement,
                    "expected_label": expected_label,
                    "prompt": template.format(statement=statement),
                }
            )

    return stimuli, dataset["metadata"]


def parse_predicted_label(response: str) -> str | None:
    match = re.search(
        r"\b(coherent|contradiction|paradox|underdetermined)\b",
        response.lower(),
    )
    return match.group(1) if match else None


def model_inputs_for_prompt(
    tokenizer: AutoTokenizer, prompt: str, device: torch.device
) -> tuple[dict[str, torch.Tensor], str]:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}, rendered


def save_checkpoint(
    output_dir: Path,
    records: list[dict[str, Any]],
    activation_rows: list[np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    records_tmp = output_dir / "records.jsonl.tmp"
    with records_tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    records_tmp.replace(output_dir / "records.jsonl")

    if activation_rows:
        matrix = np.stack(activation_rows, axis=0)
        activations_tmp = output_dir / "activations.tmp.npz"
        np.savez_compressed(
            activations_tmp,
            activations=matrix,
            layer_indices=np.arange(matrix.shape[1], dtype=np.int16),
        )
        activations_tmp.replace(output_dir / "activations.npz")


def main() -> int:
    args = parse_args()
    question_path = args.questions.expanduser().resolve()
    model_source = resolve_model_source(args.model)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_DIR / "results" / f"gemma3_4b_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stimuli, dataset_metadata = load_stimuli(question_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        stimuli = stimuli[: args.limit]

    print(f"Model source   : {model_source}")
    print(f"Questions      : {question_path}")
    print(f"Stimuli        : {len(stimuli)}")
    print(f"Device / dtype : {device} / {dtype}")
    print(f"Output         : {output_dir}")
    print(f"Local only     : {args.local_files_only}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=args.local_files_only
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    run_config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_source": str(model_source),
        "model_class": type(model).__name__,
        "device": str(device),
        "dtype": str(dtype),
        "questions_file": portable_path(question_path),
        "questions_sha256": file_sha256(question_path),
        "num_stimuli": len(stimuli),
        "max_new_tokens": args.max_new_tokens,
        "activation_position": "final prompt token before answer generation",
        "activation_layers": "embedding output followed by every transformer layer",
        "dataset_metadata": dataset_metadata,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    records: list[dict[str, Any]] = []
    activation_rows: list[np.ndarray] = []

    try:
        for stimulus_index, stimulus in enumerate(tqdm(stimuli, unit="prompt")):
            inputs, rendered_prompt = model_inputs_for_prompt(
                tokenizer, stimulus["prompt"], device
            )
            input_length = int(inputs["input_ids"].shape[1])

            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            if outputs.hidden_states is None:
                raise RuntimeError("The model returned no hidden states.")

            # Shape: [embedding + transformer layers, hidden size].
            layer_states = torch.stack(
                [state[0, -1, :].detach().float().cpu() for state in outputs.hidden_states]
            )
            activation_rows.append(layer_states.numpy().astype(np.float16))
            del outputs, layer_states

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, input_length:].detach().cpu()
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predicted_label = parse_predicted_label(response)

            record = {
                "activation_index": stimulus_index,
                **stimulus,
                "rendered_prompt": rendered_prompt,
                "input_token_count": input_length,
                "response": response,
                "predicted_label": predicted_label,
                "is_correct": predicted_label == stimulus["expected_label"],
            }
            records.append(record)

            del inputs, generated, new_tokens
            if device.type == "mps":
                torch.mps.empty_cache()

            completed = stimulus_index + 1
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                save_checkpoint(output_dir, records, activation_rows)

    except KeyboardInterrupt:
        print("\nInterrupted; saving completed samples...", file=sys.stderr)
    finally:
        save_checkpoint(output_dir, records, activation_rows)

    correct = sum(bool(record["is_correct"]) for record in records)
    accuracy = correct / len(records) if records else 0.0
    print(f"Saved {len(records)} samples to {output_dir}")
    print(f"Parsed-label accuracy: {correct}/{len(records)} ({accuracy:.1%})")
    if activation_rows:
        print(f"Activation array shape: {(len(activation_rows),) + activation_rows[0].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
