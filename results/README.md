# Results

`reference_run/` holds the compact published record of the run reported in the
manuscript: model responses and parsed labels (`records.jsonl`), the exact run
configuration (`run_config.json`), probe/transfer/cosine/permutation statistics
(`axes/axes_results.json`), and the SAE outputs for both layer-15 checkpoints
(`sae/`). The raw activation array (21 MB) is omitted; regenerate it with
`extract_activations.py` (deterministic greedy decoding).

New runs created by the pipeline are written here as sibling directories and
are ignored by git.
