# Falsehood and Impossibility Are Different Directions in an AI's Representation of Language

Code and stimuli for an exploratory activation study of Gemma 3 4B IT. The
study asks whether the model internally distinguishes statements that are
merely false ("Paris is the capital of Germany") from statements that could
not be the case at all ("Mount Everest is taller than Mount Everest"), and
finds that the two are carried by nearly orthogonal linear directions, while
the model's verbal labels conflate them.

## Repository layout

```text
data/
  questions_philosophical.json   17 philosophical families (85 prompts)
  questions_modality.json        15 topic families x 5 modality conditions (75 prompts)
  questions_combined.json        Both sets merged; input for the reported run
results/reference_run/           Published responses and derived statistics
extract_activations.py           Model inference and residual-stream extraction
analyze_axes.py                  Held-out probes, transfer AUCs, direction cosines,
                                 permutation test, surface baselines (Fig. 2)
analyze_sae.py                   Gemma Scope 2 SAE encoding of one layer
analyze_sae_modality.py          SAE feature ranking by modality contrasts (Table 3)
plot_behavior.py                 Behavioral figure for the philosophical set (Fig. 1)
```

## Setup

Python 3.10+ (3.11 used for the reported run).

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Gemma 3 is a gated Hugging Face model: accept Google's Gemma license on
Hugging Face and authenticate with `hf auth login` before the first download.
The extraction script itself runs offline from the local cache
(`~/.cache/huggingface/hub/models--google--gemma-3-4b-it` by default; override
with `--model-cache`).

## Pipeline

1. **Extract activations** (about 4 minutes on an Apple M-series GPU; also runs
   on CUDA or CPU):

   ```bash
   python extract_activations.py --output-dir results/run
   ```

   Writes `records.jsonl` (prompts, greedy responses, parsed labels),
   `activations.npz` (`[160, 35, 2560]` float16: embedding output plus 34
   layers at the final prompt token), and `run_config.json`.

2. **Probe geometry** (probes, transfer, cosines, permutation test, baselines;
   produces `figure/figure5_axes.[pdf,png]`, manuscript Fig. 2):

   ```bash
   python analyze_axes.py results/run
   ```

   Results are written to `results/run/axes/axes_results.json`. The reported
   impossibility peak is at hidden-state depth 16, i.e. transformer layer 15.

3. **SAE encoding** at the probe peak layer (downloads the Gemma Scope 2 SAE
   into the Hugging Face cache on first use):

   ```bash
   python analyze_sae.py results/run --sae-id layer_15_width_16k_l0_small
   python analyze_sae.py results/run --sae-id layer_15_width_16k_l0_big
   python analyze_sae_modality.py results/run --sae-id layer_15_width_16k_l0_big
   ```

   The modality contrast writes
   `results/run/sae/<sae-id>/modality_feature_contrasts.json`
   (manuscript Table 3).

4. **Behavioral figure** (manuscript Fig. 1):

   ```bash
   python plot_behavior.py results/run
   ```

## Reference run

`results/reference_run/` contains the exact model responses
(`records.jsonl`), run configuration, probe/transfer statistics
(`axes/axes_results.json`), and SAE outputs behind the manuscript's numbers.
The 21 MB raw activation array is not committed; extraction is deterministic
(greedy decoding), and a re-run reproduced all 85 philosophical predictions of
an earlier independent run exactly.

## Citation

See `CITATION.cff`. Model weights and SAE weights are governed by their
upstream licenses; see `THIRD_PARTY_NOTICES.md`.
