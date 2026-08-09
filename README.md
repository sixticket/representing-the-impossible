# When Language Describes the Impossible

An exploratory mechanistic-interpretability study of how Gemma 3 4B IT
processes coherent statements, contradictions, paradoxes, and underdetermined
philosophical cases.

The repository contains the 85-stimulus dataset, activation-extraction code,
group-held-out probe and control analyses, Gemma Scope 2 SAE analysis, plotting
code, a compact reference run, and the LaTeX manuscript. The included results
support reproducibility without redistributing model or SAE weights.

## Repository layout

```text
data/                    Stimulus families and prompt template
results/reference_run/   Published responses, activations, and SAE features
figure/                  Generated figures and summary statistics
paper/                   LaTeX source, bibliography, figures, and compiled PDF
extract_activations.py   Model inference and residual-stream extraction
analyze_sae.py           Gemma Scope 2 feature encoding and ranking
analyze_controls.py      TF-IDF, permutation, and residual-norm controls
plot_figures.py          Main analyses and publication figures
```

## Setup

Python 3.11 is recommended.

```bash
git clone <your-repository-url>
cd llm-represent
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Gemma 3 is a gated Hugging Face model. For a new download, accept Google's
Gemma license on Hugging Face and authenticate with `hf auth login`.

## Reproduce the included analyses

The compact reference run is already present, so the figures and controls do
not require loading Gemma or downloading model weights.

```bash
python plot_figures.py results/reference_run
python analyze_controls.py results/reference_run
```

These commands update files under `figure/`. To encode the included activations
with the official SAE (download required on first use), run:

```bash
python analyze_sae.py results/reference_run
```

## Run a new extraction

Run a two-prompt smoke test before processing all 85 stimuli:

```bash
python extract_activations.py --limit 2
python extract_activations.py
```

To use an existing local Hugging Face cache directory or resolved snapshot
without network access:

```bash
python extract_activations.py \
  --model /path/to/models--google--gemma-3-4b-it \
  --local-files-only
```

Each run is saved under `results/gemma3_4b_<timestamp>/`. The activation array
has shape `[prompt, representation depth, residual width]`; depth zero is the
embedding output and subsequent entries are transformer-layer outputs at the
final prompt token.

## Build the paper

```bash
cd paper
tectonic main.tex
```

Standard `latexmk` or `pdflatex`/`bibtex` workflows also work. The Springer
Nature class and bibliography style are retained to make the source bundle
self-contained.

## Scope

This is a deliberately small, correlational study: one model, 17 stimulus
families, 85 prompts, and no causal intervention. Probe decodability is not
evidence for a unitary philosophical concept, and the reported SAE features are
candidate correlates rather than identified “paradox neurons.” See the paper
for the complete limitations and controls.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The project code and original
data are released under the MIT License. Model weights, SAE weights, and the
Springer Nature template files remain subject to their respective terms.

