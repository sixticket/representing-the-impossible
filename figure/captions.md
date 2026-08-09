# Figure captions

## Figure 1. Behavioral classification of philosophical stimuli

**(A)** Row-normalized confusion matrix for Gemma 3 4B IT on 85 stimuli. Each
cell reports count and row percentage. **(B)** Exact four-way classification
accuracy overall and by stimulus form; error bars are Wilson 95% confidence
intervals. The model achieved 55.3%
overall accuracy and showed a strong tendency to use the `paradox` label.

## Figure 2. Layer-wise decodability of coherence

Five-fold group-held-out linear probes distinguish coherent from noncoherent
stimuli. All variants derived from one seed item were assigned to the same fold.
Blue uses the expected binary label; orange uses the model's eventual binary
judgment. Shading denotes the standard error across folds. The dotted baseline
uses input token count alone. Expected coherence peaked at transformer layer
18 (balanced accuracy
0.79); the model's own
judgment peaked at layer 17
(0.92).

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
