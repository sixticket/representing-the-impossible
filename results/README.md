# Results

`reference_run/` is the exact compact run used in the manuscript. It contains
the model's responses, the final-prompt-token residual states, the run metadata,
and the derived layer-12 SAE feature array. Model and SAE weights are not
included.

New extraction runs are written beside it as `gemma3_4b_<timestamp>/` and are
ignored by Git by default because activation files can grow quickly.

