# exp123 — Output evidence against time and gamma cycles

Measure decision timing and spike cost across a paired inhibitory-decay sweep in
one frozen trained PING network. All report figures use the same balanced sample.

Run stages independently from the repository root with `uv run python`.

1. `compute.py`: required `--source TRAINING_RUN` (validated exp022 checkpoint);
   optional `--image-source IMAGE_RUN` (validated exp123 image bank) and
   `--run-id ID`. Without an image source, the official MNIST test set must be
   available under `data/`; it is not downloaded automatically.
2. `analyse.py`: required `--source COMPUTE_RUN`; optional `--workers N`
   (default 1) and `--reuse-analysis ANALYSIS_RUN` (same compute payload).
3. `present.py`: required `--source ANALYSIS_RUN`; no optional arguments.
4. `test.py`: `uv run pytest experiments/exp123/test.py`; optional pytest arguments.
