# exp134 — Leak and capacitance modulation of PING dynamics

Compare paired leak-conductance and membrane-capacitance interventions in one
frozen trained PING circuit, measuring frequency, firing, participation and rhythm
contrast while reusing the same images and encoded spike trains.

1. `compute.py`: required `--source TRAINING_RUN` (exp022 checkpoint) and
   `--image-source IMAGE_RUN` (exp123 image bank); optional `--run-id RUN_ID`,
   `--images-per-digit N` (default 50, range 1–50), `--batch-images N` (default 5).
2. `analyse.py`: required `--source COMPUTE_RUN`; optional `--workers N` (default 4).
3. `present.py`: required `--source ANALYSIS_RUN`; no optional arguments.
4. `test.py`: run with `uv run pytest experiments/exp134/test.py`; optional pytest arguments.
