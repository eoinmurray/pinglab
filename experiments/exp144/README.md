# exp144 — MNIST-driven E/I activity

Compare dense inhibitory feedback with sparse feedback and private inhibitory
background excitation while holding the first MNIST training image fixed.
Measure asynchronous, irregular (AI) activity with fixed weights and no training.

## CLI

1. `compute.py`: `--protocol {comparison,screen,long}` (optional; default
   `comparison`: 500 ms from rest; `screen`: dense 2 s parameter screen;
   `long`: 11 s comparison at three seeds); `--run-id <reserved-id>` (optional).
2. `analyse.py`: `--source <compute-run-id>` (required);
   `--run-id <reserved-id>` (optional).
3. `present.py`: `--source <analyse-run-id>` (required; `comparison` or `long`);
   `--run-id <reserved-id>` (optional).
4. `test.py`: run with `uv run pytest experiments/exp144/test.py`; no arguments
   required and no scientific execution.
