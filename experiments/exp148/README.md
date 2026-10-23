# exp148 — External-I-drive transition

Test whether reducing direct external excitation of inhibitory neurons produces
population oscillations in the exp145-equivalent shared-input COBA network.
After 50 ms burn-in, record 500 ms baseline, ramp X→I weights from full strength
to zero over 500 ms, then hold for 500 ms. Preserve all other weights and state.

## CLI

1. `compute.py`: `--seed` (optional, 42), `--device` (optional, cpu),
   `--final-i-gain` (optional, 0–1; default 0), `--run-id` (optional).
   `SNNLAB_DEVICE` overrides `--device`.
2. `analyse.py`: `--source COMPUTE_ID` (required), `--run-id` (optional).
3. `present.py`: `--source ANALYSE_ID` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp148/test.py -q`.
