# exp145 — Shared-input conductance AI probe

Renart-inspired 4,000 E + 1,000 I recurrent neurons with 4,000 shared external
Poisson sources. Simulate 1.05 seconds from rest: exclude 50 ms burn-in and measure the following
1 second, without training.
Use direct area-matched exponential conductances, fixed mean delays and
nonnegative weights; these approximate the reference kinetics and delays.
Initializer means and standard deviations compensate native sparse rescaling,
so connected jumps retain the intended physical strengths.
The short measurement window limits estimates of irregularity and tiny correlations.

1. `compute.py`: `--seed` (optional, 42), `--device` (optional, cpu),
   `--measurement-ms` (optional, 1000; use 30000 for a 30-second window),
   `--run-id` (optional). Saves shared inputs, native bundle, start snapshot,
   complete spikes, and sampled voltages and projection conductances.
2. `analyse.py`: `--source <compute-run-id>` (required), `--run-id` (optional).
   Excludes burn-in; measures CV, EE/II/EI correlations, population-fluctuation
   scaling and signed-current cancellation. Count correlations use sliding
   5, 20 and 50 ms windows, advanced by 1 ms.
3. `present.py`: `--source <analyse-run-id>` (required), `--run-id` (optional).
   Produces the full raster, its first 200 ms, and the four-panel AI diagnostic plot.
4. `test.py`: `uv run pytest experiments/exp145/test.py`; small integration
   checks only, without running the full-size scientific configuration.
