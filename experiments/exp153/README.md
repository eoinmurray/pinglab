# exp153 — Phase drift and coupling onset

Show unwrapped relative phase between two delayed mean-field circuits derived
from exp133. One uncoupled control calibrates detuning; one trajectory enables
reciprocal coupling at 500 ms. Circuit B has deliberately faster rate relaxation
to produce a visible drift before coupling. Only the phase-time panel is plotted.

## CLI

1. `compute.py`: required `--source EXP133_COMPUTE_ID`; optional
   `--difference-hz` (10), `--coupling-us` (0.08), `--rate-scale-b` (0.8),
   `--parallel` (1) and `--run-id`. Produces two 2,500 ms trajectories.
   The reference source is `exp133-r028-compute`. B's E/I relaxation times are
   multiplied by `--rate-scale-b`; 1 restores the symmetric closure.
2. `analyse.py`: required `--source COMPUTE_ID`; optional `--run-id`.
   Measures actual-input sustained rhythms, uncoupled frequency difference,
   settled drift and unwrapped peak phase. No upstream stage is launched.
3. `present.py`: required `--source ANALYSE_ID`; optional `--run-id`.
   Exports the phase-time figure, numerical calibration and reference diagram.
4. `test.py`: `uv run pytest experiments/exp153/test.py -q`.
