# exp168 — AI and PING with and without EE/II recurrence

This experiment compares two native EE/II implementations at the same timestep
(0.05 ms) and integration duration (2050 ms). The EE/II-on case restores
exp167’s recurrent reference weights while retaining exp168’s selected AI/PING
external-drive endpoints. The EE/II-off case uses the retained absent-projection
run `exp168-r002-compute`.

Stages:

1. `compute.py simulate`: simulate AI and PING with EE and II projections enabled. Requires simulation authorization.
2. `analyse.py --source COMPUTE_ID`: compute rates, rhythmicity and median ISI CV.
3. `present.py --source ANALYSE_ID`: render the EE/II-on two-panel raster.
4. `uv run pytest experiments/exp168/test.py`: focused non-simulation checks.

AI and PING differ in external source rate, input connectivity and constant
input-to-I gain. They are selected single-seed realizations, not a
single-parameter transition or evidence of robustness across seeds.
