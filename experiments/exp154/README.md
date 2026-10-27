# EI Weight Lifecycle

Test whether E→I and I→E plasticity alone can learn scalar food values. Compare a
full-feedback capacity diagnostic with consumption-only learning, metabolic
energy loss and repeated lives. One fixed abstract genome controls weight ranges,
bounds, activity targets, phase durations and plasticity. Conception restores the
unbiased inherited circuit; 12 prenatal internal-drive windows precede birth.
The first 20 food encounters are juvenile, followed by slower adult learning.
All agents face one fixed mapping, energy = 2 × cue − 1, and learn after consumption.
There is no evolution or selection. The report replays all three phases, decisions,
energy, deaths, births and native activity of all 20 rate cells with standard controls.
The metabolic budget is calibrated to show roughly 8–10 deaths in 300 encounters
for this single-seed demonstration.

1. `recipe.py`: architecture and resolved `configuration(profile, protocol)`.
2. `compute.py --protocol {capacity,lifetime} [--profile {pilot,smoke}] [--run-id ID]`:
   independent local CPU compute. Defaults: lifetime, pilot (seed 7 only).
3. `analyse.py --source COMPUTE_ID [--run-id ID]`: validate and measure one protocol.
4. `present.py --lifetime LIFETIME_ANALYSIS_ID [--run-id ID]`:
   recorded replay and a prediction-error/weight-norm/ratio figure; use the
   full lifetime pilot for the report.
5. `test.py`: `uv run pytest experiments/exp154/test.py -q`.
6. `custom/`: rate/current extensions and native snapshot lifecycle adapter.

Run Python entry points with `uv run python` from the repository root. Stages
never invoke upstream stages or select latest inputs. Pingstore preserves all
executed settings and native bundles/snapshots; the original scratch evidence
is retained separately and is not an input to this experiment.
