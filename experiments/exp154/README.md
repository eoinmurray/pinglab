# EI Weight Lifecycle

Test whether E→I and I→E plasticity alone can learn scalar food values. Compare a
full-feedback capacity diagnostic with consumption-only learning, metabolic
energy loss and repeated lives. One fixed abstract genome controls weight ranges,
bounds, activity targets, phase durations and plasticity. Conception restores the
unbiased inherited circuit; 12 prenatal internal-drive windows precede birth.
The first 20 food encounters are juvenile, followed by slower adult learning.
All agents face one fixed mapping, energy = 2 × cue − 1, and learn after consumption.
An independent evolution protocol selects inherited settings by reproductive output on the
positive mapping, evolving E→I/I→E normal-distribution means with fixed standard
deviations; acquired weights are never inherited. The report replays all three phases, decisions,
energy, deaths, births and native activity of all 20 rate cells with standard controls.
The metabolic budget is calibrated to show roughly 8–10 deaths in 300 encounters
for this single-seed demonstration.

1. `recipe.py`: architecture and resolved `configuration(profile, protocol)`.
2. `compute.py --protocol {capacity,lifetime,evolution} [--profile {pilot,smoke}] [--run-id ID]`:
   independent local CPU compute. Defaults: lifetime, pilot (seed 7 only).
   Evolution pilot: initially 16 genomes, up to 10 generations, evaluated on seeds 7–9, plus winner/founder
   evaluation on seeds 10–14; each life stops at death or 300 food encounters.
   Living adults may reproduce once from encounter 25 onward. Each offspring costs
   0.5 energy, leaving a reserve of 0.5; litter size is the affordable count capped
   at three. Rank mean offspring, then survival, remaining energy and candidate ID.
   Retain up to four elites; actual newborns from their evaluated lives fill up to
   16 places. Record every independently mutated newborn, including those excluded
   by parental selection or the capacity cap. Never invent children to fill vacancies;
   stop early when no offspring can be admitted. Retained elites are unchanged genomes
   re-evaluated in independent lives, not continuing parental organisms.
   Initial E→I / I→E means are 0.03 / 0.20, with fixed standard deviations
   0.04/√12 / 0.20/√12. Normals are conditioned on weights lying between zero
   and one; these means/spreads describe the underlying normal before truncation.
   Shared random quantiles preserve fixed pathways when a mean changes.
   Capacity/lifetime protocols and historical evolution inputs retain uniform initialization.
   Evolution smoke: initially four genomes, up to two generations, one elite, seed 9,
   plus winner/founder on seeds 10–11, with the same 300-encounter horizon and reproductive
   thresholds. Start with smoke; the full pilot is separate.
3. `analyse.py --source COMPUTE_ID [--run-id ID]`: validate and measure one protocol.
4. `present.py --lifetime LIFETIME_ANALYSIS_ID [--evolution EVOLUTION_ANALYSIS_ID] [--run-id ID]`:
   recorded replay and a prediction-error/weight-norm/ratio figure; use the
   full lifetime pilot for the report. An explicit evolution analysis exports survival,
   inherited-parameter figures, a reproductive-output figure when available, and a
   diagnostic winner replay on the smallest fresh seed.
   Evolution outputs are kept out of the authored report and can be viewed directly.
   Normal-initialization runs also export `evolution-overview.svg` / `.png`: one
   3×3 figure of population-mean survival and all eight adaptive parameters.
5. `test.py`: `uv run pytest experiments/exp154/test.py -q`.
6. `custom/`: rate/current extensions and native snapshot lifecycle adapter.

Run Python entry points with `uv run python` from the repository root. Stages
never invoke upstream stages or select latest inputs. Pingstore preserves all
executed settings and native bundles/snapshots; the original scratch evidence
is retained separately and is not an input to this experiment.
