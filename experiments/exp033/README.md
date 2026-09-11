# exp033 — conductance mean-field onset and dimensional reductions

Exp033 tests whether a conductance-based population-rate model develops a
gamma-frequency oscillatory instability and which state variables are required
to retain it. The operational implementation follows Experiment Runner Guide
4.6.0 and Storage Guide 4.7.0.

## Current result

The accepted lineage is:

```text
exp033-r011-compute
  → exp033-r012-analyse
  → exp033-r015-present
```

Analysis also pins `exp041-r002-analyse` for the independently measured
spiking-network frequencies. Compute ran locally in 2 min 17 s.

Recipe v2 places the continuous-time LIF refractory periods at 1.2 ms for E and
0.6 ms for I. It also evaluates the negative branch of the Siegert gain integral
with the cancellation-resistant `erfcx` form. Relative to retained recipe v1,
the reference onset moves from 0.596337 to 0.593905 nA, while the onset frequency
remains 27.566445 Hz. The E-cycle amplitude at onset plus 0.4 nA increases by
2.256%. The sampled criticality and dimensional-reduction conclusions do not
change.

The numerical evidence supports a continuous, reversible onset at the sampled
resolution. It does not prove the absence of a narrow bistable interval or an
unstable cycle because no first Lyapunov coefficient was computed. The
dimensional result applies to the tested quasi-steady-state reduction family; it
is not a claim that every possible two-dimensional model must fail.

## Commands

```sh
uv run python experiments/exp033/compute.py
uv run python experiments/exp033/analyse.py \
  --source <exp033-compute-run-id> \
  --frequency-source <exp041-analyse-run-id>
uv run python experiments/exp033/present.py \
  --source <exp033-analyse-run-id>
```

Each command creates exactly one source-neutral `pingstore.run/v4` run.
Compute has no inputs. Analyse explicitly pins the exp033 compute run and the
exp041 frequency analysis. Present pins the exp033 analysis. No stage selects a
latest run, launches another stage, publishes, or materializes outputs.

- **Compute** retains fixed-point and eigenvalue continuations, refined onset,
  inhibitory-decay and effective-noise sweeps, both hysteresis directions, and
  the trajectories needed by analysis.
- **Analyse** validates and measures those saved trajectories, then combines
  them with the explicitly selected exp041 frequency measurements.
- **Present** renders six flat SVG figures and `numbers.json` from saved
  analysis evidence. It performs no integration or measurement.

## Committed recipe

- Reference drive: 0–4 nA over 401 points; effective noise 4 mV.
- Noise sensitivity: 3, 4, 5, and 6 mV, using 121-point and 241-point grids
  over 0–1.2 nA.
- Hysteresis: 25 drives from 0.1 nA below to 0.55 nA above onset; 2,000 ms per
  drive, measuring E amplitude after 1,500 ms.
- Reference cycle: onset plus 0.4 nA, integrated for 700 ms.
- Two-versus-four-dimensional comparison: onset plus 1 nA, integrated for
  300 ms and measured after 150 ms.
- Reduction ladder: common 1 nA drive and 400 ms integrations.
- Frequency comparison: six GABA decay values from 4.5 to 27 ms and the median
  of exp041 seeds 42–44 at each value.

Exact grids, solver tolerances, sample counts, thresholds, and measurement
settings are defined once by `recipe.configuration()`. There is no reduced
scientific smoke recipe.

## Outputs and limitations

Compute and analyse store their JSON structure and finite numerical arrays as
flat scientific exports. Present produces:

```text
bifurcation_compound.svg
sigma_sensitivity.svg
limit_cycle.svg
timeseries.svg
phase_planes.svg
reduction_ladder.svg
numbers.json
```

The effective voltage-noise scale is a sensitivity parameter, not a fit to the
spiking network. Driving forces are fixed and shunting dynamics are omitted.
The frequency comparison does not establish that the empirical recruitment
transition is the same bifurcation, and the measured cross-correlation lag is
not a signed causal-delay estimate.

## Historical evidence

Recipe v1 remains available solely to validate retained scientific history and
exp054's historical theory configuration. Existing imported and migrated runs
remain immutable. The live exp033 stages operate only on native compute evidence;
they do not repeat the completed Gold-2 import or carry historical SVGs into new
presentations.

Run-level provenance, input digests, migration details, and dated execution
history belong to each run's `run.json` and `README.md`, rather than this
experiment overview.
