# exp033 — conductance mean-field onset and dimensional reductions

## Contract migration

Execution follows Runner Guide 3.0.0 and Storage Guide 3.0.0. Contract migration
and the approved selective historical import are complete. Independent analysis
and presentation were executed without production simulation or training.
The article remains unchanged pending the substantive corrections listed in
[SCIENCE_REVIEW.md](SCIENCE_REVIEW.md); it is not Reviewed. Nothing was published,
materialized, staged, committed or pushed.

```sh
uv run python -m experiments.exp033.compute
uv run python -m experiments.exp033.analyse --source <exp033-compute-id> \
  --frequency-source <exp041-analysis-id>
uv run python -m experiments.exp033.present --source <exp033-analysis-id>
```

Each command independently completes one source-neutral v3 run. Compute needs no
upstream data: it solves the committed theoretical model. Analysis explicitly
pins both that computation and an exp041 analysis. Presentation pins the analysis
and consumes only its saved summaries and coordinates. There are no implicit
latest runs, legacy artifact fallbacks, downstream-trigger launches or publication
side effects. `--run-id` accepts only a fresh identity reserved by the shared
Pingstore helper. Failures remain hidden; completed runs are immutable.

- **Compute** retains the fixed-point/eigenvalue continuation, refined reference
  onset, inhibitory-decay and effective-noise sweeps, both hysteresis directions,
  full solver trajectories and the original dense-output waveform samples.
  Root refinement belongs here because it sets the controlled drives for later
  integration. Failed integrations never become completed evidence.
- **Analyse** measures the saved trajectories: peak-to-peak amplitudes, ramp
  differences, amplitude-squared regression and its R², waveform lag, and
  dimensional comparisons. It reads the saved continuation diagnostics rather
  than solving again, and combines them with explicit exp041 measurements.
- **Present** draws nine existing SVG figure families and the compatible
  `numbers.json` structure. Waveform plotting functions cannot integrate the model.

Numerical arrays use compressed NPZ with `allow_pickle=False` and a JSON structure
index; all retained values and array dtypes are preserved. Complete v3 ancestry,
manifest pins and payload checksums are checked before use and again before
completion. Source records, commands and execution environment go in provenance.

## Preserved scientific choices

The LIF gain, coupling definitions, four-variable equations and all seven
reductions retain the previous numerical functions. The only retired interfaces
are the combined runner and implicit exp041 storage lookup. Pure numerical
helpers remain importable from the package and flat compatibility module for
exp054; its legacy data lookup now fails explicitly pending that study's own
migration. The collection continues to reject legacy exp054 execution against
staged exp041.

- Reference drive grid: 0–4 nA, 401 points; effective noise 4 mV.
- Noise sensitivity: 3, 4, 5 and 6 mV; 121-point coarse and 241-point convergence
  grids over 0–1.2 nA. The reference scale is not fitted to the spiking network.
- Reference and sensitivity onsets retain Brent refinement; dimensional
  reductions retain the original unrefined grid crossing.
- Hysteresis: 25 points from onset minus 0.1 to onset plus 0.55 nA; 2,000 ms
  integrations, measuring E amplitude from 1,500 ms onward. Preserve endpoint
  continuation, the `1e-4` amplitude threshold, slope/intercept regression and
  the existing heuristic supercritical/inconclusive classification.
- Reference waveform: onset plus 0.4 nA, integrated to 700 ms; 1,500 samples
  over three onset periods for lag/amplitude and 2,000 over four periods for
  phase projections. Lag remains the absolute maximizer of the mean-subtracted
  I–E cross-correlation, not a newly estimated signed delay.
- Two-versus-four-dimensional comparison: onset plus 1 nA, integrate 300 ms,
  measure after 150 ms. The separate reduction ladder retains a common 1 nA
  drive and 400 ms integration, with the original kicks and deviations.
- Exp041 overlay: six inhibitory decays (4.5, 6, 9, 12, 18, 27 ms), median of
  the three network frequencies at each decay. Require all seeds 42–44; do not
  substitute the upstream article's condition means.

Solver tolerances, finite differences and integration settings are recorded in
`recipe.configuration()`. No reduced scientific smoke recipe was introduced:
`PINGLAB_SMOKE` does not reduce exp033's numerical work. Contract tests use
synthetic solver evidence instead of running the production sweep.

## Collection integration and outstanding ownership

The experiment-local adapter dispatches the three commands independently with
explicit source references, supports validated completed-stage reuse and rejects
interrupted reservations without replacing them. The collection plan, execution
adapter and collection fixtures include exp033. The graph's existing exp041
dependency remains conservative: exp033 computation itself is independent.

The authorized legacy-path test update is complete: exp033 is no longer tested
as an implicit old-storage consumer. Its dedicated tests assert side-effect-free
imports and explicit stage input validation.

**Shared Slurm dispatch remains blocked on ownership coordination.** Another task
added exp047 to the exact dispatch line after it was read. This task detected the
overlap and did not edit `slurm.py`. The author was notified. Until exp033 is added
by that owner or the line is handed back, do not submit exp033 through the
collection: the unreserved exp033 HPC worker fails before numerical work.

The shared plan, execution adapter and collection tests contain mixed exp033 and
concurrent exp047 changes. Future staging requires coordinated hunk ownership;
none of another task's changes are assumed to belong to this migration. No
simulator or another experiment package was edited by this task.

## Approved Gold-2 selective import

Live R2 `run.json`, `inventory.json` and `lineage.json` match the cached archive
byte for byte. Every proposed source file matches its size and SHA-256 evidence.

Exp033 owns 14 archive files totaling 1,060,262 bytes: nine SVGs, its numerical
summary and four log/status records. It has no archived raw-state directory.
Original producer: campaign `ggs-production-20260818-4ad223d3`, commit
`4ad223d32620dd9f03698b89f28aedfe944d43ac`, Slurm job 33913627.
Its stderr contains quadrature subdivision/roundoff/convergence warnings; these
remain provenance and require consideration in the later scientific review.

Exp054's separately produced `super_compound_cache.npz` (job 33913631) retains a
401-point mean-field sweep and Hopf, hysteresis and frequency results matching
exp033. This is separate execution evidence, not a recovered original exp033
trajectory. Select all five mean-field entries and exclude the independent
empirical-grid entry. The selected structure serializes to 115,015 bytes and an
estimated 38,027 bytes with lossless ZIP compression; every scalar round-trips
exactly. The source cache is 20,419,748 bytes.

The approved import selected **19 source files / 23,037,434 bytes**:

- Original numerical summary and four unreconstructable waveform SVGs:
  `limit_cycle`, `timeseries`, `phase_planes` and `reduction_ladder`.
- The complete mean-field portion of the exp054 cache, with its own distinct
  producer identity and source checksum.
- Exp033/exp054 job and completion evidence, campaign commands/submission records,
  original archive metadata, and original producer code from its recorded commit.

Observed import size: **3,207,722 bytes**, including **339,375 bytes** of
scientific export and full provenance. The selected cache ZIP is **38,027 bytes**.
Against 23,037,434 selected source bytes this avoids **19,829,712 bytes (86.1%)**
in the new import. Analysis adds 701,178 bytes and presentation 1,515,029 bytes:
all three new runs total **5,423,929 bytes**. These are new retained-run sizes,
not disk space reclaimed: the original archive remains unchanged. Provenance
includes the captured dirty-checkout patch as produced by the shared helper.

Reference `exp041-r002-analyse` and its validated ancestry through
`exp041-r001-compute` and `exp022-r001-compute`. Copy no bank, exp041 traces or
other spiking evidence. The current seed-median overlay differs from the original
by at most 0.000002545875 Hz; preserve the historical numbers and disclose the
choice of overlay rather than claiming exact recomputation.

Five figure families were redrawn; four were carried byte for byte with explicit
source pins and hashes. No missing waveform or sensitivity trajectory was
fabricated. The original exp033 and exp054 producer identities, commands, source
hashes and warnings are retained separately from the local import execution.

The approved plan is `.r2/exp033-contract-9ggzo454/import-plan.json`, SHA-256
`ab050895d14eb3ddb4a936e7120c342baf2979cabf278e81a278f4b8fc95afc8`.
Its pre-approval status field is preserved as evidence; user approval authorized
execution. Verification and preview records are in `.r2/exp033-import-ff1qdyap/`.

| Run | Role |
| --- | --- |
| `exp033-r001-compute` | Local historical import; no scientific compute executed |
| `exp033-r002-analyse` | New independent analysis of retained evidence |
| `exp033-r003-present` | New independent presentation; five redraws and four historical SVGs |

IDs were allocated by the existing atomic reservation helper, not guessed.
Import validates all selected source hashes and live archive metadata before
reservation, and rechecks original sources and upstream ancestry before atomic
completion. Operational exports use JSON/ZIP, not pickle. Only the approved,
hash-verified historical cache is unpickled during import.

All historical scientific summary scalars are preserved exactly. The local
amplitude-squared regression reproduces within rtol=1e-12, atol=1e-15; its slope
difference is 5.421010862427522e-20 and is recorded separately. The original
exp041 overlay remains unchanged with the explicitly approved upstream deltas.

## Verification

The combined regression passed **194 tests**, with 12 existing exp041 Matplotlib
layout warnings. Exp033 lint and type checks pass. Tests cover normal and
historical stage isolation, preserved grids, measurement equivalence, source and
ancestor corruption, schema/layout rejection, lossless arrays, failed solvers,
failed copies/rendering, hidden completion, reservations, and collection dispatch.
A dedicated fixture exercises all 19 import paths without using the real archive.

The unchanged article compiles to HTML and an 11-page PDF against the explicit
presentation. Browser inspection found all nine images loaded, 225 MathML
expressions and no desktop horizontal overflow. Every PDF page was inspected at
contact-sheet scale. Scientific claims, axis/label defects and remaining review
checks are listed in SCIENCE_REVIEW.md; rendering success does not resolve them.
The article has not been marked Reviewed, and no publication binding was changed.

All 48 runs visible at the start of this phase retain their original manifest
hashes and validate. At verification, all 54 completed runs (including concurrent
work) and their input pins validated. The selected archive files remain unchanged.
No exp033 files were staged; unrelated edits, including concurrent `047/048`
packages, shared collection changes, article metadata and `uv.lock`, were preserved.
