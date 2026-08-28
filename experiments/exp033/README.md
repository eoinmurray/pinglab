# exp033 — conductance mean-field onset and dimensional reductions

## Contract migration

Execution follows Runner Guide 3.0.0 and Storage Guide 3.0.0. Contract migration
and the approved selective historical import are complete. Independent analysis
and presentation were executed without production simulation or training.
The initial migration was committed and pushed as `c8f65b1c`. The subsequent
authorized science/writing corrections are recorded in
[SCIENCE_REVIEW.md](SCIENCE_REVIEW.md), including the author-requested restoration
of omitted explanations and visible `[(!) ...]` scientific corrections.
The article is not Reviewed. Nothing was published or materialized.

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

**Shared Slurm dispatch still needs the narrow exp033 addition.** The exp047
owner handed back that addition after committing its own dispatch support. This
writing-only pass leaves `slurm.py` unchanged. Until exp033 is added, do not submit
it through the collection: the unreserved HPC worker fails before numerical work.

The initial mixed collection changes were separated by ownership before commit;
exp033 and exp047 integration were committed independently. Future shared edits
still require coordination. No simulator or another experiment package was
edited by this task.

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

## Initial migration verification

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

## Science and writing revision — 2026-08-28

The author authorized the corrections after the initial commit. The article now
uses numbered Results captions before a five-step Methods account (approximately
350 words excluding displayed equations). The original derivation and reduction
algebra remain in appendices, with explicit current-coordinate rescaling,
corrected dimensions and semantic equation/figure references. Creation date
2026-05-28 is preserved; updated_at is 2026-08-28. The status line is left to the
separate article-status owner; this task does not mark the article Reviewed.

Scientific connections are distinguished by role: exp025 motivates the mechanism
but does not identify its empirical recruitment marker with a Hopf current;
exp041 supplies three-seed median frequencies of separately trained networks;
exp054 reuses the same mean-field calculation rather than supplying independent
confirmation. No claim of a causal accuracy benefit, structural firing-rate
floor, calibrated noise scale, universal minimum dimension, or proved
supercriticality is made.

Presentation revisions use `exp033-r002-analyse` without changing results,
configuration, success Booleans or numerical detail strings. Only criterion labels
are qualified. `exp033-r004-present` was the first review presentation;
`exp033-r005-present` additionally moves a sensitivity-panel annotation clear of
its data curve. Both are immutable new presentation operations, not simulations.
The final run retains 1,236,778 bytes in total, including 1,021,327 export bytes;
the intermediate run retains 1,351,967 bytes. Existing historical runs and the
archive are unchanged; this pass does not reclaim disk space.

Redrawn figures omit internal run stamps and experiment labels, state amplitude
and eigenvalue units, and show sensitivity frequency on an absolute 0–40 Hz axis.
The four historical SVGs have only the producer stamp removed; the reduction
ladder additionally moves its legend 60 SVG units into added top margin. Every
historical axes subtree and scientific path remains identical. The presentation
manifest records each original and revised hash and the precise operations.
Raw waveform samples remain unavailable: no waveforms were reconstructed.

Validation: **197 tests passed** (12 existing exp041 layout warnings); targeted
render/layout regressions passed again after the final article corrections.
Lint and type checks pass. Tests cover the frequency-axis roundoff defect,
annotation/data overlap, historical SVG edit boundaries and hashes, independent
stage isolation, absent/corrupt article inputs, and the gain-subscript rendering
failure. All nine images load in the styled HTML; equation and figure links
resolve. Desktop and 390px reading views were inspected, including the main model
and split Siegert formula. All 16 PDF pages were visually checked, with changed
pages rechecked after corrections. The historical quadrature warnings remain an
unresolved numerical limitation, not a result certified by rendering or tests.

Explicit preview and verification records:
`.r2/exp033-writing-iu8sdxuj/` (selected presentation, HTML/PDF, tests,
`verification.json`, and source/output preservation checks). The final lineage is
`exp033-r005-present` → `exp033-r002-analyse` → historical `exp033-r001-compute`,
with the pinned exp041 analysis and its exp022 bank ancestry. No new analysis,
production compute, import, publication or materialization ran in this pass.

The subsequent restoration passes 43 dedicated tests. All nine figures and all
40 original numbered equations remain; the expanded 18-page PDF and restored
equations at 390px width were inspected. The shared status migration is separate
from this experiment's commit and remains preserved in the working checkout.
