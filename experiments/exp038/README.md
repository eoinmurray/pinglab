# exp038 — switching on the inhibitory loop

Contract migration target: Experiment Runner Guide **3.0.0** and Storage Guide
**3.0.0**, using `pingstore.run/v3` and source-neutral stage IDs. Scientific
interpretation and article revision remain separate from this execution migration.

## Independent stages

```sh
uv run python -m experiments.exp038.compute --source <exp022-compute-run>
uv run python -m experiments.exp038.analyse --source <exp038-compute-run>
uv run python -m experiments.exp038.present --source <exp038-analyse-run>
```

Each command completes one immutable run. Compute generates probes without
training; analyse reads retained recordings and bank histories; present renders
saved arrays and summaries. No stage starts upstream work, chooses a latest run,
reads a legacy cache, materializes `.artifacts`, or publishes.

All operational inputs require validated v3 layout and checksums. Exact
upstream manifest and payload pins are checked recursively before use and again
before completion. A stage becomes visible only through atomic completion.
Failures leave hidden incomplete runs; they are not reusable evidence.
`--run-id` accepts a fresh reservation, including one allocated before Slurm
submission. New collection plans reserve and dispatch all three stages, while
old monolithic plans and the retired combined entrypoints fail explicitly.

## Preserved scientific procedure

The bank contains all 36 TR-02 cells: COBA and PING, six activity-ceiling
conditions, and seeds 42–44. Unlike the endpoint-dynamics studies, exp038 uses
**best-validation checkpoints (`weights.pth`)**, not final-epoch weights. The
selected epoch, filename and full checkpoint hash are retained for every cell.
Training remains the shared 50-epoch MNIST recipe; no training was repeated.

The production recipe retains 101 probe jobs:

| Probe | Jobs | Evidence and sampling |
| --- | ---: | --- |
| Input-rate snapshots | 10 | PING seed 42; first ten values of the original 40-point 0–100 Hz grid; official-test image index 0 |
| Uniform-input firing curves | 52 | COBA and PING seed 42, 26 rates, 32 uniform-Poisson trials per rate |
| Inhibitory-loop sweep | 33 | COBA seeds 42–44, eleven strengths 0–1, 1,000 official-test images per condition |
| Loop-transfer snapshots | 6 | COBA seed 42, strengths 0, 0.2, 0.4, 0.6, 0.8 and 1; image index 0 |

The loop-transfer probes retain `--skip-load W_ei. W_ie.`: the trained
feedforward/readout weights load while newly initialized loop matrices remain.
The simulation timestep is 0.1 ms and each trial lasts 200 ms. The smoke profile
uses 20 jobs: three input-rate snapshots, six uniform probes with two trials,
nine transfer evaluations with 100 images each, and two transfer snapshots.
The declared but unused legacy zoom/perturbation helpers are not new workloads.

Compute retains full E/I snapshot arrays, metadata and original metric records.
It strips only unused snapshot members via lossless ZIP compression; selected
NPY bytes and dtypes are unchanged. Configurations, commands and simulator logs
are retained under provenance. Model checkpoints remain in the pinned bank.
Analyse uses all cells to compute single-image firing rates, then samples 200 E
and 64 I cells with the original `default_rng(0)` procedure for display. It
preserves the across-seed mean and sample SD for the loop-transfer curves.

`baseline_results` and `frontier_summary` remain training-history summaries:
`best_acc` is selected validation accuracy, `final_acc` is final-epoch validation
accuracy, and `rate_e` retains the original final-history field and zero fallback.
They are not new official-test evaluations. Their mean/SEM conventions are
unchanged. Inference results remain in `ei_sweep`, `ei_sweep_summary`, and
`fi_sweep_uniform`.

Present preserves five figure families (ten PNG/SVG/PDF files) plus
`numbers.json`: `rate_rasters__ping`, `fi_curve__ping`, `fi_curve_uniform`,
`ei_rasters`, and `loop_transfer_compound`. The main article and downstream
exp092/exp109 consumers retain their existing figure and number keys.
Run IDs are no longer stamped onto scientific figures.

## Gold-2 audit and selective import

The live R2 inventory was verified byte-for-byte against the local Gold-2
inventory before migration. Its exp038 state contains **505 files / 757,518,763
bytes** (757.52 MB; 722.43 MiB). Every cached study-state file passes the
inventory size and SHA-256 check.

The archive selects the repair campaign `ggs-fr-repair-20260820-ac6f4988`.
The retained producer log identifies Slurm job **34111990**, host `gpu-q-39`,
and an NVIDIA A100-SXM4-80GB. All **36 best-validation checkpoint hashes** match
`exp022-r001-compute`. All 101 job configurations and recordings validate;
replaying the numerical summaries gives exact equality for baseline records,
frontier means/SEMs, transfer records/means/SDs and uniform-input firing curves.
This is a read-only comparison, not a new simulation or operational import.

Selection approved and imported in the subsequent pass:

- Pin the existing exp022 bank without copying checkpoints or histories.
- Retain all 85 metrics files (33 transfer evaluations and 52 uniform probes).
- Retain `dt`, `n_e`, `n_i`, `label`, `spk_e`, and `spk_i` from all 16 snapshots:
  **96 NPY members / 163,852,544 bytes before lossless compression**. Do not
  subsample trials or neurons in compute evidence.
- Preserve original configs, commands, logs, archive metadata, producer records,
  historical summary, source checksums and a per-array destination mapping.
  Keep unused voltages/other snapshot members and old plots in Gold-2.
- Create a local historical-import compute run, then independently analyse and
  present it; validate against the archive before considering publication.

The selected-array total is not an estimate of final disk size: ZIP compression
and provenance change that total. The initial audit is under
`.r2/exp038-contract-lh7rxutz/`; the executed plan, logs and verification are under
`.r2/exp038-import-x5gwo386/`.

The independently completed runs are:

| Stage | Run | Input |
| --- | --- | --- |
| Historical compute import | `exp038-r001-compute` | `exp022-r001-compute` |
| Analyse | `exp038-r002-analyse` | Imported compute and the same bank |
| Present | `exp038-r003-present` | The explicit analysis run |

The compute manifest records `origin: local` and operation `historical-import`.
Its `historical_import.producer` separately identifies the original Slurm job,
host, device and repair campaign. The original campaign manifest still says
`planned`; it is preserved verbatim. The retained exp038 completion event,
matching campaign provenance and complete outputs establish study completion.
No missing collection-status record or execution timestamp was invented.

The importer copies metrics without normalization and losslessly recompresses
selected NPY members. `provenance/import-plan.json` and `file-mapping.json`
retain source checksums, per-array checksums and destinations. Original configs,
commands, logs, archive records and summary remain under `provenance/gold-2`.
The unchanged R2 archive remains the recoverable original. No simulation or
training was performed, and no run was materialized or published.

Reproduction requires an explicit plan and the verified archive:

```sh
uv run python -m experiments.exp038.import_gold2 plan \
  --archive .r2/gold-2 --source exp022-r001-compute --plan <new-plan.json>
uv run python -m experiments.exp038.import_gold2 import \
  --archive .r2/gold-2 --plan <new-plan.json>
```

Import creates only compute; analysis and presentation use the independent
commands above. Re-execution creates fresh identities, never overwrites a run.

## Scientific and writing review

The migration and import initially left the article unchanged. The subsequent
authorized scientific/writing pass revised it under Writing Guide **9.0.0**,
preserving the creation date and setting `updated_at: 2026-08-28` for the
substantive corrections. Its milestone is now **Ready for review**, not Reviewed;
author acceptance and publication remain separate.

The revised account puts numbered Results before a four-step Methods section,
retains the training/probe settings and both main figures, and replaces rendered
repository identifiers with descriptive scientific links. Numerical claims are
interpolated from the selected presentation. The MNIST reference was verified
against the authors' publication record and DOI.

Corrections grounded in the retained evidence and implementation:

- Checkpoint selection minimizes mean validation cross-entropy over three fixed
  encoding draws; accuracy and earliest epoch break ties.
- The recorded image label is **7**, not the digit-0 default in the simulator's
  configuration. Explicit image index 0 overrides class selection.
- Loop strength varies both E→I and I→E initializer means, with their ratio fixed
  at two; it is not an E→I-only intervention.
- The 13.8-fold E-rate reduction and accuracy loss are retained. Rasters show
  illustrative burst grouping, not a spectral gamma-frequency measurement or
  proof of a continuous transition. The experiment neither isolates a causal
  benefit of gamma timing nor tests recovery by retraining; the readout is not
  established as the sole cause of the accuracy loss.
- The auxiliary E+I overlay remains an unweighted sum of population means.

Presentation `exp038-r004-present` first corrected labels and raster layout.
`exp038-r005-present` additionally shortens the sum legend to avoid covering the
uniform-input curve. Both pin the unchanged `exp038-r002-analyse`; all previous
runs remain immutable. The article accepts older v3 presentations without an
image-label projection by describing the same test image without guessing its
class. No automatic selection, materialization or publication was performed.

## Migration verification

The initial runner migration's storage and affected experiment/collection suite passed:
**493 tests, 26 plotting-layout warnings**. This includes 15 exp038 stage tests
covering explicit inputs, checkpoint roles, corrupt evidence, atomic failure,
collection dispatch/resume and article rendering. Scoped Ruff checks and
`git diff --check` pass.

Synthetic fixtures were used to inspect all five figure families and the
unchanged article's two-page render. These are rendering checks, not new
scientific results; inherited crowded labels/layout warnings remain documented
above. The historical audit separately reproduces the archived numerical
summaries exactly. All 30 existing operational runs retain their original
manifests and valid payloads; that initial pass performed no import or publication.
The subsequent authorized import and its verification are recorded above and below.

### Completed import verification

- All **85 metrics files** and **96 selected NPY members** are byte-identical
  to their Gold-2 originals. All 512 selected archive files pass size/checksum
  checks, and the live R2 metadata matches the retained archive metadata.
- Baseline/frontier records, transfer records and summaries, uniform firing
  curves, recipe metadata and training revision match the historical summary
  exactly. Checkpoint records match exactly when keyed by training cell; their
  list order differs. All 16 analysed raster arrays and rates match analysis of
  the original snapshots.
- All three completed v3 runs and their recursive input pins validate. The 30
  preceding operational runs, article and published artifact files are unchanged.
- The expanded regression suite passes **501 tests with 28 inherited plotting
  warnings**, including 23 exp038 tests. Scoped Ruff and whitespace checks pass.
- All five figure families and the unchanged article's two-page render were
  inspected using the imported data. The first imported presentation had
  overlapping ten-row raster labels, resolved in the subsequent review pass.

| Run | Export bytes | Whole run bytes, including provenance |
| --- | ---: | ---: |
| `exp038-r001-compute` | 1,028,560 | 5,612,054 |
| `exp038-r002-analyse` | 128,018 | 295,932 |
| `exp038-r003-present` | 597,189 | 763,973 |

The three runs total **6,671,959 bytes** (6.67 MB), without duplicating the bank.
These are file-byte totals, not filesystem allocated blocks. Lossless compression
accounts for the small scientific export; no retained E/I spikes were discarded.

### Article and presentation verification

The review audit is retained under `.r2/exp038-review-ma3l8_d1/`.
**113 focused tests pass**, covering stage/import behavior, article input
selection, milestones, recorded class labels and non-overlapping raster
annotations. The three-page article and all five figure families were inspected
with the retained data. Scientific numbers are unchanged from the initial
presentation; only labels, layout, execution metadata and the projection of
saved image labels differ. The 33 preceding runs and published artifacts remain
unchanged. No training, simulation or reanalysis was performed in this pass.

## Future-run data retention — 2026-08-28

Raster jobs record only E/I spikes and metadata. Uniform-drive f–I probes use the E/I spike recorder without voltage, conductance, input or readout traces. Native compute no longer rewrites snapshots to remove unused fields.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
