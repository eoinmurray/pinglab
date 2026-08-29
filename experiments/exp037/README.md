# exp037 — dropped spikes and added noise

## Contract migration

Runner Guide 3.0.0 and Storage Guide 3.0.0. The scientific recipe remains the
existing TR-02 perturbation study; training belongs to exp022. The flat runner,
implicit scratch/cache paths and legacy RunPod dispatch entrypoint are retired.
Importing the package exposes pure recipe and measurement helpers, not execution.

Each stage requires an explicit completed v3 source and completes independently:

```sh
uv run python -m experiments.exp037.compute --source <exp022-compute-run-id>
uv run python -m experiments.exp037.analyse --source <exp037-compute-run-id>
uv run python -m experiments.exp037.present --source <exp037-analysis-run-id>
```

- **Compute:** 192 quantitative evaluations and 12 illustrative snapshots from
  validation-selected `weights.pth` checkpoints. No training, analysis or publication.
  Metrics and exact NPY entries for timestep, population sizes, label and E/I spikes
  are retained. Supporting configurations and execution logs go in `export/evidence/`.
- **Analyse:** reads the pinned 36-cell TR-02 bank histories and compute evidence;
  produces baseline/frontier summaries, per-seed perturbation rows, across-seed
  aggregates, normalized curve coordinates and raster display coordinates.
- **Present:** draws only saved analysis, with the existing ten figure files and
  `numbers.json` in a flat export. It does not simulate or recalculate estimators.

Inputs and their complete ancestry are validated before use and rechecked before
completion. Writes stay in hidden temporary run directories until atomic completion.
V2, wrong-stage inputs, changed source manifests and altered payloads are rejected.
Nothing selects a latest run or writes `.artifacts`.

## Local and HPC execution

`PINGLAB_SMOKE=1` reduces quantitative evaluation to 100 images and uses four drop
levels and three add levels: 42 evaluations plus the same 12 raster jobs. Production
uses 1,000 images, 11 drop levels and 21 add levels. Both retain seeds 42–44;
raster jobs use seed 42, sample index 0. Checkpoint selection and statistics do
not change between profiles.

The collection retains **six ordered round-robin compute shards**. The adapter
reserves source-neutral compute/analyse/present IDs before submission. A worker
uses the explicit bank and compute reservation:

```sh
uv run python -m experiments.exp037.compute --source <bank-id> \
  --run-id <reserved-compute-id> --shard-index <0-to-5>
uv run python -m experiments.exp037.compute --source <bank-id> \
  --run-id <reserved-compute-id> --collect
```

Workers require committed execution code. Each shard records its bank, recipe,
code provenance, job IDs, scheduler identity and checksums of outputs and attachments.
A completed shard can be reused only after those checks pass. The collector
requires all six shards, matching worker/collector code and an exclusive compute
lock; it does not rerun jobs. An incomplete job or stale writer lock needs explicit
recovery or a fresh reservation. Old cache-tag directories and historical campaign
plans are not operational inputs. Collection orchestration dispatches analyse and
present separately after compute completes.

## Preserved measurements

- All 36 baseline rows retain final history `acc` and `rate_e`, plus the recorded
  best accuracy/epoch. These are not replaced by official-test perturbation values.
- Perturbations use the **best-validation checkpoint**, not the final checkpoint.
  Per-seed accuracy and E rates come from retained inference metrics. Hidden rate
  selection preserves the original lexicographically last `hid*` key rule.
- Perturbation summaries and curve envelopes use the across-seed mean and sample
  SD (`ddof=1`); frontier summaries retain their separate SEM convention.
- Added-noise normalization divides each added rate by the arithmetic mean of the
  three final unregularized baseline history `rate_e` values for that model. The
  plotted x coordinate multiplies this ratio by 100. This historical choice is
  preserved, not silently replaced by an official-test baseline.
- Raster rates use the complete E population and the original array dtype's sum.
  Display cells use RNG(0), choosing 200 E cells then 64 I cells without replacement
  and sorting each selection. Recorded image labels remain in the analysis.
- The initial migration preserved plot labels and limits. The subsequent writing
  review corrected intervention labels, removed run stamps and exposed the full
  saved added-rate range without changing measurements or raster coordinates.

The `baseline_results`, `frontier_summary`, `perturbation`,
`perturbation_summary`, `config` and checkpoint interfaces remain available.
`notebook_run_id` is retained as a presentation identity alias. exp092 and
exp109 consume summaries and/or `perturbation_curves.svg`; migrating their execution
is separate work.

## Gold-2 audit and selective import

On 2026-08-28, a read-only audit checked all **1,036 exp037-related archive files**
against the cached Gold-2 inventory. State contains **568,867,555 bytes in 1,020
files**, including 12 snapshots totalling 567,785,160 bytes. The local
`exp022-r001-compute` bank matches all 36 retained checkpoint records.

Replaying the migrated calculations reproduced all 36 baseline rows, 192
perturbation rows, 64 summary rows, the complete frontier summary, all 12 raster
selections and the plotted coordinates of all five figure families exactly.
The audit is not an operational import or a new simulation.

The completed selective import retains:

| Evidence | Purpose |
| --- | --- |
| 192 `metrics.json` files | Per-seed test accuracy, rates and counts |
| Six selected NPY entries from each of 12 snapshots | Exact raster replay and full-population E rates |
| Configurations, commands, logs, source inventories and lineage | Historical producer and import provenance |
| A pin to the existing exp022 compute bank | Checkpoints and histories without copying the bank |

The selected snapshot entries total 122,889,408 uncompressed bytes. The completed
import retains the exact NPY entry bytes, including dtypes, in **420,969 bytes**
of lossless ZIP storage; metrics add **158,755 bytes**. Scientific payload totals
**579,724 bytes**. No trials or neurons were subsampled. Voltages, conductances,
input spikes and other unused arrays remain in unchanged Gold-2.

On **2026-08-28**, fresh copies of the live R2 `run.json` and `inventory.json`
matched the cached archive byte for byte. The recorded plan selects 1,039 source
files (569,410,841 source bytes, including the full source ZIP sizes). Import
verifies those hashes before and after copying and checks checkpoint roles,
simulation configurations, sample counts, numerical replay and producer lineage
before allocating a run. The scoped importer was retired after completion; its
executed code, plan and checks remain in immutable run provenance.

Completed runs, each created independently:

| Run | Operation | Complete run bytes |
| --- | --- | ---: |
| `exp037-r001-compute` | Local historical import, pinned to `exp022-r001-compute` | 8,051,632 |
| `exp037-r002-analyse` | Analysis of that imported evidence and bank | 499,225 |
| `exp037-r003-present` | Ten figure files and presentation numbers | 938,728 |

The compute export is 1,281,031 bytes including its recipe and training-contract
record. The larger complete-run size includes retained commands, logs, original
metadata, summaries, the selection plan, source-to-target hashes and code evidence.
The bank itself is referenced, not copied. These sizes count file bytes, not
filesystem allocation blocks. No training or simulation was launched, and no
presentation was materialized or published.

The original producer is the repaired HPC campaign
`ggs-fr-repair-20260820-ac6f4988`, commit
`ac6f49884084811e3e05d49e8b45735d514ff245`. Its six shards used array **34111991**;
the retained logs separately identify task jobs **34114002, 34114003, 34114152,
34114300, 34114579, 34111991** for shard indices 0–5. Aggregation job **34111993**
and the experiment completion event establish the retained result's identity.
The campaign manifest remains `planned` as originally recorded; it was not
rewritten to imply campaign-wide completion. The new import's `origin` is `local`;
`historical_import.producer` retains the original Slurm identities and evidence.

All 192 metrics and all 72 selected NPY entries were compared byte for byte after
import. All 36 baseline rows, 192 perturbation rows, 64 summaries, frontier results,
configuration and checkpoint records match the historical numbers exactly. All 12
raster coordinate sets match archive replay. Every existing run remains valid and
unchanged. Import validation tests also reject changed plans, checksums,
checkpoints, configurations, counts, summaries and producer/shard records; source
mutation during import never exposes a completed run.

Read-only migration replay records are in `.r2/exp037-contract-e2810qx5/`.
The import plan, live metadata comparison, command logs, byte/number verification
and article preview are in `.r2/exp037-import-otsz5p0q/`. The immutable compute run
retains its own plan, mapping, source records and human-readable history under
`provenance/` and `README.md`; the audit directory is not an operational input.

## Science and writing review — 2026-08-28

The article now follows Writing Guide 9.0.0: Results precedes Methods, captions
carry concise interpretation, four numbered scientific operations explain the
procedure, and an appendix preserves within-step dynamics. Its original creation
date is retained; `updated_at` is `2026-08-28` and status is `Ready for review`,
not author-accepted `Reviewed`.

The evidence audit corrected the following points:

- All six selected classifiers correspond to minimum validation loss, not
  maximum validation accuracy. Selection roles and checkpoint bytes are unchanged.
- All 12 illustrative recordings contain digit **7**, not digit 0. E spikes
  appear below I spikes; captions distinguish imposed I events in the disconnected
  COBA condition from recurrent inhibition.
- Both models fall to 10.6% at full deletion; COBA is not flat across that sweep.
  Deletion probability is not the observed fractional change in population rate,
  because network feedback also changes.
- Added spikes are Bernoulli insertions with probability `rate * dt / 1000`, capped
  at one spike per slot. The nominal rate is not the realized net increase.
- Normalization remains the three-seed final-epoch reference-image E rate:
  19.8584 Hz for PING and 155.1872 Hz for COBA. The corresponding unperturbed
  test-set rates are 16.4367 and 112.0885 Hz. Neither the reference normalization
  nor the equal 0–40 Hz nominal sweep establishes matched fractional doses.
- Perturbations affect both feedback and readout. Training also used different
  voltage-gradient damping. The results do not isolate gamma gating, prove an
  activity floor, or establish loss of an underlying oscillator from visible
  raster bands alone.

A new presentation, **`exp037-r004-present`**, pins the unchanged
`exp037-r002-analyse`. It corrects labels, omits run-ID stamps and shows all saved
added-rate points through 201.43% for PING rather than truncating at 150%.
All numerical fields and illustrative labels match `exp037-r003-present` exactly,
except normal presentation identity/timing/code fields. All older runs remain
immutable. No compute or analysis was rerun and nothing was published.

The article uses explicit aspect-ratio frames with contain fitting for paged
output and native responsive images for HTML. Typst's paged layout callback
otherwise omits the images from HTML. Review evidence, source backups,
selection checks, previews and test logs are in `.r2/exp037-review-9bg8qdsx/`.

## Verification

Tests cover independent stages, complete recipe grids, checkpoint roles, ancestry
and payload corruption, v2 rejection, atomic failure, source changes during work,
exact raster selection, six-shard completion/reuse/locking, collection dispatch,
and rendering the unchanged article from an explicitly selected presentation.
The historical replay uses only read-only archive evidence and writes audit files
outside Pingstore. No production GPU/HPC job was launched.

The initial contract-migration regression suite passed **554 tests** (20 existing layout warnings
from other studies). A subsequent **60-test** focused pass includes the final
collection shard-dispatch checks. Ruff, the exp037 type check and diff whitespace
checks pass. All 41 pre-existing run manifests and payload checksums remain valid
and unchanged, and the article is byte-identical to the starting version.
All five replayed figure families were visually inspected.

After import, the focused exp037 suite passed **34 tests**; the final broader
regression passed **524 tests**, with the same 20 existing exp041/exp044 layout
warnings. The concurrently edited writing-input tests were excluded from this
broader run, and exp099 was not changed. Ruff, the exp037 type check and whitespace
checks pass. All 41 pre-existing runs and the exp037 article remain unchanged.

The unchanged article was rendered against `exp037-r003-present` and inspected
in a continuous-flow audit preview: all five figures and all 12 raster panels
are present. An initial A4 audit wrapper clipped some raster panels; paged report
layout was subsequently corrected during the writing review above. The original article and
generated presentation images were not changed to address that wrapper issue.

The writing-review regression passed **162 tests**, including **36 exp037 tests**,
article inputs/status, collection dispatch and multiseed inference. Ruff and the
exp037 type check pass. The new present run's numbers are unchanged, and all
**47 runs present at the start of this review** retain valid, unchanged manifests
and payloads. Four-page and continuous-flow article previews were rendered;
figure layout, all 12 raster panels and the equations were visually checked.

The HTML rendering fix passed **125 tests** covering exp037, writing inputs and
writing status. A regression test requires all five images and captions in HTML;
the live browser preview was also checked against `exp037-r004-present`, with all
five images loaded and the corrected curve labels and full range visible.
Concurrent exp042/simulator work was left untouched.

## Future-run data retention — 2026-08-28

Raster jobs record only E/I spikes and their metadata. Compact payloads are moved from job scratch into the staged export without a second serialization. Quantitative inference remains metrics-only; all perturbations, samples and checkpoint roles are unchanged.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
