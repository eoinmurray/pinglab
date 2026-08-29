# Exp022: compute, analyse, present

**Current status (2026-08-27):** the [verified Gold-2 ancestry repair](ANCESTRY.md)
resolved the historical-source pins described in the earlier migration record
below. The existing bank and all dependent local runs have complete operational
v3 pins. Historical source declarations remain preserved as provenance.

The shared lifecycle is defined in [the experiment guide](../README.md).
`recipe.py` owns the scientific registry and read-only bank interface.
`compute.py` owns training, campaign recovery and retained diagnostic simulations.
`analyse.py` owns measurements; `present.py` draws saved analysis only.
Run compute explicitly as `python -m experiments.exp022.compute`. Legacy Python
imports remain available, but combined `--skip-training`/`--plot-only` modes are retired.

## Code layout and dependencies

Campaign and diagnostic helpers live with the experiment they serve:

- `campaign.py` implements manifests, cell validation and retry bookkeeping;
  compute passes it the registry and training arguments from `recipe.py`.
- `tr06_diagnostic.py` and `fr_strength_pilot.py` retain the bounded compute
  diagnostics and calibration pilot, separate from the production bank.
- `slurm/` contains the Wilkes environment diagnostic, shell helpers, submission
  scripts and [operator runbook](slurm/README.md).

Compute and archive validation import `experiments.exp022.campaign`. The
gamma-gated-sparsity collection reuses the scripts in `slurm/`. Downstream
experiments keep using the existing `exp022` registry and bank interface; the
scientific recipe, checkpoint roles and stage boundaries are unchanged.
Retain historical campaign checkouts for provenance; they do not authorize v2
execution or reservation completion under the current contract. This source-code
move does not rewrite their commands, manifests, reservations or completed runs.

## Current storage contract

All operational writes and inputs require `pingstore.run/v4`: exactly `run.json`,
`README.md`, and `export/`. Compute/analyse exports may nest files;
present exports are flat figures, tables and numbers. Scripts, source patches,
import inventories and the detailed presentation-lineage attachment are retained
under `export/evidence/`. The authoritative machine record is `run.json`; README
is the human-readable history.

The r001/r002/r003 runs below have been migrated to v3 with their IDs unchanged.
The original mixed-output Gold-2 source is historical v2 evidence, not an allowed
operational input or preview candidate. Any implementation path still accepting
v2 is nonconforming with Storage Guide 2.0.0. Old hidden v2 reservations must not
be completed or silently converted; new execution requires a fresh v3 reservation.
Historical inspection for migration/recovery and migration require separate
explicit authorization. These documentation changes do not migrate any evidence.

## Staged-data v3 migration

On 2026-08-27 the three staged runs were migrated together, without training,
analysis, rendering or test execution. Each now contains `run.json`, `README.md`,
`export/` and `provenance/`; none has a `presentation/` directory.

- Compute: the 408 scientific files remain under `export/cells/`; execution and
  import records moved from `export/provenance/` to `provenance/`.
- Analyse: the three analysis exports are unchanged; execution records moved
  to `provenance/`.
- Present: all 44 former presentation files moved byte-for-byte into `export/`;
  the separate presentation-lineage attachment moved into `provenance/`.

Every original file is retained at a checksummed mapped destination, including
the original v2 `run.json` and README under `provenance/format-v2/`. That directory
also retains the mapping, migration script and validator source. New manifests
record the format change separately from original scientific execution, update
current attachment paths and re-pin the migrated inputs in dependency order.
Historical commands, timestamps, source patches and figure stamps are unchanged.

Untouched v2 originals are recoverable at
`/Users/eoin/pinglab-exp022-v3.42KUXT/originals/`. The adjacent `migration.json`
records completion and old/new references; `migrate.py` records the exact
copy-first procedure. Activation briefly paused and then resumed the identified
Pinglab preview process. All migrated runs, input pins, original backups and the
unchanged Gold-2 source passed migration checksum validation. No tests, preview
builds or publication were invoked by the migration.

The historical rollback procedure requires stopping readers and restoring all
three v2 originals together while retaining the v3 copies separately; do not mix versions
of this checksum-pinned chain. The script's `recover` mode is only for interrupted
activation and does not undo a completed migration automatically.
Under Storage Guide 2.0.0, restoration is a separately authorized recovery
operation; restored v2 originals must remain outside operational use.

## Repaired Gold-2 migration

The following runs were created locally without training or simulation. The IDs
below use the current source-neutral convention:

| Stage | Completed run | Output |
| --- | --- | --- |
| Compute import | `exp022-r001-compute` | 102 cells, 408 unchanged scientific files under `export/cells/` |
| Analyse | `exp022-r002-analyse` | Results, checkpoint-role inventories and plot-ready curves under `export/` |
| Present | `exp022-r003-present` | Seven regenerated curve figures, numbers and 35 carried historical raster/comparison images |

The unchanged source is `exp022-gold-2-repaired-slurm`. Its 60 inherited and
42 repaired cells remain distinct in the copied historical provenance. Both
best-validation and final-epoch checkpoints are preserved. The imported run's
local creation time describes the import, not the historical SLURM jobs.
Its `run.json` references the original and points to the full historical
`provenance/imported-run.json`. `provenance/import-inventory.json` records each
scientific file's preserved hash. The separate base bank was not selected or changed.

The original code discarded raw raster snapshots. Accordingly, the new analysis
does not fabricate them, and the presentation explicitly references the original
run for 34 raster PNGs and the comparison PNG. These images were copied, not
regenerated; each is marked `carry-historical` in the new run's authoritative
presentation lineage. Curves and numbers were generated from retained metrics.

## Migrated evidence and operational prerequisites

The three local staged IDs were subsequently reordered to put `rNNN` before
the stage label, so directory sorting follows execution order. Their input pins,
live metadata and replay scripts were updated together, without recomputation.
The pre-reorder originals are recoverable under
`/Users/eoin/pinglab-stage-id-order.zVpOCm/originals/`; the adjacent
`migration.json` records the mapping. Historical commands, source patches and
figure stamps retain their original identities, explained by `id_order_migration`
in each new run.json. The historical Gold-2 source was not changed.

The command forms below require validated v3 sources and resolvable v3 input
pins throughout their lineage. The separately authorized ancestry repair now
satisfies this prerequisite. The original missing historical input remains
preserved in the repair records; no v2 source is used operationally.

```sh
# Analyse the imported bank; prints a NEW analyse ID.
uv run python experiments/exp022/analyse.py --source exp022-r001-compute
```

Reuse images retained in a v3 presentation
of the exact same analysis and compute bank. This creates a new presentation run:

```sh
uv run python experiments/exp022/present.py \
  --source exp022-r002-analyse \
  --retained-presentation exp022-r003-present
```

This redraws all seven curves with the label `validation accuracy (%)`, without
training or analysis. Historical images must match their recorded checksums;
their earlier lineage is retained as `source_lineage` alongside the immediate
copy source. Existing completed runs are never changed.

To generate genuinely new raw probes later, explicitly run:

```sh
uv run python experiments/exp022/compute.py \
  --source exp022-r001-compute --diagnostics
# Then analyse that NEW compute ID and present its analysis ID.
```

That command performs inference and is not part of the migration. New complete
training runs and campaign captures retain their probes from the outset.
The [campaign runbook](slurm/README.md) covers worker retries and
preallocated scheduler identities. RunPod live dispatch prints its reserved
identity; collection requires `--runpod --collect --run-id <that-identity>`.

## Preview and publication

### HPC identity correction (superseded)

This historical naming decision was reversed by the local-origin correction below.

On 2026-08-27, at the user's request, `exp022-r001-compute-local` was renamed
to `exp022-r001-compute-slurm`. At that point its suffix and `origin` described
the historical Slurm computation. Its recorded local import command, host and
timestamps remain unchanged; `origin_semantics` and `scientific_execution`
make that distinction explicit. Earlier migration records retain their old IDs.

The three dependent exp022 runs and five dependent exp024 runs were re-pinned
in dependency order. Current bank IDs in exp022 numerical exports and replay
scripts were updated; scientific values, all 408 bank files, figures and
historical execution attachments were preserved. Each affected run retains
original manifests and changed files under `provenance/hpc-id-migration/`.
Complete rollback copies and the migration journal are retained under
`/Users/eoin/pinglab-exp022-hpc-id.hHn6uH/`.

All nine affected runs passed layout, payload checksum and available input-pin
validation. The historical `exp022-gold-2-repaired-slurm` source was already
absent from the live store; its unresolved references in r001 and r003 remain
unchanged. The preview was briefly paused during activation and resumed.
No training, analysis, rendering, materialization or publication was performed.

### Local import origin correction

On 2026-08-27 the user authorized restoring `exp022-r001-compute-local` and
`origin: local`, matching the actual import command and host. Historical
`scientific_execution.origin: slurm`, all job records, checkpoint roles, and
original import timestamps are unchanged. The earlier HPC-name decision remains
in the migration history; it no longer defines the current identity.

The bank and 14 dependent runs were migrated together. All 21 completed runs
passed v3 layout, checksum and recursive input-pin validation. Only current bank
labels in three exp022 JSON exports changed; all numerical values, checkpoint,
array and figure bytes were preserved. Historical execution commands and import
plans remain evidence of their original operations, not silently rewritten inputs.

The complete rollback tree is
`.r2/exp022-local-origin-nzqdzl89/originals/`. The adjacent migration journal,
inventories, scripts and scientific-preservation report record the correction;
each affected run also retains its original manifest and changed files under
`provenance/local-origin-correction/`. Restore the complete verified chain only
under separately authorized recovery. Demolab was paused for activation and
resumed afterward. No science was rerun and nothing was published or sent to R2.

### Source-neutral IDs

The subsequent authorized naming migration removes execution-source suffixes:
`exp022-r001-compute`, `exp022-r002-analyse`, `exp022-r003-present` and
`exp022-r004-present` are the current identities. `origin: local` still describes
the import; `scientific_execution.origin: slurm` still describes the training.
All 21 completed runs and their dependent pins were migrated together. See the
[source-neutral migration record](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md) for
verification and complete rollback locations. Earlier sections retain the history
of the superseded naming decisions.

### Presentation selection

Preview may select only a validated v3 present run with conforming input lineage.
The ancestry repair resolved the source pins described in the historical notes.
The existing article-scoped `data-file()` bindings
consume the selected presentation.
The retained v3 compute/analysis runs have no presentation directory and are
excluded by their stage. A conforming present run exposes `export/`; historical
v2 presentations are not eligible candidates. Discovery must reject a visible v2
run rather than treating it as a legacy exception.
No preview build or browser check was run as part of this requested no-test
migration. No `.artifacts/` materialization or website publication was performed.

No test suites, training smoke runs, scheduler submissions or scheduler dry-runs
were executed. Import/source/finalization checksum validation was performed as
part of the storage contract. The code and documentation changes are uncommitted.
