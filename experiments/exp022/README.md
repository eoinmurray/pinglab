# Exp022: compute, analyse, present

The shared lifecycle is defined in [the experiment guide](../README.md).
`recipe.py` owns the scientific registry and read-only bank interface.
`compute.py` owns training, campaign recovery and retained diagnostic simulations.
`analyse.py` owns measurements; `present.py` draws saved analysis only.
The original `exp022.py` launcher now delegates to compute. Legacy Python imports
remain available, but combined `--skip-training`/`--plot-only` modes are retired.

## Current storage contract

New executions write `pingstore.run/v3`: required `run.json` and `export/`, with
optional `README.md` and `provenance/`. Compute/analyse exports may nest files;
present exports are flat figures, tables and numbers. Scripts, source patches,
import inventories and the detailed presentation-lineage attachment are retained
under `provenance/`. The authoritative record is always `run.json`.

The r001/r002/r003 runs below have been migrated to v3 with their IDs unchanged.
The original mixed-output Gold-2 source remains v2 and is read through version-aware
helpers. Old hidden v2 reservations require their original code or a fresh v3
reservation; they are not silently converted.

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

Rollback after completion requires stopping readers and restoring all three v2
originals together while retaining the v3 copies separately; do not mix versions
of this checksum-pinned chain. The script's `recover` mode is only for interrupted
activation and does not undo a completed migration automatically.

## Repaired Gold-2 migration

The following local runs were created without training or simulation:

| Stage | Completed run | Output |
| --- | --- | --- |
| Compute import | `exp022-r001-compute-local` | 102 cells, 408 unchanged scientific files under `export/cells/` |
| Analyse | `exp022-r002-analyse-local` | Results, checkpoint-role inventories and plot-ready curves under `export/` |
| Present | `exp022-r003-present-local` | Seven regenerated curve figures, numbers and 35 carried historical raster/comparison images |

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

## Use the migrated evidence

The three local staged IDs were subsequently reordered to put `rNNN` before
the stage label, so directory sorting follows execution order. Their input pins,
live metadata and replay scripts were updated together, without recomputation.
The pre-reorder originals are recoverable under
`/Users/eoin/pinglab-stage-id-order.zVpOCm/originals/`; the adjacent
`migration.json` records the mapping. Historical commands, source patches and
figure stamps retain their original identities, explained by `id_order_migration`
in each new run.json. The historical Gold-2 source was not changed.

```sh
# Analyse the imported bank; prints a NEW analyse ID.
uv run python experiments/exp022/analyse.py --source exp022-r001-compute-local

# Redraw the completed analysis, explicitly retaining the historical rasters.
# This creates a NEW presentation run, never overwrites r003.
uv run python experiments/exp022/present.py \
  --source exp022-r002-analyse-local \
  --retained-presentation exp022-gold-2-repaired-slurm
```

To generate genuinely new raw probes later, explicitly run:

```sh
uv run python experiments/exp022/compute.py \
  --source exp022-r001-compute-local --diagnostics
# Then analyse that NEW compute ID and present its analysis ID.
```

That command performs inference and is not part of the migration. New complete
training runs and campaign captures retain their probes from the outset.
The [campaign runbook](../exp022_support/README.md) covers worker retries and
preallocated scheduler identities. RunPod live dispatch prints its reserved
identity; collection requires `--runpod --collect --run-id <that-identity>`.

## Preview and publication

In exp022's Demolab preview selector, choose `exp022-r003-present-local`.
The existing article-scoped `data-file()` bindings consume its presentation.
The retained v3 compute/analysis runs have no presentation directory and are
excluded by their stage. The retained present run exposes `export/`; the original
historical v2 presentation remains an available candidate.
No preview build or browser check was run as part of this requested no-test
migration. No `.artifacts/` materialization or website publication was performed.

No test suites, training smoke runs, scheduler submissions or scheduler dry-runs
were executed. Import/source/finalization checksum validation was performed as
part of the storage contract. The code and documentation changes are uncommitted.
