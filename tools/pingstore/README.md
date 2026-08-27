# Storage Guide

Version: **1.0.0**

The Storage Guide defines Pinglab's Pingstore filesystem convention for storing,
validating and consuming scientific runs. This file is the canonical guide.

## 1. Versioning

Version this guide independently of Pinglab, Demolab and the other guides.
Increment the major version when changed requirements make previously compliant
storage implementations or workflows require revision, the minor version for
compatible additions, and the patch version for corrections or clarifications
that do not change requirements. Update the version above and add a short entry
to the version history when changing the guide.

The guide version is separate from run schema versions such as `pingstore.run/v3`.
Changing this guide does not itself migrate existing runs or authorize a migration.

### 1.1. Version history

- **1.0.0** — Name and version the existing Storage Guide; storage requirements
  and run schemas remain unchanged.

## 2. Pingstore filesystem convention

Pingstore is a filesystem convention for immutable scientific runs, not a
service, database, catalogue, or general command-line interface.

```text
.pingstore/runs/<run-id>/
├── run.json         # Required: authoritative stage, inputs and provenance
├── export/          # Required: this stage's outputs
├── provenance/      # Optional: scripts, patches and retained evidence records
└── README.md        # Optional: human notes
```

This is `pingstore.run/v3`, written by new staged local/HPC executions. No other
root entries or symlinks are accepted. Compute/analyse exports may contain
subdirectories; present exports contain only flat regular files. No empty
presentation directory is created. Run IDs begin with the experiment and end
with execution source; v3 requires an explicit stage and input references.

`run.json` records execution, source provenance,
and `payload_digest`: SHA-256 of UTF-8 compact, sorted-key JSON of the sorted
file inventory (`path`, `size_bytes`, `sha256`). Paths are relative to the run,
ordered lexicographically. Include every payload file: export/, optional
provenance/ and README.md, including nested manifests; exclude only root run.json.
Directory entries and filesystem timestamps are not evidence or digest inputs.

Write to `.pingstore/runs/.<run-id>.tmp/`, finish the payload and any notes, write
run.json, validate layout and checksums, and rename to the visible run ID.
Failed runs remain hidden and are not evidence or publication input. A completed
run, including its README, is immutable. New analysis requires a new run.

For a present run, `export/` contains figures, tables and summaries such as
`numbers.json`.
Use names such as `rasters__canonical__seed42.png`, not nested folders.
Materialization validates the completed run and copies its entire export/ into
`.artifacts/<experiment>/`, without suffix filtering. Compute/analyse publication
is rejected, even if those exports contain image files. `_manifest.json` is a
compatibility projection for the publishing engine; run.json remains authoritative.
Execution scripts, patches and original declarations belong under provenance/,
not in the publication output. Numerical arrays, checkpoints and model bundles
belong under the compute/analyse export/.

A compact scientific export may declare `export_root` in run.json, for example
`export/cells` for an exp022 model bank. Readers of an explicitly selected run use
that directory; v3 records without this field default to export/. The path must
remain within export/. This does not change which run is selected or compact
other runs implicitly.

`collections.json`, when present, is a manually maintained mapping from named
views to arrays of explicit run IDs. No latest/official selection is inferred.
Existing artifact views without a locally retained backing run are not rebuilt
or silently replaced during a storage migration.

## 3. Legacy compatibility and enforcement

Completed `pingstore.run/v2` runs remain readable and unchanged. They require
exactly run.json, README.md, export/ and presentation/; export_root defaults to
export/state/. Legacy native capture and the v1-to-v2 migration utility explicitly
continue writing v2 until their producers are separately migrated. They must not
label mixed scientific/presentation output as v3 without splitting execution.

The validator selects the exact layout by schema, not by guessing from folders.
`layout.export_directory()` and `layout.presentation_directory()` centralize
version-aware consumption after validation. Untyped v2 runs may expose their
presentation/; typed v2 compute/analyse runs cannot be published or discovered.
The Demolab discovery field is still named `presentation`, even when it points
to a v3 run's export/. Consumers do not need a new discovery protocol.

New stages always write v3. Changing the writer does not rewrite existing data.
Data migration requires separate authorization and recoverable originals; no
general v2-to-v3 migration utility is introduced here. The three local staged
exp022 runs were subsequently migrated with unchanged IDs, byte-preserved outputs
and updated input pins; see [their migration record](../../experiments/exp022/README.md).
Incomplete v2 stage reservations must finish using their original code or be
replaced by new v3 reservations, never silently reused under a different schema.

## 4. Independent experiment stages

New staged runs use `<experiment>-rNNN-<stage>-<origin>`, for example
`exp022-r001-compute-local`, `exp022-r002-analyse-local` and
`exp022-r003-present-local`. Earlier stage-first IDs remain readable for
historical evidence and backups; the allocator only creates counter-first IDs.
Historical non-staged IDs remain unchanged. The explicitly authorized reorder
of the three new exp022 runs is recorded in that experiment's migration guide.
`stage` is compute, analyse or present: an execution label, not a mutable state.
The numeric allocator accounts for visible and hidden identities; exclusive
temporary-directory creation reserves each full identity before work/dispatch.
Wilkes campaigns use `--execution-origin slurm-wilkes` to reserve that suffix
before submission; the default `campaign` permits mixed workers without claiming
one training host. Individual job IDs remain in cell execution records. RunPod dispatch similarly
reserves a run-specific remote namespace before creating pods.

Every v3 `run.json` requires `stage` and `inputs`. Typed v2 records use the same
input contract. `inputs` maps explicit roles to
`{run_id, payload_digest, run_json_sha256}`; an initial compute run has
an empty mapping. Both payload and authoritative manifest are pinned. Readers
validate exact inputs before use and completion, and never silently choose a
replacement. Large scientific inputs are referenced, not recopied downstream.
Keep all referenced source runs when transferring a derived result: a presentation
run alone is not a backup of the scientific evidence.

`stages.py` owns reservation, source resolution and atomic completion as a Python
library, not a management CLI. It records actual commands, code/lock provenance,
uncommitted code patches and execution time. Each stage completes independently;
it never launches a different stage or calls materialization. Failed work remains
hidden. An unused scheduler reservation is not a completed run or a cache source.

Historical import is a new operation with a new timestamp and its actual local
origin, not a re-execution of historical jobs. Exp022's explicit
`compute.py --import-source RUN` preserves original scientific bytes and retains
the old run.json and any README inside provenance/, with an input reference to the
unchanged original. Historical SLURM attempts and inherited/repaired lineage are
therefore separate from the new import operation. The original is not deleted,
renamed, reselected, or rewritten; the separate base bank is unaffected.

See the [Experiment Runner Guide](../../experiments/README.md) for commands and
responsibility boundaries. The payload inventory/digest algorithm is shared by
v2 and v3; moving a file still changes its checksum inventory and dependent pins.

## 5. Demolab discovery

Discovery includes present runs with at least one nonempty export file other
than the compatibility bookkeeping names in `layout.RECORD_NAMES`. Untyped v2
runs use presentation/ with the same content check. Eligibility comes from the
authoritative stage, never a name substring or extension. Empty or bookkeeping-only
outputs and compute/analyse runs are omitted **after** full validation; malformed
compute runs still fail discovery. Numbers, tables, figures and videos all qualify.

`pingstore discover` is the narrow read-only CLI exception: it projects validated
completed runs into Demolab's generic discovery protocol. It does not choose a
run, write an index, copy presentation files, or manage local or remote storage.

```sh
uv run pingstore discover
uv run pingstore discover --source .pingstore/runs
# Equivalent module entry point:
uv run python -m pingstore discover --source .pingstore/runs
```

The source is the directory containing run folders, not the `.pingstore` parent.
Resolution is explicit `--source`, then `DEMOLAB_PREVIEW_SOURCE`, then
`.pingstore/runs` relative to the working directory. The source must exist and
must not use symlinks. Discovery inspects only immediate visible directories;
hidden entries, regular files and symlink candidates are ignored. Every visible
candidate must pass its schema's full layout and payload checksum checks. A malformed
candidate fails the command with an error on stderr and no JSON on stdout;
an empty store returns `[]`. Runs and their payloads are never modified.

Output is one JSON array, sorted by run directory name. Each record contains
`id` and `label` from `run.json.run_id`, `experiment` from `run.json.experiment`,
`created_at` from `run.json.created_at` normalized to UTC, and `presentation` as
`<run-id>/export` (v3) or `<run-id>/presentation` (v2), relative to the source.
Timestamps must include a timezone; filesystem dates and presentation-side
metadata are not substitutes.

With a Demolab version supporting command-based preview discovery, add:

```yaml
preview:
  source: .pingstore/runs
  discover: [uv, run, pingstore, discover]
  articles:
    exp092: [exp023, exp025, exp038, exp048]
    exp093:
      baseline: [exp025, exp038, exp049]
      candidate: [exp025, exp038, exp049]
```

Demolab supplies `DEMOLAB_PREVIEW_SOURCE`. Omitted articles automatically match
their IDs to `experiment`; lists declare multiple inputs and named groups allow
independent comparison selections. Article-scoped `data-file()` bindings are
still required in writings: discovery alone does not redirect hardcoded paths.
The current Demolab implementation defaults preview selections to Latest by
metadata timestamp and provides Published/default explicitly. Discovery itself
does not pick a default or change published inputs. This example does not enable
preview in Pinglab or migrate its writings.

Demolab has no separate checksum-validation callback, so this command verifies
**all candidate payloads on every invocation**, including `export/`. No metadata
cache or unsafe fast mode bypasses validation. Large stores can exceed Demolab's
current 30-second discovery timeout; efficient validation requires a separate
protocol change, not dropping checks here. Ordinary builds do not call discovery.

## 6. One-time v1 migration

`migrate_v2.py` is a narrowly scoped migration utility, not a Pingstore management
CLI. It does not retrain, upload, prune, or select published results.

```sh
uv run python -m pingstore.migrate_v2 prepare .pingstore .scratch/pingstore-v2-migration
# Inspect migration.json and the prepared store before the separate activation.
uv run python -m pingstore.migrate_v2 activate .pingstore .scratch/pingstore-v2-migration
```

The working directory must be new, outside the source store, on the same
filesystem. Preparation verifies the v1 digest, copies into hidden runs, records
every original file's destination/size/hash, retains original manifests and
notes under export/provenance/format-v1/, validates v2, and verifies the source did
not change. Unknown classifications and flattened-name collisions stop migration.
A known relocated-root-README case is accepted only if putting its exact bytes
back at the original inventory path reconstructs the stored v1 digest exactly;
the verification basis is recorded in the migrated manifest.

Activation requires stopped writers/readers, rechecks source and prepared
inventories, renames the original store to WORKDIR/rollback, and renames prepared
into place. This is an explicitly approved one-time format migration exception
to completed-run immutability. The two renames are recoverable, not a claim of
atomic whole-store exchange. No rollback copy is deleted automatically.

If interrupted, inspect the journal and use the same paths:

```sh
uv run python -m pingstore.migrate_v2 recover .pingstore .scratch/pingstore-v2-migration
```

Recovery restores the verified original when the source is absent, or records
successful activation when the verified new store is already present. It refuses
ambiguous states. After success, a deliberate rollback requires stopping users,
retaining the new store separately, and restoring WORKDIR/rollback together with
compatible code. Do not merge old and new run trees.

Remote R2/HPC stores are not migrated by local activation. Backups and restores
must retain the entire run in its recorded schema, not just publication outputs.
Never prune a source because a presentation copy exists; verify independent
backups first.
