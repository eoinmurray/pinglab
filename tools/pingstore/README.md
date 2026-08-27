# Pingstore

Pingstore is a filesystem convention for immutable scientific runs, not a
service, database, catalogue, or general command-line interface.

```text
.pingstore/runs/<run-id>/
├── run.json
├── README.md
├── export/           # Scientific outputs and execution records; nesting allowed
└── presentation/   # Copyable publication inputs; regular files only
```

These four entries are required, including empty export/presentation directories.
No additional root entries or symlinks are accepted. Run IDs begin with the
experiment and end with execution source. Local and HPC capture use this layout.

`run.json` uses `schema: pingstore.run/v2`. It records execution, source provenance,
and `payload_digest`: SHA-256 of UTF-8 compact, sorted-key JSON of the sorted
file inventory (`path`, `size_bytes`, `sha256`). Paths are relative to the run,
ordered lexicographically. Include README.md and every file under export/ and
presentation/, including nested manifests; exclude only root run.json.
Directory entries and filesystem timestamps are not evidence or digest inputs.

Write to `.pingstore/runs/.<run-id>.tmp/`, finish the README and payload, write
run.json, validate layout and checksums, and rename to the visible run ID.
Failed runs remain hidden and are not evidence or publication input. A completed
run, including its README, is immutable. New analysis requires a new run.

`presentation/` contains figures, tables and summaries such as `numbers.json`.
Use names such as `rasters__canonical__seed42.png`, not nested folders.
Materialization validates the completed run and copies only presentation/ into
`.artifacts/<experiment>/`, without suffix filtering. `_manifest.json` is a
compatibility projection for the publishing engine; run.json remains authoritative.
Execution scripts, patches and original declarations belong under export/provenance/.
Numerical arrays, checkpoints and model bundles belong under export/.

A compact scientific export may declare `export_root` in run.json, for example
`export/cells` for an exp022 model bank. Readers of an explicitly selected run use
that directory; records without this field default to export/state/. The path must remain within
export/. This does not change which run is selected or compact other runs implicitly.

`collections.json`, when present, is a manually maintained mapping from named
views to arrays of explicit run IDs. No latest/official selection is inferred.
Existing artifact views without a locally retained backing run are not rebuilt
or silently replaced during a storage migration.

## One-time v1 migration

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
must retain the entire v2 run, not just export/ or presentation/. Never prune a
source because a presentation copy exists; verify independent backups first.
