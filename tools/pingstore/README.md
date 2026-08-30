# Storage Guide

Version: **4.3.0**

This guide defines Pingstore's filesystem convention. Pingstore is not a
service, database, catalogue, lifecycle manager, or general management CLI.

## 1. Contract

Every completed operational run uses `pingstore.run/v4` and has exactly:

```text
.pingstore/runs/<run-id>/
├── run.json
├── README.md
└── export/
```

- `run.json` is the authoritative machine-readable record.
- `README.md` is the mandatory human-readable history.
- `export/` contains only scientific data. Run-wide and single-file unit outputs
  are flat; only genuine multi-file scientific units use a directory.

There is no operational `provenance/` directory. Do not duplicate the command,
configuration, environment, source revision, or timing in sidecar manifests or
replay scripts. Record them once in `run.json`. Historical migration material
belongs in a recoverable migration archive outside `.pingstore/runs/`.

Execution logs, commands, execution configurations, environment captures,
timing, import mappings, and copied source records are metadata: keep them in
`run.json`, not `export/`. A per-unit scientific definition may remain as a role
file when it is data needed to interpret that unit; it must not duplicate the
run's execution metadata. Writers may use `.scratch/` inside a hidden incomplete
run, but completion discards it. Compute and analyse exports permit exactly one
unit-directory level. A unit directory must contain at least two scientifically
related files. A single-file unit is named `<unit-id>--<role-file>` directly
under `export/`. Unit IDs are canonical scientific identities, not generic
containers such as `data`, `jobs`, `cells`, or `misc`; role filenames are short
and repeatable.

Use `recording.npz` for raw multimodal simulation time series, `spikes.npz` for
raw spike-only output, and `rasters.npz` only for transformed raster/event data.
The ambiguous aliases `snapshot.npz` and `recordings.npz` are forbidden. Common
roles use their standard names; genuinely domain-specific descriptive filenames
remain permitted. Present exports retain descriptive flat filenames.
Present exports contain only flat regular files. No other root entries or
symlinks are allowed in a completed run.

## 2. `run.json`

A v4 record includes:

- `schema`, `run_id`, `experiment`, `collection`, `stage`, `origin`, and
  `created_at`;
- `inputs`, mapping roles to `{run_id, payload_digest}`;
- `execution`, containing the actual command, working directory, host,
  configuration, environment when relevant, and start/completion times;
- `provenance`, containing compact source identity such as the Git commit,
  dirty flags, and lockfile checksum;
- `payload_digest`, the digest of `export/` only;
- optional experiment-specific scientific metadata.

Do not retain generated replay scripts in `export/`. Do not pin the bytes of an
upstream `run.json`. An input is scientifically fixed
by its run identity and immutable export digest. This permits corrections to
metadata or README history without invalidating every descendant.

`payload_digest` is SHA-256 of UTF-8 compact, sorted-key JSON describing the
lexicographically sorted export inventory. Each row contains `path`,
`size_bytes`, and the file SHA-256. `run.json` and `README.md` are not payload
bytes. Changing anything under `export/` changes scientific identity.

## 3. README history

Every writer creates the README before execution and appends successful
completion. It states the run, stage, explicit inputs, source revision, origin,
and dated history. Imports and migrations append what changed, what did not run,
and where the recoverable original is stored.

README and `run.json` metadata may be corrected only by an explicit operation
that appends the correction to the README. Export bytes remain immutable.

## 4. Writing and validation

Reserve a new identity before local or scheduler execution. New IDs use
`<experiment>-rNNN-<stage>`, such as `exp022-r001-compute`; execution location
belongs in `origin` and `execution`, not the ID.

Build the run under `.pingstore/runs/.<run-id>.tmp/`. Hidden working directories
may contain temporary reservation and writer-lock files. Remove those files,
finish `run.json`, README, and export, validate the exact layout and digest, then
atomically rename the directory to its visible identity. Failed work remains
hidden and is not operational evidence.

Readers must validate v4 layout and the export digest before consumption.
Compute starts with empty inputs. Analyse and present name every input explicitly;
they never select “latest”, launch upstream work, or silently substitute a run.

`export_root` is obsolete. Readers use the whole export and resolve explicit
unit IDs. Hidden writers may temporarily use deeper tool-native paths, but the
shared completion helper canonicalizes them before validation and visibility.

## 5. Discovery and publication

`pingstore discover` is the sole operational CLI exception. It validates visible
completed v4 runs and emits Demolab discovery JSON for populated present runs.
It does not select, mutate, materialize, upload, prune, or persist a catalogue.

```sh
uv run pingstore discover
uv run pingstore discover --source .pingstore/runs
```

Materialization validates an explicitly selected present run and copies its flat
`export/` to `.artifacts/<experiment>/`. Publishing compute or analyse runs is
rejected. Presentation metadata is read from `run.json`; exports do not contain
compatibility manifests.

`collections.json`, when present, maps named views to explicit run-ID arrays.
No official or latest selection is inferred.

## 6. Historical schemas and migration

V2 and v3 runs are historical evidence, not operational inputs. They may be
inspected by explicit migration/recovery tooling but not consumed, discovered,
materialized, completed, or silently relabelled.

A migration to v4 must:

1. validate every source run under its original schema;
2. create a recoverable archive of original manifests, README files, and root
   provenance trees;
3. preserve scientific export bytes and archive provenance or metadata rather
   than moving it into `export/`;
4. generate or append the human-readable README history;
5. replace manifest-byte pins with run-ID and export-digest pins;
6. validate every completed v4 run and the complete dependency graph before
   activation; and
7. leave hidden incomplete reservations unchanged as historical failed work.

Migration does not authorize experiment execution, publication, pruning, remote
store changes, or deletion of the recovery archive.

## 7. Version history

- **4.3.0** — Flatten singleton scientific units, require at least two files for
  unit directories, and standardize simulation recording role names.
- **4.2.0** — Standardize compute/analyse exports as run-wide root files plus
  `export/<scientific-unit-id>/<artifact-role>`, with no deeper nesting.
- **4.1.0** — Make exports scientific-data-only, prefer flat descriptive files,
  discard writer scratch space, and remove export evidence/metadata sidecars.
- **4.0.0** — Make README mandatory; reduce completed roots to `run.json`,
  `README.md`, and `export/`; digest only immutable export bytes; remove manifest
  byte pins and operational provenance sidecars; archive v3 provenance outside
  operational runs.
- **3.0.0** — Introduced source-neutral staged IDs.
- **2.0.0** — Required v3 for operational paths.
- **1.0.0** — Versioned the storage guide.
