# runstore data contract

`runstore` manages pinglab's local run directories, promoted publication
artifacts, and durable R2 archives. It is deliberately small: the filesystem is
the database, JSON records provenance, and SHA-256 records identity.

This document defines and documents contract version `runstore/v1`.

## Ownership

| Location | Purpose | Tracked in Git |
| --- | --- | --- |
| `runs/` | Local run/campaign state and derived outputs | No |
| `artifacts/` | Selected publication view for Demolab and GitHub Pages | Yes, except the built site |
| R2 `campaigns/<namespace>/<archive-id>/` | Verified durable archives | No |

`artifacts/` is deliberately a compact publication view, not a second run
archive. It may contain provenance metadata, `numbers.json`, compact derived
tables, final figures, and rendered publications. Raw arrays, checkpoints,
caches, repeated inputs, and other reconstructable intermediates remain in the
isolated run and its verified R2 archive. Existing historical binaries are
grandfathered until the planned history migration; new ones are rejected by
CI.

An isolated run directory is the source of truth. Running an experiment must
not update `artifacts/` as a side effect. Promotion is a separate reviewed
operation.

## Directory layouts

An ad-hoc experiment run uses:

```text
runs/adhoc/<experiment>/<run-id>/
├── state/
├── derived/artifacts/data/<experiment>/
├── logs/
├── run.json
└── inventory.json
```

A collection campaign uses:

```text
runs/campaigns/<campaign-id>/
├── exp022/
├── downstream/
├── derived/artifacts/
├── logs/
├── run.json
└── inventory.json
```

The same internal campaign layout may live beneath an explicit persistent root
on Wilkes3. Paths recorded in the contract are always relative to the run root.
Absolute host paths are optional diagnostics and are never durable identities.

Restores go to a new explicit destination, conventionally:

```text
runs/restored/<archive-id>/
```

## `run.json`

`run.json` identifies the execution. Required version-1 fields are:

- `contract_version`: exactly `runstore/v1`;
- `run_id`: unique within pinglab's run namespace;
- `kind`: `adhoc`, `campaign`, or `legacy`;
- `status`: `planned`, `running`, `complete`, `failed`, or `legacy`;
- `created_at_utc`: RFC 3339 UTC timestamp;
- `source.git_commit`: full generating commit when known, otherwise `null`;
- `source.git_clean`: boolean when known, otherwise `null`;
- `source.lockfile`: repository-relative lockfile path and SHA-256 when known;
- `execution`: experiment or collection identity plus the invoked command;
- `upstream`: referenced run/campaign identities;
- `archive`: stable archive identity and R2 URI, or `null` before archival;
- `provenance_notes`: honest free-text qualifications, especially for legacy data.

Unknown legacy provenance is represented by `null` plus a note. It must not be
reconstructed from the current checkout and presented as historical fact.

Commands are JSON arrays, not shell strings. This preserves argument boundaries
without requiring shell parsing.

## `inventory.json`

`inventory.json` describes the payload beneath the run root. Its `files` array
contains one entry per payload file:

- `path`: normalized POSIX path relative to the run root;
- `size_bytes`: exact byte length;
- `sha256`: lowercase 64-character SHA-256;
- `role`: `state`, `derived`, or `log`.

Entries are sorted lexicographically by `path`. Paths must not be absolute,
contain `..`, or name `run.json`/`inventory.json`.

`payload_digest` is the SHA-256 of the compact canonical JSON encoding of the
`files` array with object keys sorted. It identifies the complete payload while
avoiding a circular hash over the inventory itself. `file_count` and
`total_size_bytes` must agree with the entries.

The archive stores `run.json`, `inventory.json`, and every inventoried payload
file. Verification checks the two manifests structurally, recomputes the
payload inventory, and compares every size and hash.

## Archive identity and R2

Production archives are separated by purpose:

```text
r2://pinglab/campaigns/
├── legacy/<archive-id>/
├── adhoc/<archive-id>/
└── gold-star/<campaign-id>/
```

`legacy` contains historical data migrated into the contract. `adhoc` contains
accepted one-off experiment runs worth retaining. `gold-star` contains complete
collection campaigns executed from a frozen commit. Smoke runs remain local
unless there is a specific reason to retain one.

The caller selects the namespace through `--store` and
`--logical-base-uri`; the archive ID is immutable within that namespace. The
logical R2 URI is durable provenance. A local rclone remote name is merely
machine configuration.

Filesystem rehearsals use `file://<absolute-path>/<archive-id>`. They exercise
the same manifests and verification rules but are not durable R2 archives.

Version 1 refuses an archive operation when that identity already contains
objects. Correction or replacement therefore receives a new archive ID; no
version-migration framework is required.

A successful upload is not sufficient. `verify` must compare the remote
payload with `inventory.json`, and `restore` must write into a new destination
and perform the same verification locally. Only then is the original local run
eligible for manual deletion.

## Promotion and reverse provenance

Promotion copies an accepted directory from:

```text
<run-root>/derived/artifacts/data/<experiment>/
```

to:

```text
artifacts/data/<experiment>/
```

The copy is built in a sibling staging directory, checked, and renamed into
place. The source run remains unchanged.

Every promoted experiment directory contains `_provenance.json` with:

- contract version, run ID, and campaign ID when applicable;
- generating Git commit;
- source directory relative to the run root;
- mapping from promoted files to source-relative paths and SHA-256 hashes;
- the source inventory's `payload_digest`;
- promotion timestamp;
- stable R2 archive identity when available.

The reverse link is metadata, not a symlink. Local absolute paths and literal
restore commands are excluded because they become stale.

## Commands and operator sequence

The core lifecycle CLI has six operations:

```text
runstore init
runstore inspect
runstore promote
runstore archive
runstore verify
runstore restore
```

Campaign discovery and the local publication view add three operator commands:

```bash
runstore campaigns
runstore activate <local-path-or-campaign-id>
runstore current
```

`campaigns` scans only the configured local roots (`runs/campaigns` and
`runs/restored` by default) and the selected archive store. Repeat
`--local-root` to use explicit roots, set `PINGLAB_RUNSTORE_LOCAL_ROOTS` to an
OS-path-separated list, or pass `--local-only` when R2 is unavailable. It does
not crawl the filesystem. R2-only campaigns are listed but must be restored
before activation.

`activate` accepts a complete local collection campaign, stages a copy of the
existing `artifacts/data` tree, replaces every experiment supplied by that
campaign, writes reverse provenance and `.runstore-view.json`, then swaps the
whole data tree into place. Unrelated experiment directories are preserved. A
failed activation restores the previous tree. Run `demolab build` afterwards
to regenerate PDFs and the local/GitHub Pages site from the selected data.

`current` verifies that every experiment named by `.runstore-view.json` belongs
to the active campaign and, by default, re-hashes every promoted file. Use
`--no-verify-files` only for a quick metadata check.

Create a unique run root before executing science. Existing destinations are
always refused. `init` captures the current Git commit, dirty-tree state, and
`uv.lock` digest:

```bash
runstore init runs/adhoc/exp082/<run-id> \
  --run-id <run-id> \
  --kind adhoc \
  --experiment exp082 \
  --command uv run python experiments/exp082.py --out-dir <run-root>
```

For a collection campaign, use `--kind campaign --collection
gamma-gated-sparsity`. Once all required outputs exist, the collection
orchestrator finalizes the run. This changes `planned` or `running` to
`complete` and freezes the immutable inventory; no manifest hand-edit is
needed:

```bash
runstore inspect <run-root> --finalize
```

`--write-inventory` remains available for complete or honestly labelled legacy
runs whose manifest already has its final status.

`runstore inspect <run-root> --write-inventory` writes `inventory.json`
atomically after hashing the payload. It requires a valid `run.json` and refuses
to replace an existing inventory; regeneration therefore requires a deliberate
manual review/removal of the previous manifest.

The archive commands default to the existing `r2:pinglab/campaigns` rclone
root. `--store /path/to/store` selects the filesystem backend used for local
tests and rehearsals; `PINGLAB_RUNSTORE_STORE` provides the same override.

```text
runstore archive <run-root> --archive-id <id> [--store <root>]
runstore verify <id> [--store <root>]
runstore restore <id> <new-destination> [--store <root>]
```

Production calls select the appropriate namespace explicitly. For example:

```bash
runstore archive <run-root> \
  --archive-id <campaign-id> \
  --store r2:pinglab/campaigns/gold-star \
  --logical-base-uri r2://pinglab/campaigns/gold-star
```

Remote verification streams every archived payload object and recomputes its
SHA-256. This is intentionally stronger than treating a successful transfer as
proof of durable contents.

Promotion is a separate acceptance action and is allowed only for runs marked
`complete` or `legacy` with a valid inventory:

```bash
runstore promote <run-root> exp082
```

The source must contain `numbers.json` and at least one PDF, PNG, or SVG below
`derived/artifacts/data/<experiment>/`. Promotion verifies the entire source
inventory, copies into a sibling staging directory, verifies every copied file,
adds `_provenance.json`, and then swaps the accepted directory into
`artifacts/data/`. The source run is never modified. An existing publication
view is replaced only after the staged replacement has passed validation; Git
retains its previous version.

`_provenance.json` records:

- contract version, run ID, and campaign ID where applicable;
- generating Git commit and stable archive identity where available;
- source directory and source inventory payload digest;
- promotion timestamp;
- each displayed file's publication-relative path, source-relative path, size,
  and SHA-256.

`runstore` does not execute experiments, encode scientific dependencies, submit
Slurm jobs, judge scientific results, or automatically delete data. The
gamma-gated-sparsity collection orchestrator will call this lifecycle tool; it
will not duplicate the storage contract.

## Minimal example

[`examples/minimal-run`](examples/minimal-run) is a complete tiny ad-hoc run.
Its single payload is a derived `numbers.json`; its inventory size, file hash,
and aggregate payload digest are real and should validate unchanged.

## Interface for collection orchestration

An isolated experiment subprocess receives this all-or-none environment
contract:

```text
PINGLAB_REQUIRE_ISOLATED=1
PINGLAB_RUN_STATE_DIR=<campaign-root>/downstream/<experiment>
PINGLAB_RUN_DERIVED_DIR=<campaign-root>/derived/artifacts/data/<experiment>
PINGLAB_RUN_LOG_DIR=<campaign-root>/logs/<experiment>
```

All three directories must be absolute and distinct. With
`PINGLAB_REQUIRE_ISOLATED=1`, missing paths are fatal. An isolated derived path
under the repository's active `artifacts/` tree is rejected. Experiment-specific
upstream roots remain explicit alongside this generic output contract; for
example, exp024 receives `PINGLAB_TRAINING_ROOT=<campaign-root>/exp022/cells`.

exp024 is the version-1 representative integration. Its focused test invokes
the runner as a real subprocess, proves the active publication view is
unchanged, then finalizes, archives, restores, and promotes the isolated result
through the `runstore` CLI. #70 adopts the remaining collection runners through
the same interface as they enter the checked-in dependency graph.

The gamma-gated-sparsity orchestrator may depend only on this sequence:

1. `runstore init` creates a new campaign root and `run.json`.
2. Experiments write state, logs, and derived artifact candidates beneath it.
3. `runstore inspect --finalize` marks the successful campaign `complete` and
   freezes its payload identity.
4. `runstore archive`, `verify`, and `restore` establish durable recoverability.
5. `runstore promote` explicitly updates selected UI-visible experiment data.

The orchestrator must not implement its own archive layout, provenance format,
promotion copy, or R2 verification logic.
