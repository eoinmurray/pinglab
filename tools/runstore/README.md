# runstore data contract

`runstore` manages pinglab's local run directories, promoted publication
artifacts, and durable R2 archives. It is deliberately small: the filesystem is
the database, JSON records provenance, and SHA-256 records identity.

This document defines contract version `runstore/v1`. The implementation will
follow after this contract has been reviewed against the legacy data.

## Ownership

| Location | Purpose | Tracked in Git |
| --- | --- | --- |
| `runs/` | Local run/campaign state and derived outputs | No |
| `artifacts/` | Selected publication view for Demolab and GitHub Pages | Yes, except the built site |
| R2 `campaigns/<archive-id>/` | Verified durable archives | No |

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

An accepted production archive uses a caller-selected immutable identity:

```text
r2://pinglab/campaigns/<archive-id>/
```

This is the logical archive URI. A local rclone remote name is configuration,
not part of the durable identity.

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

## Version-1 tool boundary

The planned CLI has six operations:

```text
runstore init
runstore inspect
runstore promote
runstore archive
runstore verify
runstore restore
```

The archive commands default to the existing `r2:pinglab/campaigns` rclone
root. `--store /path/to/store` selects the filesystem backend used for local
tests and rehearsals; `PINGLAB_RUNSTORE_STORE` provides the same override.

```text
runstore archive <run-root> --archive-id <id> [--store <root>]
runstore verify <id> [--store <root>]
runstore restore <id> <new-destination> [--store <root>]
```

Remote verification streams every archived payload object and recomputes its
SHA-256. This is intentionally stronger than treating a successful transfer as
proof of durable contents.

`runstore` does not execute experiments, encode scientific dependencies, submit
Slurm jobs, judge scientific results, or automatically delete data. The
gamma-gated-sparsity collection orchestrator will call this lifecycle tool; it
will not duplicate the storage contract.

## Minimal example

[`examples/minimal-run`](examples/minimal-run) is a complete tiny ad-hoc run.
Its single payload is a derived `numbers.json`; its inventory size, file hash,
and aggregate payload digest are real and should validate unchanged.
