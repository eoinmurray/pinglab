# R2 Pingstore backup

The append-only backup lives in the existing `pinglab` R2 bucket at
`r2:pinglab/pingstore/runs`. Each completed local run is copied with its complete
v4 layout:

```text
pingstore/runs/<run-id>/
    run.json
    README.md
    export/
```

The source is `.pingstore/runs/`. Git continues to hold experiment code;
`run.json` holds execution provenance and input references. The backup contains
scientific exports and their authoritative records, rather than payload-only
campaign snapshots or a generated catalogue.

## Back up new runs

Use the external backup helper, separate from the narrow Pingstore CLI:

```sh
uv run python tools/backups/pingstore_r2.py --dry-run
uv run python tools/backups/pingstore_r2.py --confirm <complete-plan-hash>
```

Replace `<complete-plan-hash>` with the `sha256:...` value printed by the dry-run.
The plan names every new run, reports its total bytes, and binds the local
records and remote inventory. Confirmation rebuilds the plan and stops on drift.
Dry-run performs validation and remote inspection without uploading anything.

The helper validates every visible completed v4 run, its exact layout, export
checksum, and complete input graph. Hidden incomplete local runs are excluded.
It holds the shared Pingstore operation lock throughout the backup so pruning
cannot remove its local sources.

New runs upload their scientific files and README first. `rclone check --download`
then compares those remote bytes with their local sources. The helper revalidates
local evidence and the uploaded inventory before uploading `run.json` last, in
input-ancestry order. A final download comparison verifies those completion
records. An interrupted upload without `run.json` is an incomplete remote backup;
a later backup can resume it if existing objects match.

Existing completed backups are skipped after their file inventories and exact
`run.json` and README bytes are checked against the corresponding local run.
Existing scientific payloads are not downloaded again on routine incremental
backups. They were verified when first uploaded; recovery must independently
validate their recorded SHA-256 payload checksums.

Uploads use immutable copies: an existing conflicting object is an error.
No remote files are deleted, including runs that have since been pruned locally.
The first backup copies the local store; subsequent backups add completed run IDs.
An explicitly corrected local manifest or README will conflict with its older
backup. Updating an existing backed-up identity requires a separate explicit
operation; this helper does not silently overwrite it.

## Access and destination

The existing local rclone configuration provides remote `r2` and bucket `pinglab`.
Check configuration availability without printing credentials:

```sh
rclone listremotes
rclone lsf r2:pinglab/pingstore/runs --dirs-only
```

`PINGLAB_R2_REMOTE` and `PINGLAB_R2_BUCKET` override the defaults. An explicit
`--destination` must end in `/pingstore/runs`. An explicit `--source` accepts a
validated local runs directory. Authentication is supplied by rclone's local
configuration; credentials are never stored in this repository.

For a new machine, follow Cloudflare's [rclone setup guide](https://developers.cloudflare.com/r2/examples/rclone/)
with a bucket-scoped credential. Do not infer access from another machine's
configuration or an SSH alias.

## Recovery and historical archives

Restore explicitly selected completed runs and their full input ancestry into a
separate recovery directory. Validate every v4 root layout, scientific payload
checksum and input reference under the [Storage Guide](../tools/pingstore/README.md)
before activation or consumption. A remote `run.json` marks completion by this
backup workflow; it does not replace validation on recovery. Partial uploads
must not become operational runs.

Older campaign archives remain separate historical evidence. Their payload-only
layout is not a complete v4 run backup. This workflow neither migrates nor changes
those archives, and does not restore, publish, or execute experiments.

The implementation uses [rclone copy](https://rclone.org/commands/rclone_copy/)
with immutable checks and [download verification](https://rclone.org/commands/rclone_check/).
