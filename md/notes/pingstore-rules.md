---
title: Pingstore Guide
author: Eoin Murray
collection: notes
description: Where Pingstore runs live, basic CLI usage and safe handling of experiment outputs.
created_at: "2026-09-11"
updated_at: "2026-10-20T11:22:49Z"
---

# Pingstore Guide

Pingstore holds Pinglab’s experiment runs. Code lives in Git; scientific outputs and execution records live in `.pingstore/`.

## Locations

- **Local runs:** `.pingstore/runs/`, relative to the repository root.
- **External backup:** `r2:pinglab/pingstore/runs`, accessed through rclone.
- **Implementation:** `tools/pingstore/`.

A completed run contains:

```text
.pingstore/runs/<run-id>/
├── run.json       # execution, provenance and input references
├── README.md      # human-readable run history
└── export/        # scientific outputs
```

An optional `.protected` file beside `run.json` retains the run and its input ancestry during local pruning. It is not backed up.

## Using the tool

Run commands from the repository root. Discover available commands and options through CLI help:

```sh
uv run pingstore --help
uv run pingstore discover --help
uv run pingstore backup --help
uv run pingstore prune --help
```

- **`discover`** validates completed runs and reports available presentation data.
- **`backup`** appends completed runs to the external backup.
- **`prune`** removes unused, unprotected local runs while retaining required ancestry.

Backup and prune use separate dry-run and confirmation invocations. Inspect the plan, then confirm using its complete hash and the same scope options. Neither operation runs automatically.

Limit pruning to one experiment with `prune --experiment exp141 --dry-run`.
Runs outside that scope and their full input ancestry are retained. Report usage
is still checked across all experiments, but missing report pins outside the
selected experiment do not block scoped pruning. Missing pins that declare or
name the selected experiment still fail; pruning all experiments requires every
explicit report pin to resolve. Existing runs and their input ancestry remain
validated in either mode.

Rclone credentials belong in its local configuration, never in the repository. Check available remotes with `rclone listremotes`.

## Working safely

- Use the shared Pingstore helpers to create and consume runs; let them enforce layout, checksums, allocation and locking.
- Completed scientific outputs are immutable. Produce a new run rather than modifying `export/`. Record explicit metadata corrections in the run’s README history.
- Select stage inputs explicitly by run ID and payload digest; never silently substitute the latest run.
- Hidden temporary runs are unfinished work, not usable evidence. Historical v2/v3 runs require separately authorized recovery or migration.
- Use `prune` rather than manually deleting completed runs. Protect results that must survive even when no current report uses them.
- Restore backups into a separate recovery directory and validate selected runs and their full input ancestry before use.
