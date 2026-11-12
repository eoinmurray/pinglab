---
title: Pingstore Operations
author: Eoin Murray
collection: notes
description: Run insertion, deletion, backup, restore and migration.
created_at: "2026-10-29T17:42:34Z"
updated_at: "2026-11-12T20:22:47Z"
---

# Pingstore operations

## Local runs

Pingstore itself uses only the Python standard library. To install it without the
scientific environment, run `pip install ./libraries/pingstore`; backup and restore
additionally require the rclone executable. The repository environment also exposes
the same `pingstore` command.

```sh
uv run pingstore list --experiment exp174
uv run pingstore get exp174-r013-present
uv run pingstore insert /path/to/completed-run
uv run pingstore delete exp174-r013-present --dry-run
uv run pingstore delete exp174-r013-present
```

Use `--root /path/to/.pingstore` before the command to select another store.
Insertion requires a simple `run.json` in the source, or `--record /path/to/run.json`.
It copies into a temporary directory and completes atomically. Existing IDs cannot
be overwritten. Python can write directly without a final payload copy:

```python
from pathlib import Path
from pingstore.library import Store

store = Store(Path('.pingstore'))
with store.create('exp174', stage='present', depends_on=['exp174-r012-analyse']) as run:
    (run.directory / 'result.json').write_text('{"score": 0.9}')
```

Experiment scripts may keep using `stage_run(repo, experiment, stage, inputs=...)`.
Use `run.details` for scientific execution/configuration/provenance. After completion,
`source_run(...).record` is the small run metadata, and `.details` reads `details.json`.
No automatic export flattening or scientific payload hashing occurs.

## R2

```sh
uv run pingstore backup
uv run pingstore backup exp174-r013-present
uv run pingstore restore exp174-r013-present
uv run pingstore delete exp174-r013-present --storage r2 --dry-run
uv run pingstore delete exp174-r013-present --storage r2
```

The new default destination is `r2:pinglab/pingstore-v2`, with `runs/`, `receipts/`
and a rebuildable `index.json`. The old R2 archive is untouched. Select another
prefix with `--destination REMOTE:BUCKET/PREFIX`. Credentials stay in private
rclone configuration. Only one machine may write a destination at a time; a
local writer lock serializes commands on that machine.

Backup uses concurrent rclone transfers and supported backend integrity checks,
then writes a completion receipt. It does not use `rclone check --download`.
Existing completed IDs must have identical run metadata and are skipped without
reading payloads. Completed payloads must not be edited behind Pingstore's back.
Interrupted uploads can be retried; the small index is rebuildable from receipts.
Restore downloads named runs only and checks completion metadata and file sizes;
it does not claim an independent content audit. Dependency restoration is explicit.

Deletion shows direct dependants and permits confirmed breakage. `--yes` confirms
the displayed scope for scripts. Local deletion never deletes R2 and remote deletion
never deletes local copies. No recursive delete or automatic unused-run pruning exists.

## Migration

First inspect conversion into a separate store:

```sh
uv run pingstore --root /path/to/new/.pingstore migrate /path/to/old/.pingstore
uv run pingstore --root /path/to/new/.pingstore migrate /path/to/old/.pingstore --apply
```

Migration preserves original metadata under `migration/originals/`, copies retained
files without rewriting them, and reports unresolved dependencies. It preserves
completion order and materialises consolidated logical sources as ordinary runs.
It never overwrites the original store. Validate selected reports and payloads before
switching stores. Re-running accepts already converted matching records; an incomplete
individual conversion needs inspection before retrying. Use the new backup prefix
for the converted store; retiring the old local/R2 archive is a separate action.

For this repository, a concrete cutover is:

```sh
uv run pingstore --root ../pinglab-runs-simple migrate .pingstore
uv run pingstore --root ../pinglab-runs-simple migrate .pingstore --apply
cd site
PINGLAB_RUNS_ROOT=../../pinglab-runs-simple/runs bun run build
cd ..
```

Inspect the migration report, unresolved dependencies, unfinished runs, selections
and rendered reports. Finish or collect pending jobs before switching the default
store. Keep the old directory as an archive; rename it out of `.pingstore` and move
the validated replacement into `.pingstore` only when ready. Those directory changes
are explicit user operations, not effects of `migrate`. The existing R2 archive stays
at its old prefix until separately retired. Remote-only old runs must first be
restored into an old-format staging store with their exact original metadata, then
converted and backed up to the new prefix; no legacy reader runs during rendering.
