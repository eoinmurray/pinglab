---
title: Pingstore Guide
author: Eoin Murray
collection: notes
description: Where Pingstore runs live, basic CLI usage and safe handling of experiment outputs.
created_at: "2026-09-11"
updated_at: "2026-10-27T14:36:26Z"
---

# Pingstore Guide

Pingstore holds Pinglab’s experiment runs. Code lives in Git; scientific outputs and execution records live in local `.pingstore/` and verified R2 backups. The website can read completed backups without a local run copy.

## Locations

1. **Local runs:** `.pingstore/runs/`, relative to the repository root.
2. **External backup:** `r2:pinglab/pingstore/runs`, accessed through rclone.
3. **Implementation:** `tools/pingstore/`.

The implementation has three Python source files. `core.py` owns validation,
run creation, allocation, provenance, artifact handling and storage maintenance.
`library.py` exports the experiment-facing API from core; `cli.py` implements
the command interface using the same core. Experiments and execution adapters
import from `pingstore.library`; there are no compatibility modules for the old
import paths. Tests remain in `tools/pingstore/tests/`.

```python
from pingstore.library import source_run, stage_run

source = source_run(repo / ".pingstore", compute_id, stage="compute")
with stage_run(repo, "exp141", "analyse", inputs={"compute": source}) as run:
    # Write scientific outputs beneath run.export.
    pass
```

A completed run contains:

```text
.pingstore/runs/<run-id>/
├── run.json       # execution, provenance and input references
├── README.md      # human-readable run history
└── export/        # scientific outputs
```

An optional `.protected` file beside `run.json` retains the run and its input ancestry during local pruning and eviction. It is not backed up.

## Using the tool

Run commands from the repository root. Use the installed `pingstore` entry point
below, or `python -m pingstore.cli`
with the same arguments. The package is a namespace containing exactly the three
source modules; `python -m pingstore` is no longer an entry point.

```sh
uv run pingstore --help
uv run pingstore status --help
uv run pingstore discover --help
uv run pingstore backup --help
uv run pingstore prune --help
uv run pingstore prune-remote --help
uv run pingstore evict --help
uv run pingstore backup-papers --help
```

1. **`discover`** validates completed runs and reports available presentation data.
2. **`backup`** appends completed runs to R2, verifies their bytes and publishes the public checksummed run catalog. Existing uncataloged backups are verified before catalog publication.
3. **`prune`** removes unused, unprotected local runs while retaining required ancestry.

Backup, prune and eviction use separate dry-run and confirmation invocations. Inspect the plan, then confirm using its complete hash and the same scope options. None runs automatically.

4. **`evict`** removes verified local backup copies, including runs used by reports. It preserves R2 history, local protections, unfinished-work inputs and allocation counters. Confirmation downloads and compares selected remote files before removing local directories.
5. **`backup-papers`** is retired: public PDF uploads are disabled. Papers stay local for localhost; production uses publisher or authorised repository links from `papers/index.yaml`.
6. **`prune-remote`** removes unused R2 runs for explicit, repeatable
   `--experiment` selections. It requires the current selected evidence to be
   backed up, retains protected runs and required ancestry, and conditionally
   replaces the catalog before deleting the planned objects. Use separate
   `--dry-run` and `--confirm PLAN_HASH` invocations with the same scope.
   Refresh deployed reports to the retained presentation before removing assets
   referenced by an earlier deployment.

To reclaim space without removing scientific history, prefer eviction:

```sh
uv run pingstore evict --all --dry-run
uv run pingstore evict --all --confirm PLAN_HASH
# Or limit the operation to explicit IDs, repeating --run when needed:
uv run pingstore evict --run exp141-r001-compute --dry-run
```

The plan requires matching public and authenticated catalogs, matching local
file checksums, and complete digest-matched remote ancestry. Runs absent from
the catalog stay local. Existing-but-changed copies fail rather than being
silently discarded. Do not manually remove run directories or allocation state.
Eviction does not make Python compute stages read remote inputs; restore their
explicit inputs and full ancestry into a recovery directory and validate them
before scientific execution. Backup can use cataloged remote-only ancestors.

The public website catalog is `https://assets.eoinmurray.info/pingstore/catalog.json`.
It embeds completed records and per-file SHA-256 checksums. The server-side Astro
reader checks downloaded files against those checksums. Localhost prefers local
copies and otherwise reads R2; production builds read R2 exclusively. Core resolves
the latest valid completed presentation and its exact, digest-matched
ancestry. Reports share that selection across figures, numbers and native snapshots
for each page request in development and for each production build. Browser
animation JSON needs
public GET/HEAD CORS. Deployments use the normal GitHub Actions Pages workflow;
new run backups appear in static report text only after another site build.

Limit pruning to one experiment with `prune --experiment exp141 --dry-run`.
Runs outside that scope and their full input ancestry are retained. Current
runsets are retained even without a report. Legacy report usage
is additionally checked across all experiments, preserving historical pins until
the unified maintenance cutover.
Reports declare source experiments through `inputs`, without pinning run IDs.
Every existing run’s payload remains validated. Retained runs
must also have complete, digest-matched input ancestry; an unused run being
pruned may reference a missing or changed historical input.

Rclone credentials belong in its local configuration, never in the repository. Check available remotes with `rclone listremotes`.

## Working safely

- Use the shared Pingstore helpers to create and consume runs; let them enforce layout, checksums, allocation and locking.
- Completed scientific outputs are immutable. Produce a new run rather than modifying `export/`. Record explicit metadata corrections in the run’s README history.
- Select stage inputs explicitly by run ID and payload digest; never silently substitute the latest run.
- Hidden temporary runs are unfinished work, not usable evidence. Historical v2/v3 runs require separately authorized recovery or migration.
- Use `evict` for verified backup copies and `prune` for unused history rather than manually deleting completed runs. The current legacy
  prune commands still have separate local and R2 scopes; unified pruning is pending. Protect results that must remain local.
- Restore backups into a separate recovery directory and validate selected runs and their full input ancestry before use.

## Current-runset cutover (#178)

`status` and the site consume the same core resolver. Inspect a human-readable
inventory or save its versioned, rebuildable JSON contract:

```sh
uv run pingstore status
uv run pingstore status --source local --json
uv run pingstore status --source local --verify
uv run pingstore status --source remote --json --index-path /tmp/run-index.json
```

Automatic mode prefers the latest local presentation, otherwise verified R2
metadata. Completion time orders presentations; run ID breaks ties. Incomplete
stages never replace current evidence. Invalid local evidence is reported and
never silently replaced with a remote copy. Inputs used by scientific stages
remain explicit IDs and payload digests.

The `pingstore.index/v1` index records selections, exact ancestry, report consumers,
protection and unfinished-work reasons, storage, verification and unresolved
holdings. Local validation caches live outside runs, keyed by the runs path and
exact file metadata. Rendering reads metadata without hashing scientific arrays.
Use `status --verify` to authenticate local payload bytes and cache the result.
Changes since indexing are unresolved until explicit verification; a checksum
failure stays unresolved on later requests. Caches can be deleted and rebuilt;
a rebuilt metadata-only index makes no claim of local byte verification. R2 verification
means a checksummed catalog entry, independently of local availability or execution
origin. An unavailable remote inventory remains unknown, never an empty inventory.

Historical holdings are **Unresolved**, rather than unused, until active and
incoming publication evidence is established. Status is explanatory metadata,
not deletion authorization. Legacy pins are exposed for migration; new report
selection is automatic. Both legacy prune paths additionally retain current
runsets. Existing hash-bound maintenance plans remain required.

Each static build emits `/pingstore-evidence.json` (`pingstore.evidence/v1`) with
its build/revision identity, index generation, exact selected runs and input
pins. The normal Pages artifact deploys it with the HTML. A build manifest is not
proof of a successful deployment. Active/incoming deployment reconciliation,
unified `prune --storage`, resumable cross-store cleanup, export audits and live
migration pilots remain pending under #178. No existing payload is rewritten
or deleted by this reporting cutover.
