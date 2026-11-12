---
title: Pingstore Operations
author: Eoin Murray
collection: notes
description: Status, backup, cleanup and recovery procedures.
created_at: "2026-10-29T17:42:34Z"
updated_at: "2026-11-12T12:03:36Z"
---

# Pingstore operations

Read only the section for the requested operation. Standing evidence and cleanup
constraints live in [Pingstore Rules](/notes/rules/pingstore). Examples do not
authorize backup, publication, eviction or pruning.

## Experiment API

```python
from pingstore.library import source_run, stage_run

source = source_run(repo / ".pingstore", compute_id, stage="compute")
with stage_run(repo, "exp141", "analyse", inputs={"compute": source}) as run:
    # Retained scientific outputs go beneath run.export.
    # Temporary datasets, checkpoints and diagnostics go beneath run.scratch.
    pass
```

## Current reporting and status

Core owns selection, runset membership and retention policy. Explicit
`runset_inputs` define dependency links inside one experiment, advancing from
compute through analysis to presentation. Original `inputs` preserve scientific
source pins as provenance; they do not link corrected runsets for retention.
Reanalysis without a direct compute pin retains its original compute pin directly.
A valid completed
presentation becomes current automatically, ordered by completion time and then
run ID. Development prefers the latest local presentation, otherwise verified
R2 evidence. Explicit local-only and remote-only modes remain available. Broken
local evidence is reported, never silently replaced with R2.

One development request shares its selection and catalog snapshot across MDX
values, figures, native parameters, metrics and Dataset rows. A reload refreshes
all of them. Production freezes the same inputs for each build. A report with no
completed runset shows an explicit unavailable page; it contributes no fabricated
results to the evidence manifest. Invalid selected evidence still fails rendering.

```sh
uv run pingstore status
uv run pingstore status --json
uv run pingstore status --inventory --json
uv run pingstore status --source local --verify
uv run pingstore status --source remote --index-path /tmp/run-index.json
```

`status` and the site consume `pingstore.index/v1`. This small derived index is
rebuildable from v4 records, catalog inventory metadata, report input declarations
and publication state. `--inventory` refreshes the authenticated R2 object listing;
normal rendering uses the published inventory generation. `--verify` authenticates
local payload bytes and caches the result outside runs. File changes invalidate
that cache. Rendering reads metadata and verifies the files it actually uses;
it never rehashes entire scientific arrays.

Dataset rows separate evidence from storage:

| Field | Meaning |
| --- | --- |
| Report use | Local reports and/or the recorded active GitHub Pages deployment use this runset. Only these rows are shaded. |
| Current | A member of a runset selected by a current local report declaration. |
| Required by … | Retained by a report, shared input, protection, deployment or unfinished operation. |
| Unused | No retention root requires the run in the resolved inventory; inspect a prune plan before removing it. |
| Unresolved | Inventory, provenance, availability or retention cannot be established. It is not unused. |
| Storage | Local, R2 or Both; Unknown/Unavailable identifies unresolved or missing copies. |
| Ran on | The producing execution's machine, independently of storage. |
| Verification | Cached local byte verification, local metadata validation or verified R2 catalog evidence. Presence alone does not prove byte integrity. |

An optional local `.protected` file beside `run.json` retains the run and its
ancestry. It is not backed up. Remote protection markers, when present, are also
retention roots. Hidden temporary runs and remote operation journals retain their
required inputs. Every retained historical run has an explicit reason; hiding
rows is not cleanup.

## Consolidated source views

An explicitly declared `consolidation.resolve_sources` collection may preserve
historical source identities after pruning their original directories. It retains
each original record, README and logical export-path map. `source_run` resolves
an absent pinned identity only through its unique owner, verifies the owner and
reconstructs the original payload digest. It rejects ambiguity, changed bytes and
mismatched pins; it never selects a newer scientific version.

Views are derived hard-link caches under `.pingstore/source-views/`, outside
immutable runs. The consolidated owner must be locally available. New consumers
keep their original scientific pins and record the physical owner in
`archived_inputs` for retention. Pruning or evicting an owner clears its cached
views. Back up and validate the self-contained owner before retiring originals.

## Commands and backup

The command surface is `status`, `discover`, `backup`, `prune` and `evict`.
Use the installed `pingstore` entry point or `python -m pingstore.cli` with the
same arguments. `discover` preserves validated Demolab discovery output.
The package has exactly three implementation files: core, library and CLI;
tests live under `libraries/pingstore/tests/`. Experiments import only the library.

Only with explicit authorization to edit old records, `detach-runsets` corrects
their dependency metadata while preserving original input pins and scientific
payloads. Inspect `uv run pingstore detach-runsets --destination
r2:pinglab/pingstore/runs --dry-run`, then apply the same scope with
`--confirm PLAN_HASH`. The journal preserves original metadata and resumes
interrupted corrections; README history records each change. Local and R2
metadata and catalog fingerprints are updated under the maintenance fence.
Previously deployed manifests remain unchanged and retain their declared evidence
until a verified replacement is published.

```sh
uv run pingstore backup --dry-run
uv run pingstore backup --confirm PLAN_HASH
```

Backup uploads immutable objects, download-verifies their bytes, then publishes
completion records and the checksummed catalog/index. Remote-only ancestors are
supported. Existing uncataloged complete backups are fully verified before they
can enter the catalog. Plans identify excluded pruned or broken unused history;
a new required run cannot silently resurrect a pruned input.

Rclone credentials belong in its private local configuration, never in Git or
chat. Maintenance uses rclone for transfer/listing and the S3 SDK's atomic
`PutObject` conditions for mutable coordination and catalog records. All writers
must use the current coordinated tools before cleanup. Do not bypass the fence
with older commands or ad hoc catalog writes.

## One inspected cleanup across both stores

Pruning defaults to **both local and R2**. Select experiments explicitly, or use
`--all`. Optional repeatable `--stage` filters retain other stages and their
inputs. Runs outside the selected experiment/stage scope, shared consumers and
inputs needed by copies outside the selected storage scope remain protected.

To retire an entire experiment, remove its source and report declarations first,
then prune that experiment with an explicit `--experiment` scope. If no report
still declares it as an input, its last presentation no longer creates a current
retention root. Shared consumers, protections, unfinished work and publication
evidence still retain their required runs. Restoring the source invalidates an
uncompleted retirement plan. An unscoped `--all` cleanup does not retire experiments.

```sh
uv run pingstore prune --experiment exp152 --storage both --dry-run
uv run pingstore prune --experiment exp152 --storage both --confirm PLAN_HASH
uv run pingstore prune --all --storage both --dry-run
uv run pingstore prune --experiment exp152 --storage local --dry-run
uv run pingstore prune --experiment exp152 --storage r2 --dry-run
```

Planning and confirmation are separate invocations. Inspect the exact run IDs,
object paths, separate local/R2 byte counts, and each retaining reason. Confirm
with the complete unchanged SHA-256 plan hash and identical scope. The planner
revalidates selections, protections, unfinished work, allocation counters,
inventories and publication before mutation. Stage completion never deletes
history automatically.

Local pruning holds the exclusive operation lock and validates the full archive
before mutation and after the deletion batch. Between deletions it rechecks
retention controls and verifies only the candidate run's bytes in quarantine;
it does not rehash unrelated runs for every candidate. Newly quarantined runs
must match the complete planned inventory. An interrupted quarantine may contain
an unchanged subset so the original journal can resume a partial deletion.

Retained scientific graphs require complete digest-matched ancestry. Broken
historical ancestry confined to deletion candidates does not block cleanup.
Unrelated missing legacy report pins do not select evidence. Where independence
cannot be established, the plan names the uncertainty and retains affected data.
Malformed holdings remain unresolved unless explicitly named with repeatable
`--malformed-run RUN_ID`; those candidates still require bounded v4 identities,
exact object inventories, scope checks and a reviewed plan. A validation error
never authorizes deleting a directory or object prefix.

A shared, non-expiring remote operation fence prevents backups, publication
registration and cleanup from racing. The surviving catalog is conditionally
published before any planned R2 object is deleted. Local quarantine and per-target
progress are journaled outside immutable runs in `.pingstore/operations/`.
Allocation watermarks and remote deletion tombstones survive removal of the
highest-numbered run. Older backup plans cannot restore pruned history.

If cleanup is interrupted, keep its journal and resume **the same confirmation
hash with the same scope**. The command checks retained inventories and remaining
planned objects before continuing. It reports partial completion and the journal
path rather than claiming full cleanup. Do not delete journals, expire a fence,
or start an unrelated cleanup to work around an unfinished operation.

## Publication retention

Site builds, deployment, evidence manifests and publication retention are covered
by [Publishing Rules](/notes/rules/publishing). Read that document when publishing
or changing deployment behavior. Storage cleanup still preserves the active and
incoming evidence roots described there.

## Eviction and recovery

`evict` removes verified local copies, including current runs, while preserving
R2 scientific history. It is distinct from pruning unused history:

```sh
uv run pingstore evict --all --dry-run
uv run pingstore evict --all --confirm PLAN_HASH
uv run pingstore evict --run exp141-r001-compute --dry-run
```

Eviction requires matching authenticated/public catalogs, matching local file
checksums, complete remote ancestry and a successful remote-only site build before
relying on remote copies. Protected runs and unfinished-work inputs stay local.
Restore explicit inputs and their complete ancestry into a separate recovery
directory and validate them before scientific execution or import. Do not manually
remove completed run directories or allocation state.

Keep migration baseline inventories, digests, deployment proofs, comparisons and
operation journals outside immutable run payloads. Metadata cutover is reversible
before pruning; deleted evidence is not covered by metadata rollback. A cleanup
task ends only after validating retained graphs and checking the selected storage
inventories, with an accurate account of retained or partially removed data.
Historical v2/v3 archives require separate recovery authorization. PDFs stay local;
production uses publisher or authorised repository links from `papers/index.yaml`.
