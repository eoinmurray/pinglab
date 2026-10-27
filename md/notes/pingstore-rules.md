---
title: Pingstore Guide
author: Eoin Murray
collection: notes
description: Current scientific runsets, validated storage, publication and inspected cleanup.
created_at: "2026-09-11"
updated_at: "2026-10-27T16:26:41Z"
---

# Pingstore Guide

Pingstore holds immutable scientific evidence. Code lives in Git; runs live in
local `.pingstore/runs/` and verified R2 backups at `r2:pinglab/pingstore/runs`.
A completed experiment has one current presentation and its exact, digest-matched
input ancestry. This runset may contain several compute or analysis runs and
inputs from other experiments. It is not a three-run deletion rule.

## Scientific evidence

A completed run contains `run.json`, `README.md` and `export/`. Its ID, scientific
payload, payload checksum and input pins are immutable. Generate a new runset
for changed results or deliberately smaller exports; never rewrite completed
exports, invent missing provenance or replace raw data with surviving figures.
Record explicit metadata corrections in the run's README history.

Scientific stages consume explicit run IDs and payload digests. Automatic
current selection belongs to reporting, not computation:

```python
from pingstore.library import source_run, stage_run

source = source_run(repo / ".pingstore", compute_id, stage="compute")
with stage_run(repo, "exp141", "analyse", inputs={"compute": source}) as run:
    # Retained scientific outputs go beneath run.export.
    # Temporary datasets, checkpoints and diagnostics go beneath run.scratch.
    pass
```

Keep the native bundle needed to reproduce each model, final-training and
best-validation native parameter snapshots for training, and a simulation-start
snapshot for simulation. Retain additional snapshots, measurements, recordings,
optimizer/RNG state or continuation checkpoints only when downstream analysis,
continuation or reproduction requires them. For example, an analysis of weight
changes needs the initial weights as well as final weights. Reports read native
snapshots and the recorded configuration; do not duplicate native parameter
metadata into separate report exports. Make the selected snapshot unambiguous.

## Current reporting and status

Core owns selection, runset membership and retention policy. A valid completed
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
| Current | A member of the current runset in this source context. |
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

## Commands and backup

The command surface is `status`, `discover`, `backup`, `prune` and `evict`.
Use the installed `pingstore` entry point or `python -m pingstore.cli` with the
same arguments. `discover` preserves validated Demolab discovery output.
The package has exactly three implementation files: core, library and CLI;
tests live under `tools/pingstore/tests/`. Experiments import only the library.

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

Backup availability does not prove publication. The normal Pages workflow first
freezes a remote-only catalog/index and registers the exact incoming evidence in
`pingstore/control.json`. It then builds and deploys `/pingstore-evidence.json`
with the site. This manifest contains the build/revision identity, selections,
input digests and unavailable-report declarations.

The previous active deployment and all incoming manifests remain retention roots.
Old roots are released only when GitHub reports the replacement as the active
successful deployment and the deployed manifest bytes match. A failed build or
deployment cannot release the previous active evidence. Pruning independently
rechecks the deployment proof. Missing or unreconciled publication state retains
potentially referenced data and names the unresolved state.

Each deployment attempt registers its frozen evidence again, including a retry
of only the deploy job. Activation runs in a following job, after the Pages
environment job succeeds. An uncertain deployment keeps its incoming evidence;
a verified replacement releases completed incoming attempts and preserves the
roots of other pending workflows.

Legacy publication bootstrap uses the authenticated artifact of the active
successful Pages run and validates its declared run ancestry against the store.
The explicitly self-contained `recovered/exp124/published.html` is a text archive:
its historical Dataset labels are not current Pingstore roots. Its deployed
self-hosted assets remain part of the Pages artifact. Current exp124 evidence is
selected from its actual completed v4 runset.

Pages requires the private `PINGLAB_RCLONE_CONFIG` repository Actions secret.
There is no manual `gh-pages` branch or separate publishing framework. A build
manifest alone is not proof of deployment success.

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
