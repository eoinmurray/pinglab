---
title: Pingstore Rules
author: Eoin Murray
collection: notes
description: Pingstore run layout, provenance, validation, pruning, Cloudflare R2 backup access and recovery.
created_at: '2026-09-11'
updated_at: '2026-10-14'
---

# Pingstore Rules

Version: **4.11.0**

These rules define Pingstore's filesystem convention. Pingstore is not a
service, database, catalogue, lifecycle manager, or general management CLI.

Do not reintroduce collection/experiment directory nesting, catalogues,
lifecycle states, automatic official selections, preview overrides, archive
bundles, or a general Pingstore management CLI. The narrow discovery and
hash-bound backup and prune commands below are the only operational CLI
exceptions. The historical v2 migration utility is not a conforming upgrade
path and does not authorize v2 use.

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

Reports may declare `collection` as a single slug or a nonempty list of unique
slugs. For new runs, the first listed collection supplies the single
`run.json` provenance value. Additional memberships are site navigation metadata;
changing report membership does not rewrite completed runs.

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

New compute records declare `scope` as `"full"` or `"phase"`. Phase records
also require a descriptive lowercase hyphen-separated `phase` name, such as
`"train"`, `"evaluate"` or `"diagnostics"`; full records omit `phase`. Scope
describes scientific coverage, not execution success or lifecycle state. Every
visible phase run is complete for its declared phase and satisfies the same v4
layout, checksum and immutability requirements as a full run. Failed or unfinished
work stays hidden and must never be labelled a completed phase.

Existing completed v4 compute records without `scope` retain their existing
declared scientific coverage; do not rewrite them or infer a phase from their
contents. Consumers still validate recipe-defined completeness, including any
existing partition semantics; absent scope is not proof of an unpartitioned
full result.
New phase-aware writers and validators must enforce the explicit scope contract
and reject contradictory scope, phase and ID combinations. This does not relax
the prohibition on operational v2/v3 inputs.

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

Use shared validated v4 layout helpers, not hardcoded consumer paths. Local and
HPC execution use the same run-folder contract. Allocate HPC run IDs before
submission so concurrent jobs cannot claim the same identity. Keep complete
execution provenance in `run.json`; do not infer provenance from filesystem
timestamps or the run ID alone. Treat visible run directories as immutable.

Reserve a new identity before local or scheduler execution. New IDs use
`<experiment>-rNNN-<stage>`, such as `exp022-r001-compute`; execution location
belongs in `origin` and `execution`, not the ID.
For a completed compute phase, use
`<experiment>-rNNN-compute-phase-<name>`, such as
`expXXX-r001-compute-phase-train`. A full compute result, whether executed
together or explicitly assembled, uses `<experiment>-rNNN-compute`. Phase names
are descriptive scientific names, not `stage-1` numbering. Do not use `-partial`
or `-partial-stage-X`: a phase result has completed its declared work. Phase and
full runs share the same experiment-wide allocation counter; never reuse a
counter for another phase. `stage` remains `compute`; scope and phase belong in
`run.json` and are never inferred as provenance from folder names.

Build the run under `.pingstore/runs/.<run-id>.tmp/`. Hidden working directories
may contain temporary reservation and writer-lock files. Remove those files,
finish `run.json`, README, and export, validate the exact layout and digest, then
atomically rename the directory to its visible identity. Failed work remains
hidden and is not operational evidence.

Readers must validate v4 layout and the export digest before consumption.
Compute starts with empty inputs unless its scientific definition explicitly
requires completed source runs, including dependencies between named compute
phases or phase assembly. Every stage names every input explicitly; it never
selects “latest”, launches upstream work, or silently substitutes a run. Input
references remain `{run_id, payload_digest}` pairs, validated before consumption
and again before completion. Consumers requiring a full compute result reject
phase scope unless they explicitly support that particular phase.

`export_root` is obsolete. Readers use the whole export and resolve explicit
unit IDs. Hidden writers may temporarily use deeper tool-native paths, but the
shared completion helper canonicalizes them before validation and visibility.

### Task-owned temporary-run cleanup

At task completion, remove task-owned v4 `.pingstore/runs/.<run-id>.tmp/`
directories once they are no longer needed for execution, recovery or transfer.
Confirm that local and remote workers or queued jobs have stopped, no active
writer owns the directory, and no checkpoint or pending transfer needs it.
Perform cleanup under the shared helper's exclusive Pingstore operation lock.
Preserve the experiment's allocation high-watermark: remove a reservation only
when a newer allocated identity remains; otherwise retain it until one exists.
Do not sweep other tasks' temporary runs or historical v2/v3 reservations.
Report any temporary runs deliberately retained and why.

### Assembly of completed compute phases

Assembly is an experiment operation defined by the
[Experiment Runner Guide](/notes/experiment-runner-guide/), not a Pingstore CLI
operation. There is no generic `pingstore merge` command.

1. An explicitly invoked experiment-local assembler selects phase run IDs and
   verifies their scientific coverage, compatibility and recorded dependencies.
   Shared Pingstore helpers validate v4 layouts, digests, references, allocation,
   locking and completion. Never choose phases by recency: a dependent phase
   must agree with the exact upstream identity and digest selected for assembly.
2. Reserve a new full compute identity, copy the required scientific output bytes
   into its hidden temporary export using an explicit deterministic mapping,
   reject unexpected collisions, and validate the full export contract. This is
   a narrow allowance for copying phase outputs into a full scientific result,
   not publication materialization or another output store. Do not move, edit,
   rename, hard-link or delete source payloads as part of assembly.
3. Set `stage: "compute"`, `scope: "full"` and
   `execution.operation: "assemble"`. Record every selected phase input under a
   stable role, its run ID and export digest, the actual assembly command, output
   mapping, configuration, source identity, origin and timing in `run.json`.
   README records the assembly history. Source execution details remain in the
   referenced phase records; assembly must not claim it reran their computation.
4. Hold the shared operation lock throughout input use and writing, revalidate
   inputs before completion, calculate the new payload digest and atomically
   finalise the full run. Assembly executes no missing computation and never
   extends an already completed run. Its input graph remains authoritative even
   when the full export contains copied phase outputs.
5. Phase inputs remain ancestry for retention, backup ordering and transfer.
   Assembly does not authorize their pruning. Compute phases are compute runs
   for stage filters and HPC retention; they are not directly publishable and
   discovery never lists them as present runs. Preserve all normal allocation
   high-watermark and active-writer protections.

These rules specify phase-aware storage support; they do not claim that current
helpers already implement it. Implement and validate shared allocation and
scope-aware readers before writing new phase runs. Do not silently reinterpret
or migrate historical records.

## 5. Discovery and publication

`pingstore discover` is the read-only integration command. It validates visible
completed v4 runs and emits Demolab discovery JSON for populated present runs.
It does not select, mutate, materialize, upload, prune, or persist a catalogue.

```sh
uv run pingstore discover
uv run pingstore discover --source .pingstore/runs
```

Preview and publication read selected present runs directly from their validated
`export/`. Do not create `.artifacts/` or another per-experiment copy of run
outputs. Publishing compute or analyse runs is rejected. Presentation metadata
is read from `run.json`; exports do not contain compatibility manifests.

The [local publisher](https://github.com/eoinmurray/pinglab/blob/main/tools/publishing/README.md) freezes complete selected present
runs in a disposable build workspace under `.demolab/`, validates the copies,
and emits a static site. These build inputs retain the run-folder contract;
they do not form another persistent scientific store. Collection execution keeps
run-ID and payload-digest references instead of copying presentation exports.

`collections.json`, when present, maps named views to explicit run-ID arrays.
No official or latest selection is inferred.

## 6. Explicit pruning

`pingstore prune` is the sole command that removes local completed runs. It is a
narrow garbage-collection operation, not a lifecycle manager or general storage CLI.
It retains every run whose authoritative execution provenance identifies HPC,
except superseded HPC presentations explicitly allowed by the option below,
the newest populated present run for each experiment currently exposed by
discovery, explicit collection and MDX frontmatter `runs` pins, inputs of incomplete
runs, and the complete transitive ancestry of those roots. It also retains each
experiment's highest allocated counter so a removed identity cannot be reused.

Retirement is an explicit one-shot pruning mode, not persistent repository state.
`--retire-experiment expNNN` is accepted only when that experiment has completed
runs but no runnable code or `experiments/expNNN/report/report.mdx`. It limits an otherwise unscoped operation
to the named experiment and permits its non-HPC newest presentation and identity
high-watermark to be pruned. The dry-run derives the highest allocated counter
from the validated run inventory and binds the retirement request into the plan
hash. Pins, incomplete inputs, out-of-scope descendants, transitive ancestry and
HPC provenance still take precedence.

Experiment IDs are permanent and must never be reused after removal. Once an
experiment's last run is explicitly retired and pruned, its old counter does not
need to remain as live filesystem state because future work must use a new
experiment ID. An absent experiment directory or writing is not by itself
permission to prune an experiment's protected runs.

Pruning requires two separate invocations:

```sh
uv run pingstore prune --dry-run
uv run pingstore prune --confirm <complete-sha256-plan-hash>
```

Dry-run validates every visible v4 run and the complete dependency graph, prints
every retained and proposed run with its reason and byte count, and makes no
filesystem changes. The plan hash binds the exact run records, payload digests,
sizes, incomplete-run input state, and candidate set. Confirmation recomputes
the plan under an exclusive pruning lock and aborts on any difference or active
writer. A bare confirmation is forbidden.

An explicit `--allow-superseded-hpc-presentations` option, requiring
`--stage present` only, permits deletion of older HPC presentation runs.
Latest presentations, explicit pins, incomplete inputs, full ancestry and
allocation high-watermarks remain protected. HPC compute and analyse retention
is unchanged. This exception is bound into the plan hash and must be repeated
at confirmation.

Pruning may be limited to stages with repeatable `--stage compute`,
`--stage analyse`, or `--stage present`. Runs in other stages remain retention
roots, including their full ancestry. Stage filters are bound into the plan hash
and must be repeated at confirmation. For presentation-only cleanup:

```sh
uv run pingstore prune --stage present --dry-run
uv run pingstore prune --stage present --confirm <complete-sha256-plan-hash>
```

Pruning may be limited to one or more experiments with a repeatable filter. All
runs outside the filter remain untouched, and any filtered run required by an
out-of-scope descendant remains as ancestry. The same filters are required for
dry-run and confirmation because the sorted experiment scope is bound into the
plan hash. Unknown or malformed experiment filters are rejected.

```sh
uv run pingstore prune --experiment exp099 --dry-run
uv run pingstore prune --experiment exp099 --confirm <complete-sha256-plan-hash>
uv run pingstore prune --experiment exp099 --experiment exp110 --dry-run
uv run pingstore prune --retire-experiment exp114 --dry-run
uv run pingstore prune --retire-experiment exp114 --confirm <complete-sha256-plan-hash>
```

Run reservation and execution hold a shared lock at `.pingstore/.operation.lock`;
confirmation requires the corresponding exclusive lock. The lock coordinates
processes but does not record lifecycle state or select runs.

Confirmation builds and validates a hard-linked retained tree beside `runs/`,
atomically exchanges it with the original tree, validates it again, and only
then removes the old tree. Visible run contents are never edited. A separately
authorized `pingstore backup` may precede confirmation; pruning itself does not
create archives, catalogues, tombstones, lifecycle states, or publication
selections. Historical v2/v3 evidence and migration archives are outside this
operation.

## 7. Explicit external backups

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

### Back up new runs

`pingstore backup` appends completed v4 runs to an external R2 backup using
rclone. It validates every visible completed local run, its exact layout,
scientific payload checksum and complete input graph. Hidden incomplete runs
are excluded. Backup is independently invoked; pruning, experiment execution
and publication never launch it automatically.

```sh
uv run pingstore backup --dry-run
uv run pingstore backup --confirm <complete-sha256-plan-hash>
```

Replace `<complete-sha256-plan-hash>` with the full `sha256:...` value printed
by the dry-run. The plan names every new run and reports its total bytes.

The source defaults to `.pingstore/runs` in the current working directory;
`--source` accepts an explicit runs directory. The destination defaults to
`r2:pinglab/pingstore/runs`, with `PINGLAB_R2_REMOTE`, `PINGLAB_R2_BUCKET` or
`--destination` overrides. The destination must be an rclone remote ending in
`/pingstore/runs`. Authentication stays in rclone's configuration outside this
repository.

Dry-run performs local validation and remote inspection without uploading.
Its complete SHA-256 plan hash binds the destination, local run records,
payload digests and file inventories, and remote inventory. Confirmation
rebuilds the plan and aborts on drift. A shared Pingstore operation lock prevents
pruning from removing local sources during either invocation.

Uploads preserve each complete run folder and refuse conflicting remote
objects. Scientific files and README upload first and are verified by download
comparison using `rclone check --download`. The local runs and remote inventory
are revalidated before
`run.json` completion records upload last in input-ancestry order, followed by
another download comparison. An upload without `run.json` is incomplete and
must never be consumed; a later backup can resume an interrupted upload if
existing objects match. Existing completed backups are checked against local
file inventories and exact manifest and README bytes; their scientific files
are not downloaded again on routine incremental backups. Recovery must
independently validate the recorded payload checksums.

Backup never deletes remote files or edits visible local runs. Local pruning
does not remove earlier backups. A corrected local manifest or README conflicts
with its previous backup and requires a separately authorized update operation.
The first backup copies the local store; subsequent backups add completed run IDs.

### Access and destination

The existing local rclone configuration provides remote `r2` and bucket `pinglab`.
Check configuration availability without printing credentials:

```sh
rclone listremotes
rclone lsf r2:pinglab/pingstore/runs --dirs-only
```

Authentication is supplied by rclone's local configuration; credentials are
never stored in this repository. An explicit `--source` must name a validated
local runs directory.

For a new machine, follow Cloudflare's [rclone setup guide](https://developers.cloudflare.com/r2/examples/rclone/)
with a bucket-scoped credential. Do not infer access from another machine's
configuration or an SSH alias.

### Recovery and historical archives

Restore explicitly selected completed runs and their full input ancestry into a
separate recovery directory. Validate every v4 root layout, scientific payload
checksum and input reference under these rules before activation or consumption.
A remote `run.json` marks completion by this backup workflow; it does not replace validation on recovery. Partial uploads
must not become operational runs.

Older campaign archives remain separate historical evidence. Their payload-only
layout is not a complete v4 run backup. This workflow neither migrates nor changes
those archives, and does not restore, publish, or execute experiments.

The [implementation](https://github.com/eoinmurray/pinglab/blob/main/tools/pingstore/backup.py) uses [rclone copy](https://rclone.org/commands/rclone_copy/)
with immutable checks and [download verification](https://rclone.org/commands/rclone_check/).

## 8. Historical schemas and migration

V2 and v3 runs are historical evidence, not operational inputs. They may be
inspected by explicit migration/recovery tooling but not consumed, discovered,
materialized, completed, or silently relabelled.

Require v4 for writers, readers, stage inputs, discovery and publication.
Reject v2/v3, including legacy capture and completion of old reservations; there
is no compatibility exception for flat runners or historical presentations.
Preserve historical runs and reservations unchanged as non-operational evidence.
Historical inspection for migration/recovery and migration itself require
separate explicit authorization; never silently rewrite or reactivate them.

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

## 9. Version history

**4.11.0** — Define completed compute phase scopes and names, explicit phase
dependencies, and immutable full-run assembly by experiment code using shared
storage helpers; retain ordinary lineage, backup and pruning protections.

- **4.10.4** — Read article metadata, pins, and retirement protection from
  `experiments/expNNN/report/report.mdx`.

- **4.10.3** — Accept multiple report collections, using the first for new run provenance.

- **4.10.2** — Read article metadata and retirement protection from
  `md/experiments/expNNN/report.mdx` after separating articles from code.

- **4.10.1** — Read experiment collection metadata and explicit presentation
  pins from Astro report frontmatter; recognize `report.mdx` in retirement
  checks instead of depending on the historical writings directory.

- **4.10.0** — Integrate the append-only R2 backup into the narrow Pingstore CLI;
  retain explicit hash-bound confirmation, immutable uploads and download verification.

- **4.9.0** — Add hash-bound stage filters and an explicit superseded HPC presentation exception; preserve other stages and full ancestry.

- **4.8.0** — Replace the persistent experiment-history registry with explicit,
  hash-bound one-shot retirement; make experiment IDs permanently single-use.
- **4.7.0** — Permit explicitly retired experiments to prune their local latest
  presentation and run high-watermark after recording that counter in experiment
  history; all stronger retention roots remain in force.
- **4.6.0** — Retire artifact materialization; preview and publication consume
  validated present exports directly, with complete run copies confined to
  disposable publication workspaces.
- **4.5.0** — Add hash-bound, repeatable experiment filters to pruning while
  retaining all out-of-scope runs and cross-scope ancestry.
- **4.4.0** — Add hash-bound pruning that retains HPC work, each experiment's
  newest visible presentation, explicit pins, incomplete inputs, ancestry and
  allocation high-watermarks.
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
