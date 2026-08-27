# Storage Guide

Version: **3.0.0**

The Storage Guide defines Pinglab's Pingstore filesystem convention for storing,
validating and consuming scientific runs. This file is the canonical guide.

## 1. Versioning

Version this guide independently of Pinglab, Demolab and the other guides.
Increment the major version when changed requirements make previously compliant
storage implementations or workflows require revision, the minor version for
compatible additions, and the patch version for corrections or clarifications
that do not change requirements. Update the version above and add a short entry
to the version history when changing the guide.

The guide version is separate from run schema versions such as `pingstore.run/v3`.
Changing this guide does not itself migrate existing runs or authorize a migration.

### 1.1. Version history

- **3.0.0** — Require source-neutral IDs for new staged runs and reservations.
  Keep origin in the v3 manifest; retain read support for existing suffixed v3
  runs until explicitly migrated. Schema fields and v2 rejection are unchanged.

- **2.0.0** — Require v3 for all operational storage paths; remove allowances
  for v2 reads, legacy capture, discovery, publication and reservation completion.
  Retaining historical evidence does not make it conformant or authorize migration.
- **1.1.0** — Add source-preserving, scoped conformance guidance and evidence-led
  suggestions for encoding reusable rules with explicit approval. Storage
  contracts and run schemas remain unchanged.
- **1.0.0** — Name and version the existing Storage Guide; storage requirements
  and run schemas remain unchanged.

### 1.2. Applying the guide to existing work

Read the current implementation and relevant tests immediately before editing.
Use the live files, including uncommitted edits, as the baseline; do not rebuild
them from an older revision or remembered design. Preserve settled decisions
whether made directly by the user or through an agent.

- Treat conformance as a minimal, scoped change, not permission to redesign
  storage. Edit only the requested target and strictly necessary dependencies;
  preserve unrelated behaviour, compatibility and manual edits.
- Apply the requested guide version, not silently the latest one. If that
  version cannot be recovered, ask which available version to use. Explicit
  user instructions control scope and any exceptions to the guide.
- A code or documentation change does not authorize migration, rewriting runs
  or reservations, pruning, selection, materialization or publication. Preserve
  the separate authorization boundaries below.
- If conformance conflicts with retained evidence, compatibility or a settled
  decision, identify the conflict and ask before making a material change.
  Preservation does not establish correctness or waive validation requirements.
- Review the diff against the live starting files and run relevant checks.
  Report unresolved gaps and checks not performed; do not infer successful
  validation of stored runs from code changes or passing unit tests alone.

### 1.3. Improving the guide through use

While applying this guide, use concrete difficulties, validation failures and
the user's corrections to identify improvements to its reusable instructions.

- Suggest encoding a rule when the lesson would materially improve future
  storage work; do not manufacture a suggestion for every task.
- Distinguish reusable rules from run-specific repairs or migration decisions.
  Check existing rules and refine them rather than adding duplicates. Put a
  rule in the guide that owns it; execution and writing have separate guides.
- For each suggestion, state the observed problem, propose exact wording and
  its location, explain the expected benefit, and ask whether to encode it.
- Complete the requested work first unless a conflict requires clarification.
  Present suggestions in the task response, not inside retained run records.
- Do not modify the guide or broaden implementation edits without explicit
  approval. Approved guide changes follow its versioning rules and do not
  themselves authorize changes to stored data.

## 2. Pingstore filesystem convention

Pingstore is a filesystem convention for immutable scientific runs, not a
service, database, catalogue, or general command-line interface.

```text
.pingstore/runs/<run-id>/
├── run.json         # Required: authoritative stage, inputs and provenance
├── export/          # Required: this stage's outputs
├── provenance/      # Optional: scripts, patches and retained evidence records
└── README.md        # Optional: human notes
```

This is `pingstore.run/v3`, required for all local/HPC executions. No other
root entries or symlinks are accepted. Compute/analyse exports may contain
subdirectories; present exports contain only flat regular files. No empty
presentation directory is created. New run IDs contain the experiment, counter and stage only;
execution source belongs in `origin` and `execution`, not in the ID. V3 requires
an explicit stage and input references.

`run.json` records execution, source provenance,
and `payload_digest`: SHA-256 of UTF-8 compact, sorted-key JSON of the sorted
file inventory (`path`, `size_bytes`, `sha256`). Paths are relative to the run,
ordered lexicographically. Include every payload file: export/, optional
provenance/ and README.md, including nested manifests; exclude only root run.json.
Directory entries and filesystem timestamps are not evidence or digest inputs.

Write to `.pingstore/runs/.<run-id>.tmp/`, finish the payload and any notes, write
run.json, validate layout and checksums, and rename to the visible run ID.
Failed runs remain hidden and are not evidence or publication input. A completed
run, including its README, is immutable. New analysis requires a new run.

For a present run, `export/` contains figures, tables and summaries such as
`numbers.json`.
Use names such as `rasters__canonical__seed42.png`, not nested folders.
Materialization validates the completed run and copies its entire export/ into
`.artifacts/<experiment>/`, without suffix filtering. Compute/analyse publication
is rejected, even if those exports contain image files. `_manifest.json` is a
compatibility projection for the publishing engine; run.json remains authoritative.
Execution scripts, patches and original declarations belong under provenance/,
not in the publication output. Numerical arrays, checkpoints and model bundles
belong under the compute/analyse export/.

A compact scientific export may declare `export_root` in run.json, for example
`export/cells` for an exp022 model bank. Readers of an explicitly selected run use
that directory; v3 records without this field default to export/. The path must
remain within export/. This does not change which run is selected or compact
other runs implicitly.

`collections.json`, when present, is a manually maintained mapping from named
views to arrays of explicit run IDs. No latest/official selection is inferred.
Existing artifact views without a locally retained backing run are not rebuilt
or silently replaced during a storage migration.

## 3. Schema enforcement and historical evidence

`pingstore.run/v3` is the only conforming operational run schema. All writers,
readers, stage inputs, discovery and materialization must require v3 and its
explicit stage and input references. There is no legacy allowance for flat
experiment runners, native capture, typed or untyped v2 runs, or incomplete v2
reservations. Existing implementations that accept or write v2 require revision
to conform to this guide; legacy descriptions elsewhere are not an exception.
Do not relabel mixed scientific/presentation output as v3 without splitting
execution into independent stages.

Validate the declared schema, exact v3 layout and payload checksum before use;
reject unsupported schemas rather than inferring a layout or silently falling
back to legacy paths. Shared layout helpers resolve validated v3 scientific
exports and presentation exports. The Demolab discovery field remains named
`presentation`, but points only to a present run's export/.

Historical v2 runs and reservations remain unchanged as retained evidence, not
conforming operational inputs. Preservation does not permit their normal
consumption, discovery, publication or completion. Migration or historical
inspection for migration/recovery requires separate explicit authorization and
recoverable originals; this guide change does not authorize either operation.
Do not silently reuse a v2 reservation under v3; new execution requires a new v3
reservation. No general v2-to-v3 migration utility is introduced here. The three
local staged exp022 runs were previously migrated with unchanged IDs,
byte-preserved outputs and updated input pins; see
[their migration record](../../experiments/exp022/README.md).

## 4. Independent experiment stages

New staged runs use `<experiment>-rNNN-<stage>`, for example
`exp022-r001-compute`, `exp022-r002-analyse` and `exp022-r003-present`.
Do not append `local`, `slurm`, `runpod`, a cluster name or a job ID. Record the
operation's location in `origin` and `execution`; preserve original scientific
producer information separately for imported evidence. A future HPC computation
uses the same naming scheme as local execution.

The authorized migration of the current 21 completed runs is documented in the
[source-neutral migration record](SOURCE_NEUTRAL_IDS.md), including preservation
checks and recovery locations.

The v3 manifest fields and payload layout are unchanged. Readers still validate
existing counter-first, origin-suffixed v3 runs without renaming them. Their
recorded origin must match their historical suffix. This does not grant v2 runs
or stage-first IDs operational eligibility. Writers and reservation consumers
require the new source-neutral shape; never resume or rewrite an older suffixed
reservation automatically. Existing runs and reservations change only through
an explicitly authorized, reversible migration.

The allocator counts visible and hidden identities in both old and new formats.
Exclusive temporary-directory creation reserves each complete identity before
work or dispatch, independently of execution origin. Wilkes campaigns retain
`--execution-origin slurm-wilkes` as manifest metadata; `campaign` permits mixed
workers without claiming one training host. Individual job IDs remain in cell
execution records. RunPod similarly reserves its identity before dispatch.

Every `run.json` requires `stage` and `inputs`. `inputs` maps explicit roles to
`{run_id, payload_digest, run_json_sha256}`; an initial compute run has
an empty mapping. Both payload and authoritative manifest are pinned. Readers
validate exact inputs before use and completion, and never silently choose a
replacement. Large scientific inputs are referenced, not recopied downstream.
Keep all referenced source runs when transferring a derived result: a presentation
run alone is not a backup of the scientific evidence.

`stages.py` owns reservation, source resolution and atomic completion as a Python
library, not a management CLI. It records actual commands, code/lock provenance,
uncommitted code patches and execution time. Each stage completes independently;
it never launches a different stage or calls materialization. Failed work remains
hidden. An unused scheduler reservation is not a completed run or a cache source.

An explicitly authorized historical migration/import is a new operation with a
new timestamp and its actual origin, not a re-execution of historical jobs.
Preserve original scientific bytes and retain the original run.json, notes and
source checksums as migration evidence under provenance/. Historical SLURM
attempts and inherited/repaired lineage remain distinct from the import
operation. Operational input references must resolve to validated v3 runs; a
reference to an unmigrated v2 run does not satisfy the input contract. Preserve
the original separately; do not delete, rename, reselect or rewrite it without
separate authorization.

See the [Experiment Runner Guide](../../experiments/README.md) for commands and
responsibility boundaries. Moving a payload file changes its checksum inventory
and dependent pins.

## 5. Demolab discovery

Discovery includes v3 present runs with at least one nonempty export file other
than the compatibility bookkeeping names in `layout.RECORD_NAMES`. Eligibility
comes from the authoritative stage, never a name substring or extension. Empty or bookkeeping-only
outputs and compute/analyse runs are omitted **after** full validation; malformed
compute runs still fail discovery. Numbers, tables, figures and videos all qualify.

`pingstore discover` is the narrow read-only CLI exception: it projects validated
completed runs into Demolab's generic discovery protocol. It does not choose a
run, write an index, copy presentation files, or manage local or remote storage.

```sh
uv run pingstore discover
uv run pingstore discover --source .pingstore/runs
# Equivalent module entry point:
uv run python -m pingstore discover --source .pingstore/runs
```

The source is the directory containing run folders, not the `.pingstore` parent.
Resolution is explicit `--source`, then `DEMOLAB_PREVIEW_SOURCE`, then
`.pingstore/runs` relative to the working directory. The source must exist and
must not use symlinks. Discovery inspects only immediate visible directories;
hidden entries, regular files and symlink candidates are ignored. Every visible
candidate must declare v3 and pass its full layout and payload checksum checks.
A malformed or unsupported-schema candidate, including a v2 run, fails the
command with an error on stderr and no JSON on stdout;
an empty store returns `[]`. Runs and their payloads are never modified.

Output is one JSON array, sorted by run directory name. Each record contains
`id` and `label` from `run.json.run_id`, `experiment` from `run.json.experiment`,
`created_at` from `run.json.created_at` normalized to UTC, and `presentation` as
`<run-id>/export`, relative to the source.
Timestamps must include a timezone; filesystem dates and presentation-side
metadata are not substitutes.

With a Demolab version supporting command-based preview discovery, add:

```yaml
preview:
  source: .pingstore/runs
  discover: [uv, run, pingstore, discover]
  articles:
    exp092: [exp023, exp025, exp038, exp048]
    exp093:
      baseline: [exp025, exp038, exp049]
      candidate: [exp025, exp038, exp049]
```

Demolab supplies `DEMOLAB_PREVIEW_SOURCE`. Omitted articles automatically match
their IDs to `experiment`; lists declare multiple inputs and named groups allow
independent comparison selections. Article-scoped `data-file()` bindings are
still required in writings: discovery alone does not redirect hardcoded paths.
The current Demolab implementation defaults preview selections to Latest by
metadata timestamp and provides Published/default explicitly. Discovery itself
does not pick a default or change published inputs. This example does not enable
preview in Pinglab or migrate its writings.

Demolab has no separate checksum-validation callback, so this command verifies
**all candidate payloads on every invocation**, including `export/`. No metadata
cache or unsafe fast mode bypasses validation. Large stores can exceed Demolab's
current 30-second discovery timeout; efficient validation requires a separate
protocol change, not dropping checks here. Ordinary builds do not call discovery.

## 6. Migration and recovery boundary

The historical `migrate_v2.py` utility produces v2, so it is not a conforming
upgrade path under this guide. Its presence does not authorize v2 writes or
activation of a v2 store. Historical recovery under separately authorized legacy
procedures does not make the recovered store conformant with this version.

Any separately authorized migration must verify original payloads, retain
recoverable originals and provenance, record file mappings and checksums, and
validate the resulting v3 runs and their input pins before operational use.
Unknown classifications, collisions, missing evidence or ambiguous recovery
states must stop the operation. Activation requires stopped writers/readers and
rechecked source and prepared inventories. Preserve rollback copies; do not
merge old and new run trees or delete backups automatically.

Remote R2/HPC stores are not migrated by a local operation. Historical backups
must retain entire runs in their recorded schema, not just publication outputs;
restoring a historical v2 backup does not make it eligible for operational use.
Never prune a source because a presentation copy exists; verify independent
backups first. This guide change authorizes no migration, activation, recovery,
deletion or rewriting of retained runs.
