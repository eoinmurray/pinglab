---
title: Experiment Rules
author: Eoin Murray
collection: notes
description: Independent compute, analyse and present stages, scientific inputs and execution contracts.
created_at: "2026-10-14"
updated_at: "2026-10-16"
---

# Experiment Runner Guide

Version: **4.13.0**

This guide defines Pinglab's independent compute, analyse, and present commands.
The [Pingstore Rules](/notes/pingstore-rules) owns run layout and validation.

## 1. Experiment shape

```text
experiments/expXXX/
    recipe.py
    compute.py
    compute_parts/       # optional descriptive implementation modules
    assemble.py          # optional explicit assembly of completed compute phases
    analyse.py
    present.py
    report/              # article MDX and supporting article files
        report.mdx
    js/                  # optional experiment-specific browser JavaScript
```

Authored articles live in `experiments/expXXX/report/`, with
`report.mdx` and optional `parameters.mdx`, `appendix.mdx`, and
`references.mdx`. The [Writing Guide](/notes/writing-rules) governs them.
Keep experiment-specific browser JavaScript in `experiments/expXXX/js/`;
article components can import it from `../js/`. Shared site infrastructure
remains under `site/src/`. These are authored sources, not run exports.

`recipe.py` owns committed scientific definitions without executing on import.
Compute produces expensive or primary scientific outputs. Analyse measures
explicit compute evidence. Present creates flat figures, tables, videos, and
report-ready numbers. A change of estimator belongs in analyse; a change of
appearance belongs in present; inference that generates new activity belongs in
compute.

`compute.py` remains the public compute entry point. Long implementations may be
split into ordinary functions or descriptively named modules under
`compute_parts/`. Do not use numbered entry points such as `compute-1.py` merely
to organize source code.

### Standalone experiments

Each experiment must be runnable from its committed code and explicitly named
data inputs. Keep experiment-specific definitions and execution logic in its own
directory. Shared maintained helpers and supported tool interfaces are allowed;
reuse another experiment's results through validated completed v4 runs, not by
importing its implementation.

Experiment commands must never import or execute code from `.scratch/`, read
pre-existing scratch recordings or configuration, or require scratch files to
exist. A compute command must implement the scientific computation; a wrapper
that only copies scratch outputs is not a standalone compute stage. Promote
required exploratory code into maintained code before exposing the experiment.

Preserving previously completed scratch evidence is a separately authorized
import operation, not a replacement for the experiment's runnable compute
stage. This restriction does not prohibit temporary files created within
`StageRun.scratch` during the current execution.

## 2. Commands and boundaries

```sh
uv run python experiments/exp022/compute.py
uv run python experiments/exp022/analyse.py --source <compute-run-id>
uv run python experiments/exp022/present.py --source <analyse-run-id>
```

Each command creates exactly one `pingstore.run/v4` run and prints its identity.
Every input is an explicit completed v4 run. Commands never choose latest,
automatically execute another stage, materialize, or publish.
An explicitly invoked `assemble.py` may create a new full compute run from
completed phase runs under the rules below; this is scientific result assembly,
not publication materialization.

New IDs are source-neutral: `exp022-r001-compute`,
`exp022-r002-analyse`, and `exp022-r003-present`. Local and scheduler-backed
execution use the same shape. Execution location and scheduler details belong
in `run.json`.
Completed compute phases use `<experiment>-rNNN-compute-phase-<name>`; assembled
full results retain `<experiment>-rNNN-compute`. All share the experiment's
allocation counter. Scope and phase identity are authoritative in `run.json`,
not inferred from the suffix.

Experiment IDs are permanent identities. Never reuse the number of a removed
experiment; assign any successor a fresh `expXXX` number. This keeps old run
identities unambiguous after unused runs are pruned. Allocation high-watermarks
persist separately under `.pingstore/.allocation/`; never reset them when a run
folder is removed.

Failed work remains in its hidden temporary run. Downstream stages do not consume
it. Rerun with a new identity unless an experiment-specific compute recovery
procedure explicitly resumes the incomplete working directory.

Run reservation and execution refuse to start while the Pingstore Rules'
exclusive pruning operation is active. Pruning likewise refuses an active
writer, so it cannot race stage completion.

### Completed compute phases and explicit assembly

An experiment may divide its computation into named scientific phases when
separate execution supports iteration or reuse, for example training,
evaluation and diagnostics. These are subdivisions of compute, not additional
Pingstore stages. Keep `compute.py` as the public entry point, with an explicit
argument such as `--phase train`. Define phase names, required outputs, scientific
configuration and dependencies in `recipe.py`. Source-code organization alone
does not justify phase runs.

1. Running all phases together produces one full compute run. Running a single
   phase produces one completed compute run with `scope: "phase"` and a
   descriptive `phase` name, using the Pingstore Rules' naming contract. It must
   contain the complete scientific result required for that phase. A failed or
   interrupted phase remains hidden; `phase` never means incomplete execution.
2. Every retry receives a new identity. Dependent phases explicitly name the
   completed upstream runs they consume, recording stable input roles and
   `{run_id, payload_digest}` pairs. They never select the most recent attempt,
   execute a missing phase or change an existing run. Independent phases need
   not reference one another.
3. Assembly belongs in an explicitly invoked experiment-local `assemble.py`.
   It accepts every selected phase run ID explicitly and validates v4 layout,
   payload digests, experiment identity, phase coverage, required outputs,
   scientific configuration, seeds and source provenance under the recipe's
   declared compatibility rules. It checks actual dependency references: an
   evaluation produced from training attempt B cannot be combined with training
   attempt C merely because C is newer. Missing, duplicate, incompatible or
   ambiguous phase selections must fail. Legitimate source/configuration
   differences require explicit compatibility rules, not silent acceptance.
4. Assembly performs no missing scientific computation. It copies only the
   scientific outputs required for the full result into a newly reserved hidden
   compute run. The recipe defines a deterministic output mapping that respects
   the standard export layout; reject unexpected filename collisions and never
   silently overwrite files. Record every selected phase as an input, the
   assembly command and output mapping in `run.json`, and assembly history in
   README. Validate the full result and revalidate inputs before atomically
   completing it with `scope: "full"` and `execution.operation: "assemble"`.
5. The full result has a new identity and its own payload digest. Source phase
   runs remain unchanged; ancestry, transfer, backup and pruning rules still
   apply. Assembly provenance describes the assembly execution, while the input
   graph retains the actual compute executions. Do not claim the phases were
   rerun or performed together.
6. Ordinary analyse commands requiring a full compute result must reject phase
   scope. A consumer may accept a phase only when its recipe explicitly declares
   that phase and validates its required scientific outputs. Full and phased
   execution must satisfy the same declared full-result contract after assembly.

Use shared validated Pingstore reservation, input, locking and completion
helpers. Experiment code owns scientific completeness and compatibility; shared
helpers own filesystem correctness and provenance recording. Do not add a
generic `pingstore merge` command, mutate completed runs, or maintain a separate
catalogue. This guide specifies the contract; implement shared phase-aware
allocation and validation support before producing phase runs. A guide update
does not itself implement that support or authorize historical migration.

### Exceptional chunking and resumable compute

The default is one compute invocation producing one compute run. Chunking is an
exception for work that is expensive or failure-prone, exceeds scheduler or
resource limits, or contains genuinely independent scientific partitions. Do
not chunk work merely for code organization, progress reporting, speculative
reuse, or minor runtime savings. Use the fewest chunks that materially reduce
rerun cost or make execution feasible. Chunking must justify its additional run,
lineage, validation, transfer, and analysis complexity.

Prefer one parameterized `compute.py` entry point with explicit partition
arguments, such as `--chunk-index` and `--chunk-count`, or a descriptively named
scientific partition argument. Define shared partition semantics in `recipe.py`
when they are part of the committed scientific design. Record the exact
partition rule, chunk identity, random seeds, and other scientific configuration
in `run.json`.

Each independent chunk invocation creates exactly one compute run. Its export
must be a complete, valid scientific output for its declared partition. Chunk
runs do not select, launch, modify, or depend on one another. A downstream
analyse command names every required chunk run explicitly under stable input
roles, validates each input, and performs any combination or aggregation. It
must not select the latest chunks, infer missing chunks, or merge them
automatically.

When intermediate pieces are recovery checkpoints rather than completed
scientific results for a declared phase, keep them inside one hidden incomplete
run. Named sequential phases may instead complete independently under the phase
contract above. An experiment-specific recovery procedure may explicitly resume
that same working directory only after validating its run identity, recorded
configuration, provenance, and checkpoint state. Checkpoints and recovery
bookkeeping belong in `StageRun.scratch` and are discarded before completion.
Unfinished execution must never become a visible completed run or be consumed
downstream; a completed phase is valid evidence only for its declared scope.

A completed compute run is immutable and cannot be resumed or extended. Further
independent work receives a new run identity.

### Concurrent HPC workers within one compute run

When one scientific compute result requires concurrent workers before it is
meaningful or consumable, use `experiments.helpers.hpc.concurrent_compute`. Do not
copy its locking, completion-record, provenance, or collection machinery into an
experiment.

The experiment's `recipe.py` remains authoritative for scientific work-item
definitions and partition constraints. Before submission, the experiment-local
HPC adapter freezes the complete resolved recipe, ordered work-item IDs, exact
work-item-to-shard allocation, input pins, clean source identity and scheduler
resources in a reviewable plan. Workers consume that allocation; they do not
reconstruct it from mutable environment state.

Workers share one hidden reserved compute run but write only their assigned
scientific outputs and `.scratch/shards/<index>/` bookkeeping. The shared helper
provides a nonblocking shared worker lock, exclusive collector lock, one
exclusive writer lock per shard, an immutable `pinglab.concurrent-shard/v1`
completion record, input revalidation and payload checksums. A matching completed
shard may be reused explicitly. Stale writer locks require reviewed recovery.

The collector never executes missing work. It requires and verifies the exact
complete shard set, then projects worker timing, host, device, command and Slurm
identity into `run.json`; shard bookkeeping is discarded when the run completes.
Submitters use the receipt-first helper in `experiments.helpers.hpc.slurm_submit` so
an ambiguous scheduler response cannot trigger automatic resubmission. Analyse
and present remain separate explicit stages.

## 3. Run records

Every completed run contains `run.json`, `README.md`, and `export/`, plus an
optional manual `.protected` file as defined by the Pingstore Rules. The marker
is local retention metadata outside the scientific payload; it protects the run
and its input ancestry without marking them used by a report.

- Put machine-readable execution history in `run.json` once.
- Put human-readable history in README.
- Put scientific outputs required by downstream stages in `export/`.
- Keep exports to scientific data. Run-wide files and single-file units live
  directly under `export/`; use `<unit-id>--<role-file>` for a singleton. A unit
  directory is allowed only for at least two related files, with no deeper nesting.

Do not create `provenance/`, copied command manifests, replay scripts, source
patches, or parallel provenance envelopes. The shared stage helper already
records command, environment, configuration, Git commit, dirty state, lockfile
checksum, inputs, and timing in `run.json`.

Use `StageRun.export` for scientific outputs and `StageRun.scratch` for temporary
logs, commands, duplicate execution configurations, and recovery bookkeeping.
Per-unit scientific definitions needed to interpret the exported data may remain
in the unit directory. Scratch is discarded before completion. Read scientific sources through validated
`SourceRun.export` or `SourceRun.outputs`; read provenance and execution metadata
from `SourceRun.record`.

Completed unit directories are canonicalized from tool-native working paths.
Readers use `SourceRun.unit(...)` for a unit directory and `SourceRun.file(...)`
for a file addressed through a formerly nested path. Do not introduce generic
container directories such as `data`, `jobs`, `cells`, or `misc`.

Use `recording.npz` for raw multimodal time series, `spikes.npz` for spike-only
output, and `rasters.npz` for transformed raster/event data. Do not write the
legacy aliases `snapshot.npz` or `recordings.npz`.

Present exports remain flat publication inputs and therefore do not use
`StageRun.evidence`. Presentation lineage belongs in `run.json` or the exported
`numbers.json` when required.

### Mandatory network bundles and simulation snapshots

Every new compute run must retain the complete compiled `snnlab.lang` bundle
for every distinct SNNLab network configuration it executes, and use
`snnlab.sim.SnapshotSpec` to capture every simulation, inference or training
execution. This applies to local and HPC runs, full runs, phases and chunks.
Define stable scientific unit IDs in the recipe so each result can be matched
unambiguously to its bundle and snapshots. A compute run with no network
execution records that fact in `run.json`; it must not fabricate artifacts.

Preserve the non-SNNLab backends in exp080/081 (local nonspiking membrane
equations), exp117 (SciPy mean-field continuation/ODEs), and exp125 (snnTorch).
These use the shared local `external_execution` writer for explicit backend,
scientific configuration and actual parameter snapshots before and after their
executions. Do not fabricate Lang bundles or replace these backends to satisfy
the SNNLab-specific requirement. Local snapshots use
`pinglab.external-run-snapshot/v1` and are inspection records, not checkpoints.

1. Save bundles through `Bundle.write(...)`, including `manifest.json`,
   `graph.json`, any `training.json`, and all referenced assets. A graph-only
   JSON dump or a reference to a local bundle is insufficient. Preserve the
   complete bundle within the run's scientific `export/` payload.
2. Pass an explicit `SnapshotSpec(path=..., stage="both", parameters=True)`
   through `ExecutionSpec(snapshot=..., ...)`. Retain the start and end
   `snapshot.json` and `parameters.npz` outputs in `export/`. Use distinct paths
   for repeated executions, conditions, seeds and workers; never overwrite or
   silently omit snapshots. Capture one scientific training, inference or
   simulation execution around all its minibatches or continuation chunks;
   do not duplicate full parameters at every internal timestep or minibatch.
   Low-level SNNLab execution loops must use the library's
   snapshot capture/export facilities rather than handwritten configuration or
   weight serializers; adapt shared integration if the required hook is missing.
3. Use the shared Pingstore layout mapping for tool-native bundle and snapshot
   paths. Preserve original file bytes and enough path information to restore
   a bundle's native layout in scratch before authenticated loading. Validate
   bundle authentication and required snapshot pairs before completing the run;
   do not flatten or rename authenticated assets without a reversible mapping.
4. Treat bundles and snapshots as scientific definitions and parameter evidence.
   Keep Pingstore command, host, source and timing provenance in `run.json`;
   do not add parallel experiment-local save/configuration formats. Library
   snapshot metadata remains intact even where it describes resolved settings.
5. Snapshots are inspectable records, not resumable checkpoints: they do not
   retain input data or dynamic, optimizer and RNG state. Retain required inputs
   and use supported checkpoints separately when continuation is needed.

An assembled full compute run retains the selected phases' required bundles
and snapshots through its explicit output mapping; assembly does not capture
new simulations. Existing completed runs remain unchanged. Experiment runners
use `experiments.helpers.network_artifacts` for SNNLab evidence and
`experiments.helpers.external_artifacts` for retained external backends.
Pingstore completion validates snapshot pairs and parameter checksums, preserves
library-generated bytes and records reversible paths in
`run.json.scientific_artifacts.native_paths`. `SourceRun.restore_bundle(...)`
reconstructs native paths in caller-owned scratch for authenticated loading.

### Mandatory SNNLab bundle diagrams in reports

Every SNNLab bundle must have its library-generated diagrams saved with the
run's scientific outputs and displayed in the experiment report. Use
`Bundle.write(..., visualise=True)` to retain the circuit, training and expanded
views in SVG and PNG, or the supported SNNLab diagram API for equivalent saved
views. Preserve them through the shared Pingstore layout mapping alongside the
complete bundle; diagrams do not replace the bundle or simulation snapshots.

Present must export report-ready copies of the saved diagrams from its explicit
source-run lineage. Embed these diagrams in `experiments/expXXX/report/report.mdx`
or an included report component, using the report's selected present run.
Cover every distinct network configuration used in the reported work; repeated
executions of the same configuration may share one displayed set. Identify the
configuration and diagram view in each caption. Generate diagrams from the
actual retained bundles, never from a separately reconstructed network or
unrelated current code. A download link alone does not satisfy report inclusion.

## 4. Input identity

Each input role stores `{run_id, payload_digest}`. The digest covers all export
bytes. Readers validate the source and compare that pair before use and again
before completion. They do not pin `run.json` bytes: correcting metadata or
README history must not invalidate scientific descendants.

Large inputs remain in their source runs. A downstream run stores references and
its own outputs rather than copying upstream payloads. Keep all referenced runs
when transferring a derived result.
Explicit compute assembly is the narrow exception: it copies the recipe-defined
scientific outputs needed for a self-contained full compute payload, retaining
all phase input references and their ancestry.

## 5. Preview and publication

`pingstore discover` validates completed v4 runs and lists populated present
exports. Preview selects one of those runs and renders current Typst source; it
does not mutate Pingstore or published artifacts.

Publication is separately authorized and reads selected present runs directly
from their validated `export/`. Do not copy outputs into `.artifacts/`.
Collection execution retains explicit run-ID and payload-digest references.
Use the [local publisher](https://github.com/eoinmurray/pinglab/blob/main/tools/publishing/README.md) for a frozen static
build; its complete selected-run copies are disposable build inputs under
`.demolab/`. Compute and analyse runs cannot be published directly.

## 6. Historical work

V2 and v3 runs and incomplete reservations are non-operational historical
evidence. Migration must be explicit and recoverable. It does not claim that an
experiment was rerun and does not authorize publication or remote-store changes.

Ground scientific claims in retained outputs and history, not current code alone.
Preserve scientific definitions and distinguish an import or metadata migration
from training, inference, analysis, or plotting.

## 7. Applying this guide

Read the live target, recipe, dependencies, and relevant tests before editing.
Preserve unrelated changes and scientific choices. A guide update does not
authorize execution, migration, publication, or rewriting evidence unless the
user separately requests that operation.

Review the live diff and run proportionate tests. Report what was not executed
or verified. If a reusable rule needs improvement, propose exact wording and ask
before changing the guide outside the requested scope.

## 8. Version history

**4.13.0** — Align run layout with optional manual `.protected` markers and
keep allocation high-watermarks independently of prunable run folders.

**4.12.0** — Require saved library-generated SNNLab bundle diagrams and their
inclusion in experiment reports through explicitly selected present runs.

**4.11.0** — Require complete `snnlab.lang` bundles and library-managed start/end
parameter snapshots for compute executions, retained as scientific payloads;
preserve authenticated bundle paths and distinguish snapshots from checkpoints.

**4.10.0** — Define completed named compute phases, explicit dependencies,
phase-scoped IDs and metadata, and experiment-owned assembly into a new full
compute run; distinguish phase results from unfinished recovery checkpoints.

- **4.9.3** — Colocate articles in `experiments/expXXX/report/` and browser
  JavaScript in `experiments/expXXX/js/`; stage boundaries are unchanged.

- **4.9.2** — Move authored Astro articles to `md/experiments/expXXX/`;
  experiment implementation and stage boundaries are unchanged.

- **4.9.1** — Correct the article location to the experiment's co-located
  Astro `report.mdx`; execution and stage boundaries are unchanged.

- **4.9.0** — Require standalone experiment implementations, prohibit runtime
  dependencies on scratch code or files, and restrict cross-experiment reuse to
  validated data inputs rather than implementation imports.
- **4.8.0** — Make experiment IDs permanently single-use so an explicitly
  retired experiment needs no persistent allocation tombstone.
- **4.7.0** — Standardize concurrent HPC compute around recipe-owned work items,
  frozen reviewed allocations, shared resumable shard locking and verification,
  receipt-first Slurm submission, and an explicit non-computing collector.
- **4.6.0** — Remove the artifact-copy publication step and retain collection
  stage references instead of duplicated presentation exports.
- **4.5.0** — Permit sparing, justified compute partitioning and explicit
  hidden-run checkpoint recovery while retaining unchunked compute as the
  default.
- **4.4.0** — Coordinate reservation and execution with hash-bound Pingstore
  pruning so completed or in-progress lineages cannot race deletion.
- **4.3.0** — Flatten single-file units and standardize simulation recording
  role filenames while retaining multi-file scientific bundles.
- **4.2.0** — Limit compute/analyse exports to one scientific-unit directory
  level and standardize shared unit/file resolution.
- **4.0.0** — Adopt the v4 three-entry run root, mandatory README history,
  export-only digests, compact input pins, and `export/evidence/` for supporting
  scientific outputs.
- **4.1.0** — Restrict exports to scientific data, replace retained evidence
  sidecars with discarded scratch space, and stop writing compatibility manifests.
- **3.0.0** — Adopted source-neutral staged IDs.
- **2.0.0** — Required v3 execution.
- **1.0.0** — Versioned the runner guide.
