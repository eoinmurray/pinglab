# Experiment Runner Guide

Version: **4.7.0**

This guide defines Pinglab's independent compute, analyse, and present commands.
The [Storage Guide](../tools/pingstore/README.md) owns run layout and validation.

## 1. Experiment shape

```text
experiments/expXXX/
    recipe.py
    compute.py
    compute_parts/       # optional descriptive implementation modules
    analyse.py
    present.py
writings/expXXX.typ
```

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

## 2. Commands and boundaries

```sh
uv run python experiments/exp022/compute.py
uv run python experiments/exp022/analyse.py --source <compute-run-id>
uv run python experiments/exp022/present.py --source <analyse-run-id>
```

Each command creates exactly one `pingstore.run/v4` run and prints its identity.
Every input is an explicit completed v4 run. Commands never choose latest,
automatically execute another stage, materialize, or publish.

New IDs are source-neutral: `exp022-r001-compute`,
`exp022-r002-analyse`, and `exp022-r003-present`. Local and scheduler-backed
execution use the same shape. Execution location and scheduler details belong
in `run.json`.

Failed work remains in its hidden temporary run. Downstream stages do not consume
it. Rerun with a new identity unless an experiment-specific compute recovery
procedure explicitly resumes the incomplete working directory.

Run reservation and execution refuse to start while the Storage Guide's
exclusive pruning operation is active. Pruning likewise refuses an active
writer, so it cannot race stage completion.

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

When intermediate pieces are sequential checkpoints of one calculation rather
than independent scientific outputs, keep them inside one hidden incomplete
run. An experiment-specific recovery procedure may explicitly resume that same
working directory only after validating its run identity, recorded
configuration, provenance, and checkpoint state. Checkpoints and recovery
bookkeeping belong in `StageRun.scratch` and are discarded before completion.
Partial work must never become a visible completed run or be consumed
downstream.

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

Every completed run contains exactly `run.json`, `README.md`, and `export/`.

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

## 4. Input identity

Each input role stores `{run_id, payload_digest}`. The digest covers all export
bytes. Readers validate the source and compare that pair before use and again
before completion. They do not pin `run.json` bytes: correcting metadata or
README history must not invalidate scientific descendants.

Large inputs remain in their source runs. A downstream run stores references and
its own outputs rather than copying upstream payloads. Keep all referenced runs
when transferring a derived result.

## 5. Preview and publication

`pingstore discover` validates completed v4 runs and lists populated present
exports. Preview selects one of those runs and renders current Typst source; it
does not mutate Pingstore or published artifacts.

Publication is separately authorized and reads selected present runs directly
from their validated `export/`. Do not copy outputs into `.artifacts/`.
Collection execution retains explicit run-ID and payload-digest references.
Use the [local publisher](../tools/publishing/README.md) for a frozen static
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
