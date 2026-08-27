# Experiment Runner Guide

Version: **3.0.0**

The Experiment Runner Guide defines the conventions for Pinglab's experiment
execution code and independent compute, analyse and present stages. This file
is the canonical guide.

## 1. Versioning

Version this guide independently of Pinglab, Demolab and the other guides.
Increment the major version when changed requirements make previously compliant
experiment execution code require revision, the minor version for compatible
additions, and the patch version for corrections or clarifications that do not
change requirements. Update the version above and add a short entry to the
version history when changing the guide.

### 1.1. Version history

- **3.0.0** — Use source-neutral staged IDs and reservations, following Storage
  Guide 3.0.0. Execution location remains explicit manifest metadata; stage
  boundaries and the v3 schema are unchanged.

- **2.0.0** — Require v3 for all operational execution and inputs, aligning with
  Storage Guide 2.0.0. Remove legacy capture, discovery, publication and reservation
  completion allowances; historical preservation does not authorize operational use.
- **1.0.0** — Name and version the existing Experiment Runner Guide; execution
  requirements remain unchanged.

## 2. Experiment lifecycle

An experiment has three independent execution stages and a Typst writing:

```text
experiments/expXXX/
    recipe.py       # shared scientific definitions; no execution on import
    compute.py      # training, simulation, retained recordings/checkpoints
    analyse.py      # measurements and numerical results from explicit evidence
    present.py      # figures, tables, videos and report-ready numbers
writings/expXXX.typ  # explanation and interpretation
```

Compute is usually the expensive GPU/HPC stage, analysis is usually intermediate,
and presentation is usually cheap. These are expectations, not cost guarantees.
Scientific parameters live in committed recipes, not arbitrary CLI overrides.
Changing a measurement belongs to analysis; changing its appearance belongs to
presentation. Inference to generate a raster is computation, not plotting.

## 3. Commands and boundaries

```sh
uv run python experiments/exp022/compute.py
uv run python experiments/exp022/analyse.py --source <compute-run-id>
uv run python experiments/exp022/present.py --source <analysis-run-id>
```

Each command creates one completed run and prints its ID. Inputs are explicit
completed Pingstore runs: no latest/active fallback, automatic upstream execution,
or mutation of source evidence. The stage commands do not materialize or publish.
Large inputs remain in their source runs; downstream runs retain references and
their own outputs. A separately authorized historical import may copy scientific
payloads to preserve the original and give the new compute run a self-contained
export. Operational inputs must still resolve to validated v3 runs; migration
evidence belongs in provenance, not in an unresolved or v2 operational input pin.

Run IDs contain the experiment, counter and stage, for example
`exp022-r001-compute`. Local, Slurm and RunPod execution share that format;
`origin` and execution records carry location and scheduler provenance.
Use `--run-id` only for a source-neutral identity reserved before dispatch.
Distributed compute may own mutable campaign/checkpoint working directories until
completion; those are not completed scientific evidence. Scheduler retries and checkpoint recovery
remain inside compute. A failed stage leaves its hidden run for inspection, never
modifies an earlier completed run, and is not silently resumed by a downstream
stage. Rerun with a fresh identity unless a compute-specific recovery procedure
explicitly handles the incomplete work.

## 4. Storage, preview and publication

The [Storage Guide](../tools/pingstore/README.md) owns layout, stage IDs,
input provenance, validation and immutability. Stages are execution labels, not
mutable lifecycle states. The shared implementation is `tools/pingstore/stages.py`.

All operational runs require `pingstore.run/v3`: required `run.json` and `export/`,
optional `README.md` and `provenance/`. All three stages put outputs in `export/`;
compute/analyse may nest files, while present exports are flat publication inputs.
Execution attachments belong in `provenance/`, outside the copyable output.
Run provenance is always authoritative in `run.json`. Use `StageRun.export` and
`StageRun.provenance` when writing; use validated `SourceRun.export` (the declared
scientific root), `SourceRun.outputs` (the whole export), and
`SourceRun.presentation` (validated v3 present export) when reading.
Writers and readers must reject v2; existing shared helpers that still accept
it require revision and do not provide an exception to this contract.

`pingstore discover` validates all completed runs and lists populated present
exports. Every visible candidate must pass v3 schema, layout and checksum
validation; v2 fails discovery rather than being listed or silently skipped.
Compute/analyse runs are excluded regardless of file contents. Preview selects a
listed run and renders the current Typst source; it does not change Pingstore, `.artifacts/` or
the published site. Writings must
use article-scoped `data-file()` bindings; see the [Writing Guide](../writings/README.md).
Publication is separately authorized: materialize the explicitly selected run's
complete `export/` into `.artifacts/<experiment>/`, then build/publish. V2
presentations are not valid publication inputs.

## 5. Progressive example

Compute produces `exp022-r001-compute`; analyse reads it and produces
`exp022-r002-analyse`; present reads that and produces
`exp022-r003-present`. Preview selects the third run.

- Change colours: present the same analysis again, producing a new run.
- Change an estimator: analyse the same compute evidence, then present explicitly.
- Change simulation conditions: compute again, then explicitly analyse and present.
- Change prose: edit Typst and refresh preview with the same presentation run.

## 6. Migration boundary

All experiments, including exp022 and flat `expXXX.py` runners, must write and
consume validated v3 runs. Existing v2 readers and capture paths are nonconforming;
this requirement does not itself change code, migrate evidence or authorize execution.
Old hidden v2 reservations must not be completed or silently converted. New
execution requires a fresh v3 reservation.

The three local staged exp022 runs were explicitly migrated to v3 with recoverable
originals; see [the migration record](exp022/README.md). Historical v2 runs remain
unchanged as non-operational evidence. Historical inspection for migration/recovery
and migration require separate explicit authorization and recoverable originals.
Every operational input pin must resolve to validated v3 evidence; a migrated
manifest alone does not repair missing or v2 upstream references.

Preserve existing completed-run identities unless a migration is explicitly
authorized. A historical import must preserve scientific bytes and checkpoint
roles without claiming retraining. Historical rasters lacking raw snapshots may
be explicitly carried from a validated v3 source, retaining their original lineage
and making no claim of regeneration. Importing unmigrated historical evidence is
a separately authorized operation governed by the Storage Guide.
