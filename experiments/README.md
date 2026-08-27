# Experiment lifecycle

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

## Commands and boundaries

```sh
uv run python experiments/exp022/compute.py
uv run python experiments/exp022/analyse.py --source <compute-run-id>
uv run python experiments/exp022/present.py --source <analysis-run-id>
```

Each command creates one completed run and prints its ID. Inputs are explicit
completed Pingstore runs: no latest/active fallback, automatic upstream execution,
or mutation of source evidence. The stage commands do not materialize or publish.
Large inputs remain in their source runs; downstream runs retain references and
their own outputs. A one-time historical import may copy scientific payloads to
preserve the original and give the new compute run a self-contained export.

Use `--run-id` only for an identity reserved before dispatch. Distributed compute
may own mutable campaign/checkpoint working directories until completion; those
are not completed scientific evidence. Scheduler retries and checkpoint recovery
remain inside compute. A failed stage leaves its hidden run for inspection, never
modifies an earlier completed run, and is not silently resumed by a downstream
stage. Rerun with a fresh identity unless a compute-specific recovery procedure
explicitly handles the incomplete work.

## Storage, preview and publication

The [Pingstore contract](../tools/pingstore/README.md) owns layout, stage IDs,
input provenance, validation and immutability. Stages are execution labels, not
mutable lifecycle states. The shared implementation is `tools/pingstore/stages.py`.

New staged runs write `pingstore.run/v3`: required `run.json` and `export/`,
optional `README.md` and `provenance/`. All three stages put outputs in `export/`;
compute/analyse may nest files, while present exports are flat publication inputs.
Execution attachments belong in `provenance/`, outside the copyable output.
Run provenance is always authoritative in `run.json`. Use `StageRun.export` and
`StageRun.provenance` when writing; use validated `SourceRun.export` (the declared
scientific root), `SourceRun.outputs` (the whole export), and
`SourceRun.presentation` (version-aware publication input) when reading.

`pingstore discover` validates all completed runs and lists populated present
exports. Legacy untyped v2 presentations remain discoverable; typed compute/analyse
runs are excluded regardless of file contents. Preview selects a listed run and
renders the current Typst source; it does not change Pingstore, `.artifacts/` or
the published site. Writings must
use article-scoped `data-file()` bindings; see [writing guidance](../writings/README.md).
Publication is separately authorized: materialize the explicitly selected run's
complete `export/` into `.artifacts/<experiment>/`, then build/publish. The shared
reader resolves legacy v2 publication inputs to `presentation/` instead.

## Progressive example

Compute produces `exp022-r001-compute-local`; analyse reads it and produces
`exp022-r002-analyse-local`; present reads that and produces
`exp022-r003-present-local`. Preview selects the third run.

- Change colours: present the same analysis again, producing a new run.
- Change an estimator: analyse the same compute evidence, then present explicitly.
- Change simulation conditions: compute again, then explicitly analyse and present.
- Change prose: edit Typst and refresh preview with the same presentation run.

## Migration boundary

Exp022 writes v3 and reads completed v2/v3 inputs. Its three local staged runs
were explicitly migrated to v3 with recoverable originals; the historical
Gold-2 source remains unchanged in v2. See [the migration record](exp022/README.md).
Other flat `expXXX.py` runners retain v2 capture until separately migrated; this
guide does not silently change their CLI or storage.
Old hidden v2 stage reservations must be finished with their original code or
replaced by freshly reserved executions, never silently rewritten as v3.
Exp022 preserves its historical import interface for downstream experiments, but
its retired combined execution modes fail with directions to the stage commands.
Existing completed runs retain their IDs. Importing the repaired Gold-2 bank
preserves its 102 cells and both checkpoint roles; it never claims to have
retrained them or replaced the separate base bank. Historical rasters lacking
retained raw snapshots may be explicitly carried into a presentation run, with
their old source recorded and no claim of regeneration.
