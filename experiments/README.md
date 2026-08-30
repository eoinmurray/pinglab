# Experiment Runner Guide

Version: **4.2.0**

This guide defines Pinglab's independent compute, analyse, and present commands.
The [Storage Guide](../tools/pingstore/README.md) owns run layout and validation.

## 1. Experiment shape

```text
experiments/expXXX/
    recipe.py
    compute.py
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
`exp022-r002-analyse`, and `exp022-r003-present`. Local, Slurm, and RunPod use the
same shape. Execution location and scheduler details belong in `run.json`.

Failed work remains in its hidden temporary run. Downstream stages do not consume
it. Rerun with a new identity unless an experiment-specific compute recovery
procedure explicitly resumes the incomplete working directory.

## 3. Run records

Every completed run contains exactly `run.json`, `README.md`, and `export/`.

- Put machine-readable execution history in `run.json` once.
- Put human-readable history in README.
- Put scientific outputs required by downstream stages in `export/`.
- Keep exports to scientific data. Run-wide files live directly under `export/`;
  repeated units use `export/<unit-id>/<role-file>` with no deeper nesting.

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

Publication is separately authorized. Materialize the complete flat export of
an explicitly selected present run into `.artifacts/<experiment>/`, then build or
publish. Compute and analyse runs cannot be published directly.

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
