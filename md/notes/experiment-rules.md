---
title: Experiment Rules
author: Eoin Murray
collection: notes
description: Experiment layout, identities, shared helpers, Pingstore and SNNLab evidence.
created_at: "2026-10-14"
updated_at: "2026-10-24T06:26:47Z"
---

# Experiment Rules

<a id="file-layout"></a>

## 1. File layout

1. Use this layout; `dataset.py` is optional, mainly for training:

   ```text
   experiments/expXXX/
       README.md          # goals, then CLI files and arguments
       recipe.py          # scientific settings and network definitions
       dataset.py         # data loading and preprocessing
       compute.py         # generation, simulation, inference and training
       analyse.py         # validation, measurements and estimators
       present.py         # figures, tables and report-ready outputs
       test.py            # all experiment-specific tests
       custom/            # extensions and necessary execution adapters
       report/
           report.mdx
           …              # supporting article files and authored assets
   ```

2. Keep CLI entry points small, scientific settings in `recipe.py`, and imports
   free of execution side effects. The canonical stage files own their stage's
   implementation, including simulation, measurements and plotting helpers.
3. Limit root-level Python files to those above. Minimise `custom/`: retain genuine
   extensions, native lifecycle adapters and necessary execution support there.
   Do not split ordinary recipe, compute, analyse or present code into `custom/`
   merely to shorten the canonical files. Put data loading and preprocessing in
   `dataset.py` where applicable.
4. Keep all experiment tests in one `test.py`, not separate `test_*.py` files.
5. Limit `README.md` to brief goals and CLI files with required and optional arguments.

<a id="structure-and-readability"></a>

## 2. Structure and readability

1. **Important:** Keep only code, tests and execution support needed to reproduce or validate
   the report and its evidence. Remove obsolete, unused and speculative paths.
2. Follow `experiments/exp141` for readable, idiomatic structure and style.
3. Keep `recipe.py` centred on `configuration(...)` and `author_network(...)`;
   the latter defines the network and training specification and returns a compiled
   SNNLab bundle. Keep loading and execution elsewhere; use small helpers for clarity.
4. Put each argument on its own line in long calls and definitions. Use descriptive
   names, consistent indentation and blank lines between meaningful steps.
5. Start each module with a concise two-to-three-paragraph docstring covering its
   scientific purpose, responsibilities, usage and relevant inputs and outputs.
6. Comment scientific meaning, units, assumptions and non-obvious API choices beside
   the code. Introduce meaningful steps without narrating obvious Python operations.

<a id="configuration"></a>

## 3. Configuration

1. Define a plain `configuration(...)` in `recipe.py` returning a fresh dictionary
   of resolved scientific settings. Keep defaults together in a readable `dict(...)`.
2. Use explicit arguments for supported overrides; validate protocol names and values.
3. Include architecture, data selection, preprocessing, numerical and training
   settings, recording choices and interpretation where applicable. State units
   in names and explain non-obvious semantics.
4. Resolve defaults and overrides before deriving dependent values. Copy mutable
   constants so calls do not share configuration state.
5. Route CLI overrides through `configuration(...)`; do not scatter scientific
   defaults or silently patch the returned dictionary.
6. Execute from and record the resolved configuration; keep derived values
   consistent with the settings used.

<a id="identity-and-dependencies"></a>

## 4. Identity and dependencies

1. Allocate a free, zero-padded `expXXX` never used before, including deleted
   experiments. Check directories, Git history and Pingstore allocation records.
2. Keep experiment-specific code inside its own directory.
3. Keep shared helpers directly in `experiments/`; reuse and extend maintained
   helpers instead of copying their machinery.
4. Obtain user permission before creating a new shared helper in `experiments/`.
5. Consume other experiments' data only through explicitly identified, validated
   Pingstore runs.
6. Never import another experiment's code or `.scratch/` code, or depend on
   pre-existing scratch files.

<a id="migrating-scratch-work-to-an-experiment"></a>

## 5. Migrating scratch work to an experiment

1. Allocate a never-used ID under [Identity and dependencies](#identity-and-dependencies).
2. Adapt required source to the standard layout: settings in `recipe.py`, separate
   execution stages, extra modules in `custom/`, tests in `test.py` and a brief CLI README.
3. Remove scratch imports, hard-coded scratch paths and local-file dependencies.
   Reuse maintained helpers and SNNLab; identify scientific inputs explicitly.
4. Preserve existing outputs' original settings and provenance. Follow
   [Pingstore Rules](/notes/pingstore-rules) for import or recovery, or generate
   new runs; moving files does not validate evidence or change its settings.
5. Run tests, check side-effect-free imports and verify independence from the
   original scratch directory. Follow Compute and Pingstore Rules for execution.
6. Remove the scratch directory only after verification and preservation of needed
   source and results. Update migrated report paths and evidence under Writing Rules.

<a id="snnlab-first"></a>

## 6. SNNLab first

1. Use SNNLab's network, simulation, inference, training, recording, analysis and
   visualization features where available. Check the installed API before adding
   custom scientific machinery; do not reimplement equivalent features.

<a id="network-diagrams"></a>

### 6.1. Network diagrams

1. Save one circuit diagram per unique architecture from an actual compiled bundle:
   `bundle.visualise(diagram_path, view="circuit")` (`snnlab.lang.Bundle.visualise`).
2. Save the complete bundle separately with `bundle.write(bundle_path, visualise=False)`.
   Avoid `visualise=True` per sweep bundle; it generates all three views in SVG and PNG.
3. Reuse diagrams across parameter, seed, timestep and sweep changes; generate
   another only when the network structure changes.

<a id="network-snapshots"></a>

### 6.2. Network snapshots

1. Capture resolved network settings and parameters with SNNLab's native snapshots.
2. For training, require end-of-training and best-validation-checkpoint snapshots;
   use each checkpoint's actual parameters. A pre-training snapshot is not required.
3. For simulation, including inference, require only a simulation-start snapshot.
4. Pass native snapshots directly to execution calls:

   ```python
   from snnlab.sim import SnapshotSpec

   # Simulation or inference:
   snapshot = SnapshotSpec(path=snapshot_path, stage="start", parameters=True)

   # End of training:
   snapshot = SnapshotSpec(path=snapshot_path, stage="end", parameters=True)

   # Pass snapshot=snapshot to simulate(...) or train(...).
   ```

5. Call keyword-only `simulate(...)` and `train(...)` directly, spelling out arguments
   without request-dictionary unpacking or wrappers. `ExecutionSpec` is internal
   since 0.21; `simulate` handles inference since 0.23, when `infer` was removed.
6. Group optimizations in `Performance(...)`; retain requested policy in the recipe
   and native resolved performance metrics. `SNNLAB_DEVICE` overrides automatic selection.
7. Use distinct snapshot paths per execution and retain files in Pingstore.
   Snapshots preserve metadata and parameters; they are not resumable checkpoints.
8. Bind report parameter tables to native snapshots and recorded configuration under
   [Writing Rules](/notes/writing-rules#parameter-table-data-binding). Keep formatting
   in reports; do not duplicate parameter payloads or add parameter-table serializers.
9. For native calls, write the bundle locally with `visualise=False` and let SNNLab
   capture `snapshot=SnapshotSpec(...)`. For custom loops, keep the native lifecycle
   adapter in `custom/`; never duplicate metadata or parameter serialization.

<a id="parallel-compute"></a>

## 7. Parallel compute

1. Follow `experiments/exp124/compute.py` for independent work on one multi-GPU
   host, including Modal.
2. Define work items with stable IDs and device allocations. Validate resources
   before loading data and divide CPU threads between workers.
3. Use one worker function for sequential and parallel work. Prepare shared inputs
   once and verify their checksums before and after execution.
4. Give workers disjoint exports and scratch files. Use
   `hpc.concurrent_compute.run_active_workers` for spawned processes, completion
   records, output-hash validation and cancellation on failure.
5. Give one coordinator ownership of `stage_run`, output validation and merging,
   execution provenance and run completion. Workers must not edit its manifest.
6. Expose `--parallel`; Modal supplies resources and forwards it within one invocation.
   The helper is non-resumable. Follow [Compute Rules](/notes/compute-rules) for
   dispatch and [Pingstore Rules](/notes/pingstore-rules) for storage.

<a id="saving-and-consuming-results"></a>

## 8. Saving and consuming results

1. Use Pingstore for scientific runs, provenance and outputs; follow
   [Pingstore Rules](/notes/pingstore-rules) (`md/notes/pingstore-rules.md`).
2. Create runs with `pingstore.stages.stage_run`; validate explicit inputs with `source_run`.
3. Export scientific outputs through `StageRun.export`; use `StageRun.scratch` for temporary work.
4. Record configuration and input provenance in the run record, not parallel manifests.
5. Invoke compute, analyse and present independently. Never select “latest” or
   automatically launch an upstream stage.
6. Treat completed runs as immutable; changed results require a new run.
