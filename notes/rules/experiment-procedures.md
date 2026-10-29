---
title: Experiment Procedures
author: Eoin Murray
collection: notes
description: Task-specific scratch migration, native SNNLab calls and parallel execution.
created_at: "2026-10-29T17:42:34Z"
updated_at: "2026-10-29T17:42:34Z"
---

# Experiment procedures

Read only the section needed for the task. Standing constraints live in
[Experiment Rules](/notes/rules/experiment); these examples do not authorize execution.

## Migrating scratch work to an experiment

1. Allocate a never-used ID under [Identity and dependencies](/notes/rules/experiment#identity-and-dependencies).
2. Adapt required source to the standard layout: settings in `recipe.py`, separate
   execution stages, extra modules in `custom/`, tests in `test.py` and a brief CLI README.
3. Remove scratch imports, hard-coded scratch paths and local-file dependencies.
   Reuse maintained helpers and SNNLab; identify scientific inputs explicitly.
4. Preserve existing outputs' original settings and provenance. Follow
   [Pingstore Rules](/notes/rules/pingstore) for import or recovery, or generate
   new runs; moving files does not validate evidence or change its settings.
5. Run tests, check side-effect-free imports and verify independence from the
   original scratch directory. Follow Compute and Pingstore Rules for execution.
6. Remove the scratch directory only after verification and preservation of needed
   source and results. Update migrated report paths and evidence under Writing Rules.

## Network diagrams

1. Save one circuit diagram per unique architecture from an actual compiled bundle:
   `bundle.visualise(diagram_path, view="circuit")` (`snnlab.lang.Bundle.visualise`).
2. Save the complete bundle separately with `bundle.write(bundle_path, visualise=False)`.
   Avoid `visualise=True` per sweep bundle; it generates all three views in SVG and PNG.
3. Reuse diagrams across parameter, seed, timestep and sweep changes; generate
   another only when the network structure changes.

## Network snapshots

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
   [Writing Rules](/notes/rules/writing#parameter-table-data-binding). Keep formatting
   in reports; do not duplicate parameter payloads or add parameter-table serializers.
9. For native calls, write the bundle locally with `visualise=False` and let SNNLab
   capture `snapshot=SnapshotSpec(...)`. For custom loops, keep the native lifecycle
   adapter in `custom/`; never duplicate metadata or parameter serialization.

## Parallel compute

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
   The helper is non-resumable. Follow [Compute Rules](/notes/rules/compute) for
   dispatch and [Pingstore Rules](/notes/rules/pingstore) for storage.
