# exp077: independent integration stages

Conforms to Experiment Runner Guide 3.0.0 and Storage Guide 3.0.0.
The article follows Writing Guide 18.0.0. Original scientific constants,
network definitions, workloads and selection rules were preserved; the live
simulator's actual evaluation protocol is reported, not the old article's assumptions.

## Execute explicitly

```sh
uv run python experiments/exp077/compute.py
uv run python experiments/exp077/analyse.py --source <compute-run-id>
uv run python experiments/exp077/present.py --source <analysis-run-id>
```

Each command completes one fresh immutable v3 run. The flat runner and bare
package entrypoint reject combined execution. Recipe imports perform no work.
No stage starts another stage, discovers a latest input, seeds from an active
artifact view, materializes output, or publishes. `--run-id` accepts only an
unused source-neutral reservation allocated before dispatch.

Compute retains bundles, recordings/checkpoints and acquisition evidence.
Analyse measures explicit pinned compute evidence. Present renders the analysis
and its exact compute ancestry into flat exports. The scoped shared helper
`experiments/helpers/snnlang_stages.py` validates v3 layout, payload and manifest
pins throughout the ancestry before use and again before completion. Simulator
supporting execution records belong in `run.json`; transient logs use discarded `.scratch/`.

## Fresh execution verified on 2026-08-28

| Stage | Run |
|---|---|
| Compute | `exp077-r001-compute` |
| Analyse | `exp077-r002-analyse` |
| Present | `exp077-r003-present` |

These are new executions, not historical imports. The selected chain's payloads
and exact upstream pins were validated. Presentation is available locally;
no publication defaults or materialized outputs were changed. Completed
intermediate runs and hidden failed work remain untouched; do not use an
incomplete run as evidence. The table names the final verified chain, not an
automatically selected or published run.

## Verification

The shared synthetic tests in `experiments/helpers/test_snnlang_stages.py` exercise
stage isolation, explicit inputs, immutable sources, v2 and corruption rejection,
transitive source changes and failed-stage preservation. Real presentations were
rendered against the articles in both HTML and paged PDF. Runtime-specific
results below refer to retained new evidence, not synthetic tests.

The missing goal-file dependency and copied historical activity log were removed
from new execution. Original historical statements remain in version control;
they are not represented as newly executed evidence. All four graph variants,
input tensors and original parity/performance workload sizes were preserved.
Parity acquisition retains actual comparison tensors, a checkpoint and every
individual timing, with measurement performed by analyse. Numerical pulse-delay
and causal-planning tests execute during compute and retain JUnit evidence.
Fresh measurements found zero parameter/output/spike/replay discrepancies and
-26.564785990591833% median graph runtime overhead, passing the unchanged 10%
gate. The bounded Inductor first call took 21.36156795802526 seconds. Compiled
repeat equality does not establish compiled-versus-eager equality. Python memory
tracing excludes native tensor allocations. These are one-session CPU timings.
