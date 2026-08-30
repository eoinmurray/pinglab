# exp075: independent integration stages

Conforms to Experiment Runner Guide 4.3.0 and Storage Guide 4.3.0.
The article follows Writing Guide 18.0.0. Original scientific constants,
network definitions, workloads and selection rules were preserved; the live
simulator's actual evaluation protocol is reported, not the old article's assumptions.

## Execute explicitly

```sh
uv run python experiments/exp075/compute.py
uv run python experiments/exp075/analyse.py --source <compute-run-id>
uv run python experiments/exp075/present.py --source <analysis-run-id>
```

Each command completes one fresh immutable v4 run. The flat runner and bare
package entrypoint reject combined execution. Recipe imports perform no work.
No stage starts another stage, discovers a latest input, seeds from an active
artifact view, materializes output, or publishes. `--run-id` accepts only an
unused source-neutral reservation allocated before dispatch.

Compute retains bundles, recordings/checkpoints and acquisition evidence.
Analyse measures explicit pinned compute evidence. Present renders the analysis
and its exact compute ancestry into flat exports. The scoped shared helper
`experiments/helpers/snnlang_stages.py` validates v4 layout, payload and run record
pins throughout the ancestry before use and again before completion. Simulator
supporting execution records belong in `run.json`; transient logs use discarded `.scratch/`.

## Fresh execution verified on 2026-08-28

| Stage | Run |
|---|---|
| Compute | `exp075-r002-compute` |
| Analyse | `exp075-r005-analyse` |
| Present | `exp075-r006-present` |

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

Training used 900 optimisation and 100 validation images, four epochs and three
fixed validation encoder draws. Checkpoint selection minimises mean validation
cross-entropy (accuracy and earliest epoch break ties); it does not maximise
accuracy. The selected epoch was 4, with 63.666666666666664% validation accuracy.
Earlier compute/presentation attempts were superseded by the documented chain;
they were preserved rather than relabelled or overwritten.
