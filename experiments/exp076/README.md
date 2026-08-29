# exp076: independent integration stages

Conforms to Experiment Runner Guide 3.0.0 and Storage Guide 3.0.0.
The article follows Writing Guide 18.0.0. Original scientific constants,
network definitions, workloads and selection rules were preserved; the live
simulator's actual evaluation protocol is reported, not the old article's assumptions.

## Execute explicitly

```sh
uv run python experiments/exp076/compute.py
uv run python experiments/exp076/analyse.py --source <compute-run-id>
uv run python experiments/exp076/present.py --source <analysis-run-id>
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
commands, complete logs and scripts are retained under `provenance/`.

## Fresh execution verified on 2026-08-28

| Stage | Run |
|---|---|
| Compute | `exp076-r001-compute` |
| Analyse | `exp076-r004-analyse` |
| Present | `exp076-r006-present` |

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

The obsolete tools/snn executable path was corrected to tools/snnsim. The
one-step parity test is now executed during compute and its JUnit report and
complete logs are retained; analysis cannot assert equality from hardcoded text.
The test passed for parameters, logits, loss, gradients and one AdamW update.
Training used 144 optimisation and 16 validation images, with three validation
encoder draws. Replay used 160 official-test images and one fixed encoding:
33.75% replay accuracy versus 31.25% validation accuracy is not an exact replay
comparison. The article and final analysis explicitly distinguish those protocols.
