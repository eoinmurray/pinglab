# exp074: independent integration stages

Conforms to Experiment Runner Guide 3.0.0 and Storage Guide 3.0.0.
The article follows Writing Guide 18.0.0. Original scientific constants,
network definitions, workloads and selection rules were preserved; the live
simulator's actual evaluation protocol is reported, not the old article's assumptions.

## Execute explicitly

```sh
uv run python experiments/exp074/compute.py
uv run python experiments/exp074/analyse.py --source <compute-run-id>
uv run python experiments/exp074/present.py --source <analysis-run-id>
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
supporting execution records are retained under `export/evidence/`.

## Fresh execution verified on 2026-08-28

| Stage | Run |
|---|---|
| Compute | `exp074-r003-compute` |
| Analyse | `exp074-r004-analyse` |
| Present | `exp074-r005-present` |

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

The original 0.1 ms timestep is stored as float32 in raster metadata; validation
allows only its representation error. One failed analysis remains hidden.
The earliest compute was superseded after moving simulator execution attachments
out of the scientific export; it was not rewritten. Final measured rates were
20.9228515625 Hz (E) and 62.5 Hz (I), across four 200 ms trials.
