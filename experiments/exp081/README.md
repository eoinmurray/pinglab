# Exp081: filtered-response theory

Independent stages follow Storage Guide 2.0.0 and Experiment Runner Guide 2.0.0.
The article follows Writing Guide 8.0.0.
The physical equations, rate/probe grids, random-seed derivation, integration
scheme and default sample counts are preserved from the combined runner.

```sh
uv run python experiments/exp081/compute.py
uv run python experiments/exp081/analyse.py --source <compute-run-id>
uv run python experiments/exp081/present.py --source <analyse-run-id>
```

Each command prints one completed v3 run ID. `--run-id` accepts an unused v3
identity reserved through the shared stage library before dispatch. No stage
selects a latest run, launches another stage or materializes publication output.
Both downstream stages reject v2, wrong-stage, wrong-experiment and changed inputs.
Presentation also verifies the analysis run's pinned compute source.

- **Compute:** retains every moment-grid and distribution-probe feature sample.
  The authoritative manifest contains the complete scientific configuration,
  including actual random seeds and sample counts; runtime details live in
  `provenance/environment.json`.
- **Analyse:** reads retained samples and computes moments, comparison statistics,
  common histogram bins and analytical frequency responses. It retains these
  numerical outputs and pins its computation without copying raw samples.
- **Present:** reads completed analysis and creates four SVG figures and
  `numbers.json` in a flat export. Its manifest pins analysis and computation.

`PINGLAB_SMOKE=1` selects the existing reduced-draw profile only when computing.
Analysis and presentation use the retained recipe, regardless of their process
environment. `EXP081_DEVICE` selects the compute device; the default remains
automatic CUDA/MPS/CPU selection. Replay requires the same device/runtime for
bitwise random-stream reproducibility.

The old script and `python -m experiments.exp081` now fail with stage directions.
New collection plans use an explicit three-stage adapter, reserve identities
before scheduler submission, and retain exact stage references for resumption.
They do not recapture exp081 as v2. Old monolithic campaign plans fail rather than
being silently rewritten. Publication remains separately authorized.

## Evidence and verification boundary

This refactor does not migrate legacy runs or constitute a new scientific run.
The full-profile local chain is `exp081-r001-compute-local`,
`exp081-r002-analyse-local` and `exp081-r003-present-local`.
The current conformance pass reused those completed runs without simulation.
Historical interpretations remain in the article's Discussion and Conclusion;
they require verification against selected scientific evidence before publication.
The article shows the shared unavailable-data notice when no presentation is
selected. Test fixtures and render checks are not scientific evidence.

For undefined correlations or ratios, analysis now records JSON null instead of
nonfinite numbers; normal nondegenerate comparisons retain the original estimator.
An all-zero distribution uses a 0–5 mV histogram range to keep valid bin edges.
