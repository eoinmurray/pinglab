# Exp081: filtered-response theory

For GPU compute on Modal, use the [shared Modal wrapper](../helpers/MODAL.md)
with `--experiment exp081`. It downloads a validated compute run to local Pingstore.

## Native full-profile run — 2026-10-14

`exp081-r002-compute` ran on Modal A10G/CUDA with 512 draws per moment-grid
condition and 4,096 draws per distribution rate. Compute took approximately one
second, excluding image build and container startup. Its retained Volume copy
and local Pingstore copy passed v4 layout and payload-checksum validation.
An interrupted transfer was recovered without repeating the computation.

Local `exp081-r003-analyse` pins that compute, and `exp081-r004-present` pins
both stages and exports four SVG figures plus `numbers.json`. All three runs
and their exact lineage references were validated; the four figures were
visually checked. Quadrature refinement changed predicted variance by at most
`9.875618987389394e-07` relative. The article's availability tag is now `data`;
its authored dates and review status were preserved. No publication or test
suite was run. The failed pre-compute reservation `exp081-r001-compute` was
removed after its Modal app stopped and the successful later runs were validated.

Local-data update, 2026-10-05: runs unused by the latest exp110 manuscript
were explicitly deleted. Historical operation and size reports below describe
the earlier store; they do not establish current local availability.

Independent stages follow Storage Guide 4.3.0 and Experiment Runner Guide 4.3.0.
The article follows Writing Guide 8.0.0.
The physical equations, rate/probe grids, random-seed derivation, integration
scheme and default sample counts are preserved from the combined runner.

```sh
uv run python experiments/exp081/compute.py
uv run python experiments/exp081/analyse.py --source <compute-run-id>
uv run python experiments/exp081/present.py --source <analyse-run-id>
```

Each command prints one completed v4 run ID. `--run-id` accepts an unused v4
identity reserved through the shared stage library before dispatch. No stage
selects a latest run, launches another stage or materializes publication output.
Both downstream stages reject v2, wrong-stage, wrong-experiment and changed inputs.
Presentation also verifies the analysis run's pinned compute source.

- **Compute:** retains every moment-grid and distribution-probe feature sample.
  The authoritative manifest contains the complete scientific configuration,
  including actual random seeds and sample counts; runtime details live in
  `run.json`.
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
No collection adapter remains: scheduler submission must invoke the independent
stage commands with explicit reserved identities and source references. Old
monolithic campaign plans remain historical and cannot be resumed. Publication
remains separately authorized.

## Evidence and verification boundary

This refactor does not migrate legacy runs or constitute a new scientific run.
The former full-profile local chain comprised the earlier exp081 compute run,
the earlier exp081 analyse run and `exp081-r003-present`.
The current conformance pass reused those completed runs without simulation.
Historical interpretations remain in the article's Discussion and Conclusion;
they require verification against selected scientific evidence before publication.
The article shows the shared unavailable-data notice when no presentation is
selected. Test fixtures and render checks are not scientific evidence.

For undefined correlations or ratios, analysis now records JSON null instead of
nonfinite numbers; normal nondegenerate comparisons retain the original estimator.
An all-zero distribution uses a 0–5 mV histogram range to keep valid bin edges.

Run IDs now use the [source-neutral convention](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md).
Execution origin remains in `run.json`; migration preserved scientific results.
