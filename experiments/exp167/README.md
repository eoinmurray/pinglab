# exp167 — AI and PING state metrics

Preserve the single-seed scratch comparison, combine its two rasters and compare
its metrics. Historical data lacks raw spikes, executed bundles and snapshots;
the report explicitly marks this limitation. No new simulation accompanies migration.

1. `compute.py --import-from PATH [--run-id ID]`: import original metrics and raster PNGs, preserving source and checksums. Settings are reconstructed, not authenticated execution evidence.
2. `compute.py --simulate [--seed 42] [--device cpu] [--run-id ID]`: explicitly run both conditions, retaining native bundles, diagrams, start snapshots and sparse spike events. Requires separate simulation authorization.
3. `analyse.py --source COMPUTE_ID [--run-id ID]`: carry forward historical metrics or calculate them from a fresh native run.
4. `present.py --source ANALYSE_ID [--run-id ID]`: one multipanel raster figure in SVG/PDF/PNG, with metrics in `numbers.json` for the report table.
5. `uv run pytest experiments/exp167/test.py`: focused non-simulation checks.

Run each Python CLI with `uv run python experiments/exp167/<stage>.py`.
Native future executions preserve the legacy measurement algorithm, including
first-200-cell selection and binary 1 ms occupancy for correlations.
