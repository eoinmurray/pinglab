# Exp024: convergence audit

Uses the [Experiment Runner Guide](../README.md), [Storage Guide](../../tools/pingstore/README.md),
and [Writing Guide](../../writings/README.md). No training or inference is performed.

## Independent stages

```sh
uv run python experiments/exp024/analyse.py --source exp022-r001-compute
uv run python experiments/exp024/present.py --source <printed-exp024-analyse-id>
```

`recipe.py` owns the audit definitions and queries exp022 for the six TR-02
unregularised baseline identities. Analysis requires complete, contiguous
histories and explicit validation-split metadata for all six cells. It reads
configurations from the selected evidence, not today's training defaults.
Its `export/` contains `curves.json` and `results.json`; source configurations
are retained under `provenance/`. Checkpoints remain in the source compute run.

Presentation reads saved analysis, not cell histories. It writes the existing
three SVGs and `numbers.json` into its flat `export/`, with figure lineage in
`run.json`. Both stages validate and pin their inputs, write v3 through shared
helpers, and print a new stage ID. `--run-id` accepts an unused reservation only.
Sources must declare their stage and use v3. Shared operational source validation
now rejects typed and untyped v2 before reserving a downstream run. The
[verified exp022 ancestor repair](../exp022/ANCESTRY.md) resolved the historical
source prerequisite and re-pinned all five retained exp024 runs without changing
their scientific values. Operational inputs and pins resolve to validated v3 runs.
For a diagnostic compute run with a pinned `bank` input, analysis resolves and
records that exact bank as an additional input.

## Measurement definitions

The slope is the difference between the last and first values of the last ten
epochs, divided by their epoch separation, **not** a regression slope. Histories
shorter than ten epochs use their entire available window (minimum two), recorded
explicitly. Stability requires absolute slope strictly below 0.1 percentage
points/epoch for accuracy or 0.05 Hz/epoch for excitatory rate. This is a finite
window diagnostic, not an asymptotic convergence proof.

The dotted marker is the first epoch reaching 99% of final accuracy, averaged
across seeds. It does not require subsequent accuracy to stay above the threshold
and does not imply an unchanged decision boundary. Weight-norm drift starts at
the first **completed epoch**, not initialization. Model summaries retain means,
sample standard deviations, and counts of seeds meeting each threshold.
Margin/confidence/logit-scale studies are future work, not new outputs here.

## Compatibility and collection execution

The flat launcher and `python -m experiments.exp024` reject the retired combined
execution with directions to the two stages. The public import interface exposes
`MODELS`, `SEEDS`, and `cell_name`; it does not resolve mutable training roots.
Unused legacy plotting/probe helpers are no longer part of the execution path.

New collection plans use the exp024 adapter, dispatching analysis and presentation
with explicit IDs and recording their checksum-pinned references in the campaign
working directory. Scheduler execution reserves stage identities before submission.
The adapter does not materialize output or stamp mutable metadata into completed
runs. Finalization excludes exp024 from legacy v2 recapture. Keep the referenced
repository Pingstore runs when transferring a campaign: its reference document
does not contain the scientific evidence. Preserve legacy campaign checkouts,
plans, reservations and completed runs as historical evidence; this does not
permit v2 execution or reservation completion. New executions require v3.

## Preview and publication

Select an exp024 present run in Demolab preview. The article's explicit input
binding works unchanged with discovery; absent inputs show the unavailable-data
notice. Run stages never change `.artifacts/` or publish the site. Materialization
and publication remain separate authorized operations. Historical exp024 outputs
have not been migrated or replaced.

Run IDs now use the [source-neutral convention](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md).
Execution origin remains in `run.json`; migration preserved scientific results.
