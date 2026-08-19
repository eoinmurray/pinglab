# Gamma-gated sparsity campaign

This package owns the dependency graph and isolated execution of the collection.
Exp022 owns shared training contracts: public TR IDs, registered cell names,
training grids, seeds, and training-only cell parameters. Downstream runners read
those values through `training_run_cells`, `training_run_values`, and
`training_run_cell`; they do not copy them. Each downstream runner continues to
own its inference interventions, evaluation grids, raster samples, plotting, and
other analysis-only parameters.

Initialize a production campaign from a clean, frozen checkout:

```bash
uv run python -m experiments.collections.gamma_gated_sparsity init \
  --campaign-root /absolute/persistent/path/campaigns/<campaign-id> \
  --campaign-id <campaign-id>
```

Copy `resources.example.json` outside the repository and replace every placeholder
with the Wilkes3 account, persistent paths, and resources measured by diagnostics
and canaries. Review the complete dependency submission without spending compute:

```bash
uv run python -m experiments.collections.gamma_gated_sparsity submit \
  --campaign-root /absolute/persistent/path/campaigns/<campaign-id> \
  --resources /absolute/private/path/resources.json
```

The review payload includes the frozen source and lockfile, exp022 manifest
hash, resource-file hash, tier cell lists, exact commands and dependencies, and
expected outputs. On Wilkes3, add `--test-only` to ask Slurm to validate every
job shape without creating jobs. `--test-only` and `--live` are mutually
exclusive.

Only the same command with `--live` creates jobs. It submits the five exp022
arrays, aggregation, downstream experiments in dependency order, and finalization.
Exp037, exp042, and exp082 declare bounded inference shards followed by a normal
experiment aggregation job. The campaign records the reviewed condition and
simulator-launch contracts, rejects drift in the condition count, and writes one
status document per shard. Other downstream experiments remain monolithic.
Use `slurm-status` to combine scheduler and output state. After a failed campaign,
`resume` prints the missing-work plan and `resume --live` submits it. Publication is
a separate `build` command and requires a clean disposable worktree at the campaign
commit.

Use the identical commands with a campaign initialized using `--smoke` for the
Slurm rehearsal. Smoke and production require separate unique campaign roots and
resource files; the smoke campaign exercises every registered tier and downstream
dependency with reduced experiment-owned workloads.

The private resource file has separate `downstream` and `heavy_downstream`
profiles. The latter applies only to the bounded inference arrays; their aggregate
jobs use the ordinary downstream profile. Resource values must come from the most
recent Wilkes benchmark rather than the placeholder example.

Before the production arrays, use `canaries` with the production campaign and
provisional resource file. It selects one still-missing cell from each of the
five exp022 tiers. Review it dry, run it with `--test-only`, and use `--live`
only after approval. Successful canary cells belong to the production bank and
are skipped by the later full submission. Replace provisional requests with
the measured wall time, host memory, and GPU memory plus reviewed margins.
