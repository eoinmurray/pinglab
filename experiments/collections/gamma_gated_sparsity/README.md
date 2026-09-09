# Gamma-gated sparsity campaign

This package owns the dependency graph and isolated execution of the collection.
Exp022 owns shared training contracts: public TR IDs, registered cell names,
training grids, seeds, and training-only cell parameters. Downstream runners read
those values through `training_run_cells`, `training_run_values`, and
`training_run_cell`; they do not copy them. Each downstream runner continues to
own its inference interventions, evaluation grids, raster samples, plotting, and
other analysis-only parameters.

## Adopted model and retained evidence — 2026-09-09

Hidden E/I neurons use **1.2/0.6-ms refractory reset holds**. These are the
physical durations already executed by the retained 0.1-ms spiking runs.
`experiments.helpers.operating_point` owns the explicit collection override;
generic simulator defaults remain separate. The timestep experiment uses
**0.05, 0.1, 0.2, 0.3 and 0.6 ms**, each representing both refractory holds
exactly. Nominal 200-ms trials last 199.8 ms at 0.3/0.6 ms and 200 ms otherwise;
analysis uses realised durations.

The accepted replacement evidence is:

| Experiment | Compute | Analyse | Present |
| --- | --- | --- | --- |
| exp022 | exp022-r007-compute | exp022-r010-analyse | exp022-r014-present |
| exp023 | exp023-r011-compute (reused) | exp023-r012-analyse (reused) | exp023-r014-present (metadata correction) |
| exp033 | exp033-r011-compute | exp033-r012-analyse | exp033-r015-present |
| exp044 | exp044-r007-compute | exp044-r008-analyse | exp044-r009-present |
| exp054 | exp054-r008-compute (reused spikes) | exp054-r013-analyse | exp054-r014-present |
| exp110 | — | — | exp110-r021-present |

The exp022 bank contains 90 unchanged reused models and twelve new timestep
models. Exp054's analysis separately pins exp033's replacement theory and
unchanged exp041 frequencies. Other spiking consumers retain their original
source pins, including the earlier exp022 bank. Exp080/081 are passive models;
exp048/111 and noncollection experiments are outside this reconciliation.
Inserted transmitted spikes and inhibitory replay retain their experimental
semantics; they do not acquire native-neuron refractory gating. In particular,
the retained 43.95-Hz reference defines replay clock windows, not the current
measured spectral frequency or inferred cycle boundaries.

New exp033 theory uses recipe v2; new combined exp054 compute uses v6. Both use
1.2/0.6-ms gains with cancellation-resistant evaluation. Old scientific recipe
versions and recorded payloads remain unchanged. The explicit exp054
`--theory-source` refresh is a separate workflow from a new campaign's combined
compute. This table documents evidence; it does not select publication inputs
or authorize rerunning an entire campaign. See the root PLAN.md for exact
digests, validation and the staged repair history.

## Campaign execution

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
status document per shard. Exp024 dispatches independent analyse and present
stages from the campaign's explicit completed exp022 compute run. It records
pinned stage references, not a materialized output directory; stage IDs are
reserved before live submission. Historical campaigns require their original
checkout. See the [exp024 notes](../../exp024/README.md) for source validation,
retries, preview and retention of referenced evidence. Other downstream
experiments not yet migrated remain monolithic. Exp042 now dispatches eight
reserved compute shards, completes its compact v3 compute evidence, and invokes
analyse and present independently; see [its stage notes](../../exp042/README.md).
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
