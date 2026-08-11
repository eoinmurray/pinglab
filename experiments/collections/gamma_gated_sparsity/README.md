# Gamma-gated sparsity campaign

This package owns the dependency graph and isolated execution of the collection.
It does not own experiment parameters; those remain in the experiment runners.

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

Only the same command with `--live` calls `sbatch`. It submits the five exp022
arrays, aggregation, downstream experiments in dependency order, and finalization.
Use `slurm-status` to combine scheduler and output state. After a failed campaign,
`resume` prints the missing-work plan and `resume --live` submits it. Publication is
a separate `build` command and requires a clean disposable worktree at the campaign
commit.
