# exp111 — independent Brian2 comparisons

This experiment compares snnsim with independently encoded Brian2 dynamics for
the mechanisms and manuscript-level protocols used by the gamma-gated sparsity
collection.

Run the three stages with explicit source identities:

```sh
uv run python experiments/exp111/compute.py --source <exp022-compute-run>
uv run python experiments/exp111/analyse.py --source <exp111-compute-run>
uv run python experiments/exp111/present.py --source <exp111-analyse-run>
```

The experiment reports the signed, absolute and relative distance between
backends for each measured quantity. It does not classify comparisons as passes
or failures.
