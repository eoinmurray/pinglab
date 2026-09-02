# exp110 — manuscript presentation synthesis

Exp110 owns figures whose panel composition is specific to the manuscript rather
than to one source experiment. It performs no simulation and no measurement.

```sh
uv run python experiments/exp110/present.py --source <exp054-analysis-run-id>
```

The first migrated output is the former exp054 Figure 6. The presentation reads
the validated exp054 analysis coordinates, whose immutable lineage includes the
exp041 frequency measurements and retained exp033 mean-field evidence. It renders
the nine-panel compound under exp110 and records the exp054 analysis run as its
explicit input. Historical exp054 presentation runs remain unchanged.
