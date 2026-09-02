# exp110 — manuscript presentation synthesis

Exp110 owns figures whose panel composition synthesizes evidence from more than
one source experiment. It performs no simulation and no measurement. Figures
already owned by one source experiment are consumed directly by the manuscript
rather than copied or bundled here.

```sh
uv run python experiments/exp110/present.py --source <exp054-analysis-run-id>
```

The onset output originated as exp054 Figure 6. The presentation reads the
validated exp054 analysis coordinates, whose immutable lineage includes the
exp041 frequency measurements and retained exp033 mean-field evidence. It also
composes the cycle-participation and robustness syntheses from their explicit
validated sources. Historical presentation runs remain unchanged.
