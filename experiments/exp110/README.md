# exp110 — manuscript presentation synthesis

Exp110 owns figures whose panel composition synthesizes evidence from more than
one source experiment. It performs no simulation and no measurement. Figures
already owned by one source experiment are consumed directly by the manuscript
rather than copied or bundled here.

```sh
uv run python -m experiments.exp110.present --source <exp054-analysis-run-id> \
  --exp041-source <exp041-present-run-id> --exp046-source <exp046-present-run-id> \
  --exp037-source <exp037-present-run-id> --exp044-source <exp044-present-run-id>
```

The onset output originated as exp054 Figure 6. The presentation reads the
validated exp054 analysis coordinates, whose immutable lineage includes the
exp041 frequency measurements and retained exp033 mean-field evidence. It also
composes the cycle-participation and robustness syntheses from their explicit
validated sources. Historical presentation runs remain unchanged.

## 2026-09-09 — Refractory reconciliation (PLAN 5.10)

Presentation recipe v11 accepts exp054's separate-source theory-refresh
configuration. Plot settings use its preserved spike recipe; run provenance keeps
the complete spike and theory configuration. The ordinary collection dispatcher
uses the combined-compute route (new exp054 executions now use recipe v6);
use the explicit command
below for this refresh.

```sh
uv run python -m experiments.exp110.present --source exp054-r013-analyse \
  --exp041-source exp041-r005-present --exp046-source exp046-r005-present \
  --exp037-source exp037-r017-present --exp044-source exp044-r009-present
```

This produced `exp110-r021-present` (payload
`sha256:b33dcc71d836b61aec282b3c6b9f8a1c4993149f45c599a82df995e3428bb37d`)
in two recorded seconds, with six exports. It replaces the theory and timestep
sources used by `exp110-r020-present`, retaining all five exp037/exp041/exp046
source references exactly. Figure 6's PNG is byte-identical; Figures 2 and 7
reflect the replacement evidence. Exp023's metadata-corrected
`exp023-r014-present` is a direct article input, with unchanged scientific images,
and therefore is not bundled into this presentation.

The manuscript now reports the 0.05–0.6-ms sweep, fixed E/I reset holds of
1.2/0.6 ms, realised presentation durations, and the 0.594-nA mean-field onset.
It records completed replacement training and reuse, and the accompanying
gain-integrand cancellation correction. The existing Methods/appendix scaffolds
and standalone article structure remain. Fourteen targeted tests passed; all
three scientific PNGs were inspected and the local 38-entry article build passed.
