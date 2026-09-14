# exp110 — manuscript presentation synthesis

Exp110 owns figures whose panel composition synthesizes evidence from more than
one source experiment. It performs no simulation and no measurement. Figures
already owned by one source experiment are consumed directly by the manuscript
rather than copied or bundled here.

```sh
uv run python -m experiments.exp110.present --source <exp054-analysis-run-id>
```

The runner pins the accepted exp041, exp046, exp037 and exp044 presentation
sources in code; they are not runtime recipe overrides.

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
uv run python -m experiments.exp110.present --source exp054-r013-analyse
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

## 2026-09-14 — Equal gradient damping in COBA and PING

The replacement exp022 bank was completed on HPC as `exp022-r011-compute`,
with all 21 COBA training replicates using voltage-gradient damping 1,000 and
the other 81 models reused byte-for-byte. Because the isolated HPC checkout
allocated identities already present locally, the completed exports were
imported without scientific re-execution under collision-free local identities:

- bank: `exp022-r015-compute`;
- exp025: `exp025-r008-compute` → `exp025-r009-analyse` →
  `exp025-r010-present`;
- exp037: `exp037-r018-compute` → `exp037-r019-analyse` →
  `exp037-r020-present`;
- exp038: `exp038-r009-compute` → `exp038-r010-analyse` →
  `exp038-r011-present`.

Each imported record retains the byte-identical payload digest and the original
HPC execution, timing and source provenance. A comparison of 173 unchanged PING
JSON outputs found no missing files, scientific-configuration changes, accuracy
changes or spike/count changes. The largest numerical rate difference was
0.23 Hz and reflected floating-point evaluation variation rather than recipe
drift.

The local synthesis `exp110-r022-present` pins `exp037-r020-present` while
retaining `exp054-r013-analyse`, `exp041-r005-present`,
`exp046-r005-present` and `exp044-r009-present`. Its payload digest is
`sha256:87acaed5752a7ddc50a35ad5c7c5816fd3d5da6a993a3ebf7ad744d649b751c3`.
The manuscript now selects the replacement exp025, exp037, exp038 and exp110
presentations and reports equal damping for COBA and PING. Historical runs are
unchanged.

The final local validation passed: all exp110 tests succeeded, the 20-entry
Demolab site built successfully, and the four selected replacement presentation
roots resolved through 26 ancestry-linked runs with matching declared input
digests. The rendered exp110 article contained 33 sequential equations, 14
labelled figures or tables, 11 images with alternative text, no broken local
experiment links, and values consistent with the replacement `numbers.json`
exports.
