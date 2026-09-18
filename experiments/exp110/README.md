# exp110 — manuscript presentation synthesis

Exp110 owns figures whose panel composition synthesizes evidence from more than
one source experiment. It performs no simulation and no measurement. Figures
already owned by one source experiment are consumed directly by the manuscript
rather than copied or bundled here.

```sh
uv run python -m experiments.exp110.present \
  --source <exp054-analysis-run-id> \
  --theory-source <exp115-analysis-run-id>
```

The runner pins the accepted exp041, exp046, exp037 and exp044 presentation
sources in code; they are not runtime recipe overrides.

For the onset compound, exp110 reads the coupling grid directly from exp054,
the mean-field onset and numerical-criticality evidence directly from exp115, and the trained-network
frequency measurements directly from exp041. It validates and joins these
independent sources itself. It also
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

## 2026-09-14 — Manuscript grounding and frequency labels

The grounding review verified the completed equal-damping bank and the
replacement exp025/exp037/exp038 compute results. It corrected the manuscript's
10-Hz-ceiling COBA rate to 9.0 Hz, clarified which training and evaluations were
repeated, and qualified comparisons at tighter ceilings as not rate-matched.

The complete timescale sweep now uses population-cycle terminology and
spectral-peak notation because its slowest condition is approximately 12 Hz.
The presentation-only refresh `exp110-r024-present` updates Figures 2 and 6
labels with all seven source references unchanged from `exp110-r022-present`.
Its payload is
`sha256:980bbd9c827739fb81beafce927df061e828be8ef3c627d6d74b1fe2548f58a8`.
No training, inference or analysis stage was rerun. The intermediate
`exp110-r023-present` remains unchanged.

See the [grounding review](../../reviews/exp110-grounding-2026-09-14.md) for
the verified values, complete article selections and remaining evidence gaps.

## 2026-09-14 — Equal-network cycle participation

Presentation recipe v12 uses `exp046-r010-present` and its pinned
`exp046-r007-analyse` for Figure 6C–H. The compound redraws the same six
equal-network distributions as exp046 Figure 1, using black bars in one row
without individual-network markers. Figure 6A's labels alternate sides of their
points to avoid overlap; its measurements and all other source pins are unchanged.

The completed presentation is `exp110-r027-present`. Its 24 distribution bars
match exp046 Figure 1 exactly. Figures 2 and 7 retain byte-identical PNGs.
Intermediate layout presentations `exp110-r025-present` and
`exp110-r026-present` remain unchanged.

The manuscript's Results, Figure 6 caption, Methods and Appendix B3 now describe
only the equal-network estimator. Appendix Figure B1 and the comparison with
opportunity pooling were removed. The summary reports 76.35% zero-spike,
22.09% one-spike and 1.56% multiple-spike pairs, or 98.44% with at most one spike.
These are means across the 18 equally weighted networks, not newly measured data.
No training, inference or analysis was rerun. The existing plotting regression
was updated; tests were not run under the author's pre-commit instruction.

## 2026-09-18 — Numerical criticality source

Presentation recipe v14 replaces exp033 with `exp115-r008-analyse` as the
mean-field source. Figure 2G now shows the leading eigenvalue real part across
drive; Figure 2H uses exp115's fixed-sample upward/downward amplitude ramps;
Figure 2I uses its six refined onset frequencies. The manuscript Appendix C now
matches exp115's analytic Jacobians, three continuation grids, scaled gain
quadrature and numerical criticality criteria.

The completed presentation is `exp110-r029-present`, retaining
`exp054-r016-analyse` and the prior exp037, exp041, exp044 and exp046 sources.
Its payload digest is
`sha256:e42da7a0410d941b277a2a1172c27a803b533153e3e9602a65cff5da5d22ae02`.
No simulation, training or analysis was rerun. Five exp110 tests, Ruff, visual
inspection of Figure 2 and the 22-entry Demolab build passed.
