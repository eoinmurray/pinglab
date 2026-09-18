# exp115 — numerical Hopf investigation of the PING closure

Exp115 studies a four-variable population-rate closure of the canonical
conductance-based E/I circuit. It locates equilibrium Hopf crossings and tests
sampled criticality with matched upward and downward time-domain ramps. The
effective voltage-noise scale and rate-relaxation multiplier are prescribed
closure choices, not fits to spiking observations.

## Independent stages

1. Compute equilibrium continuations for all 120 closure conditions and
   criticality ramps at the six inhibitory decay times of the reference closure:

   ```sh
   uv run python experiments/exp115/compute.py
   ```

2. Analyse one explicit compute run:

   ```sh
   uv run python experiments/exp115/analyse.py --source <compute-run-id>
   ```

3. Present one explicit analysis run:

   ```sh
   uv run python experiments/exp115/present.py --source <analyse-run-id>
   ```

Each command creates one immutable v4 run. Stages never select a latest run,
launch another stage or publish.

## Revised numerical design

1. Equilibria and Hopf crossings are calculated on 401-, 801- and 1,601-point
   drive grids. Crossing acceptance checks the critical eigenpair, the remaining
   modes, the quartic identity, transversality and convergence of drive and
   frequency.
2. For `sigma_mV = 4` and `kappa = 1`, each accepted onset receives a 25-point
   drive ramp from 0.10 nA below to 0.55 nA above onset. The model integrates for
   2 s per point upward and downward while carrying endpoints between steps.
3. Analysis measures peak-to-peak excitatory-rate amplitude over the final
   500 ms. A branch gap below `1e-4 /ms`, positive amplitude-squared slope and
   `R² > 0.9` are labelled `consistent_with_supercritical`; all other completed
   ramps are `subcritical_or_inconclusive`.
4. This finite-duration, finite-grid test cannot exclude a narrower bistable
   interval or an unstable periodic orbit. It classifies sampled behaviour of
   the deterministic closure, not the separate spiking network.

## Completed recipe-v3 execution — 18 September 2026

The validated lineage is `exp115-r007-compute` → `exp115-r008-analyse` →
`exp115-r009-present`. All 120 closure conditions contained one accepted Hopf
onset. All six reference-closure decay-time ramps were consistent with
supercriticality under the predefined finite-grid criteria; none was unresolved
or classified subcritical/inconclusive.

The reference condition had onset drive 0.5939047777721255 nA and frequency
27.566444771089603 Hz. Its branch gap was 7.424340296782739e-6 /ms and its
amplitude-squared fit had R² 0.9994110749117254. Across the six decay times,
onset frequency decreased from 30.197527262692226 to 17.940004453875506 Hz.

The earlier recipe-v2 lineage `exp115-r004-compute` → `exp115-r005-analyse` →
`exp115-r006-present` calculated first Lyapunov coefficients. Those immutable
runs remain historical evidence but are not inputs to the revised article. They
must not be relabelled as amplitude-ramp results.

## Presentation refresh — 18 September 2026

`exp115-r010-present` reuses `exp115-r008-analyse` without recomputation. It
moves the article's parameter table after Methods and redraws the three-panel
figure at 180 × 64 mm using the shared paper typography, canonical panel labels
and distinct upward/downward ramp encodings. Its payload digest is
`sha256:dc4450b18a540a5705cfa3a5681484b392ad68c6d5f9591a13120e46ab30b997`.
