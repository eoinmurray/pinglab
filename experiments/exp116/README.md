# exp116 — minimal PING onset evidence for exp110

Exp116 asks only whether the four-variable deterministic PING closure has a
continuous oscillatory onset and whether slower inhibition lowers its onset
frequency. It is the deliberately small theoretical experiment intended to
support those claims in exp110.

## Design

1. Continue the reference closure over tonic drive and locate its first
   stable-to-unstable complex-pair crossing.
2. Apply one matched 25-point upward/downward amplitude ramp around the reference
   onset and classify the sampled response using branch agreement and
   amplitude-squared scaling.
3. Repeat onset detection at inhibitory decay times 4.5, 6, 9, 12, 18 and 27 ms.
4. At the 4.5- and 27-ms endpoints only, repeat the calculation at the four
   low/high corners of effective noise and rate relaxation.
5. Repeat every continuation on 401- and 801-point drive grids.

The design contains 14 conditions. It does not include a complete closure grid,
Lyapunov coefficient, dimensional reductions, phase planes, standalone cycle
figures or a claim that the separate spiking network undergoes the same
bifurcation.

## Independent stages

```sh
uv run python experiments/exp116/compute.py
uv run python experiments/exp116/analyse.py --source <compute-run-id>
uv run python experiments/exp116/present.py --source <analyse-run-id>
```

Each command creates one immutable `pingstore.run/v4` run. Stages require
explicit inputs and never select a latest run, launch upstream work or publish.

The numerical engine is reused from `experiments.exp115.numerics`; exp116 owns
its smaller scientific recipe, analysis criteria and presentation. The compute
run records the exact engine digest alongside the exp116 implementation digests.

## Completed execution — 19 September 2026

The validated lineage is `exp116-r002-compute` → `exp116-r003-analyse` →
`exp116-r004-present`. All 14 conditions contained an accepted onset on both
continuation grids. The reference onset was 0.5939047777721255 nA at
27.566444771089603 Hz, and its amplitude ramps were consistent with the
prespecified supercriticality criteria. Reference onset frequency decreased
from 30.197527262692226 to 17.940004453875506 Hz across the inhibitory-decay
sweep. All four robustness corner pairs retained the same direction.

## Presentation refresh — 19 September 2026

`exp116-r005-present` reuses `exp116-r003-analyse` without recomputation. It
replaces the original compound figure with separate Hopf-onset,
sampled-criticality and frequency-versus-GABA figures so the article can present
the four claims independently; the calculated reference frequency is reported
in an article table.
