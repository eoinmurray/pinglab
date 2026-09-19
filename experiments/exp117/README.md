# exp117 — independent mean-field PING bifurcation analysis

Exp117 independently derives and analyses a four-variable population-rate
closure of the conductance-based E–I circuit.

The planned work is to:

1. derive the mean-field model;
2. locate any Hopf bifurcation and its frequency;
3. determine whether the bifurcation is supercritical or subcritical; and
4. plot Hopf frequency against the GABA decay time constant.

`recipe.py` owns literal copies of established neuronal, synaptic and network
parameters plus the experiment's explicit mean-field assumptions and numerical
protocol. The implementation imports no other experiment and declares no
upstream run.

## Current result

The equilibrium equations were reduced analytically to one scalar
self-consistency residual and solved with a bracketed Brent method over 401
external-drive values from 0 to 4 nA. The analytical flow Jacobian identified a
simple Hopf bifurcation at 0.59390478 nA with angular frequency
0.17320508 rad/ms, corresponding to 27.56644 Hz. The two remaining eigenvalues
had real part −0.45833333 per ms at the crossing, and the critical pair crossed
transversely with slope 0.28957071 per ms per nA.

The nonlinear extension tests objective 3 with ascending and descending
near-onset drive ramps. Their peak-to-peak excitatory-rate amplitudes differed
by at most 0.00687 Hz, crossed the amplitude threshold at the same sampled
drive, and gave a linear squared-amplitude fit with R² = 0.9994. These results
are consistent with a supercritical Hopf bifurcation under the predefined
criteria. The sampled test cannot exclude a narrower bistable interval or an
unstable periodic orbit; no first Lyapunov coefficient is calculated.

Across the prescribed inhibitory decay constants of 4.5, 6, 9, 12, 18 and 27
ms, the refined Hopf frequency decreased monotonically from 30.19753 to
17.94000 Hz. This completes objective 4 for the deterministic closure.

## Commands

```sh
uv run python experiments/exp117/compute.py
uv run python experiments/exp117/analyse.py \
  --source <exp117-compute-run-id>
uv run python experiments/exp117/present.py \
  --source <exp117-analyse-run-id>
```

Each command creates one `pingstore.run/v4` run. Analyse explicitly consumes the
compute run, and present explicitly consumes the analysis run. The retained
presentation contains `bifurcation_compound.svg` and `numbers.json`.
