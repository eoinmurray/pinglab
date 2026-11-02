# exp164 — Untrained digit routing with partial recruitment

Two fixed 8×8 digits drive a standalone A/B → C spiking circuit. Opposite cue
orders share complete sensory/background input draws. No classifier, readout or
synaptic weight is trained. The first 1,100 ms are shown; measurement uses
650–1,100 ms and conductance close-ups use 900–1,050 ms.

1. `uv run python experiments/exp164/compute.py` runs the two-condition CPU batch
   for 2,400 ms. `--seed` sets input generation (default 101); native seed is 85.
2. `uv run python experiments/exp164/analyse.py --source COMPUTE_ID` validates
   matched inputs, reconstructs conductances using native weights/planner timing,
   pools raw E counts within known pixel groups, and records phase/routing checks
   plus raw-spike recruitment and diversity.
3. `uv run python experiments/exp164/present.py --source ANALYSE_ID` renders the
   two cue figures, all-neuron rasters and native network diagram. All stages
   accept `--run-id`; analysis and presentation never run upstream stages.
4. `uv run pytest experiments/exp164/test.py -q` runs focused non-simulation checks.

The accepted heterogeneous compute is `exp164-r052-compute`, analysed as r053.
Each circuit has 320 E / 80 I cells, with five E cells per pixel. Sensory inputs
have 8,192 channels, 128 per pixel. Cue and C background projections have fixed
128-source masks; local E→I fan-in is 80. Excitatory target gain CVs are 0.03 for
A/B E, zero for C E and 0.2 for I. Local E→I / I→I multipliers are 0.8 / 0.1.
Sender→C E/I nominal incoming strengths are 0.6 / 0.12 µS; C background is
20 Hz/channel, sensory base 24 and cue 8. Image contrast is 0.075 and the Gaussian
cross-kernel SD remains 0.45 pixels. Pixel-map values are mean raw cell rates,
with no spatial smoothing. Every individual E/I spike remains in the rasters.

Cued PLVs are 0.992 / 0.980; cued image correlations are 0.779 / 0.887 and
cued-minus-uncued margins 0.345 / 0.500. Mean E cycle participation is 57–66%
across circuits and cues; all measured I trains are distinct, but I remains
strongly rhythmic. The recorded exploratory gates require PLV >0.95, drift
<0.25 cycles/s, phase span <0.3 cycles, regular 30–80 Hz rhythms, image correlation
≥0.7 and correlation margin ≥0.15. This is not a biological normality test.

These settings were selected on one fixed image pair and realization; no held-out
or cross-seed robustness is established. The older homogeneous condition and all
failed heterogeneous attempts remain immutable. This condition changes population
size, encoder averaging, gain variation, local strengths, image contrast and
receiver balance; it does not isolate their individual causal contributions.

Compute retains source images, inputs, native start snapshot, bundle and spikes.
Conductances are float64 reconstructions of linear synapses, not retained native
conductance recordings. The accepted simulation took about 49 seconds. The larger
encoder uses roughly 10 GB transient memory and compressed outputs of tens of MB.
