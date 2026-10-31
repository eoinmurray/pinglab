# exp162 — Spiking competitive entrainment

One spiking A/B → C cue-switch trial. Equal sensory drives and symmetric
sender→C projections stay fixed while a constant-rate cue spike train changes its
delivery target at 1,200 ms. Default settings: sensory 24 Hz/channel, cue
6 Hz/channel, C background 28 Hz/channel, sender→C incoming weight 0.12 µS per
pathway. This is a homogeneous-network prototype with perfectly synchronous
within-population volleys, not a complete test of preferential transmission.

1. `recipe.py` owns resolved settings, input generation, native graph and cross
   execution scaling. Cell and synaptic settings come from an explicit saved
   mean-field reference; no mean-field dynamics run here.
2. `compute.py --parameter-source COMPUTE_ID` runs one native conductance-LIF
   simulation on CPU. The reference is `exp161-r002-compute`. Optional tuning:
   `--coupling-us`, `--cue-hz`, `--background-hz`. Optional migration checks:
   `--exploration-source PHASE_ID`, `--preserve-exploration DIRECTORY`.
3. `analyse.py --source COMPUTE_ID` authenticates the execution-scaled native
   start matrices, validates spike/recording coordinates and cue routing, and
   measures phase switching in the two predeclared epochs.
4. `present.py --source ANALYSE_ID` renders the three-panel figure, saved native
   network diagram and bound report numbers. Each stage accepts `--run-id`.
5. `uv run pytest experiments/exp162/test.py -q` runs nine focused non-simulation
   checks covering inputs, cue topology, once-only coupling scaling, phase
   discrimination, settings isolation and side-effect-free standalone imports.

```sh
uv run python experiments/exp162/compute.py --parameter-source exp161-r002-compute
uv run python experiments/exp162/analyse.py --source COMPUTE_ID
uv run python experiments/exp162/present.py --source ANALYSE_ID
```

Native matrices are restored from the reference checkpoint. Its 0.04 µS cross
weights are multiplied by the recorded execution scale of three, giving the
selected 0.12 µS incoming weight. Sensory gain is already present and is not applied
again. Each cue port receives the same sampled U train; complementary delivery
gains redirect it without resetting state. Input generation uses per-channel/bin
Bernoulli probability `rate*dt/1000`, approximating Poisson spikes. The native
0.1 ms timestep, input seeds, six population spike arrays and five native per-cell
conductance recordings are retained. Symmetric 1 ms SD Gaussian smoothing is used
only for rate plotting and peak-phase estimation; native conductances are unsmoothed.

The first compute is `exp162-r001-compute`, analysis `exp162-r002-analyse`,
presentation `exp162-r003-present`. All input spikes, all six neuronal spike arrays
and every recorded conductance sample exactly reproduce the selected scratch run
`exp161-r006-compute-phase-scratch-spiking-competition`. Cued PLVs are 0.982167 and
0.982046; uncued PLVs are 0.191683 and 0.082880. Simulation took 16.148 seconds,
analysis 0.163 seconds and presentation 0.552 seconds. Nine focused tests, Ruff,
formatting and diff checks pass. Report, figures, note link and collection entry
were checked locally.

The formal compute pins the selected scratch run and its retained failed
0.04 µS attempt. Its byte-verified `export/exploration.zip` preserves the scratch
source, tests, README, settings, iteration history and recorded validation notes.
The scratch directory was removed after migration verification; reproduction has
no scratch dependency. No completed Pingstore runs were removed or rewritten.
Both exploratory simulations took 31.239 seconds in total. The report is in the
CTC attention collection, and plan 4 of `notes/ctc-attention/index.mdx` links to it.

The original phase gates are exploratory: cued PLV >0.95, absolute drift <0.25
cycles/s, cued phase span <0.3 cycles, uncued span >0.5 cycles and C rate amplitude
>0.5 Hz in both 650–1,100 and 1,850–2,300 ms. All circuits additionally require at
least 12 volleys, 30–80 Hz frequency and interval CV ≤0.2 per epoch. Low-inhibition
shading is a conductance proxy, not measured response gain. One tuned realization
cannot establish reliable switching across seeds, asymptotic locking, heterogeneous
network performance or preferential information transmission. No probes, classifier
or rate-matched controls were run.

The earlier `exp162-r024-present` placed symmetric sender columns above C. Its
scientific figures and `numbers.json` were byte-identical to `exp162-r003-present`;
the diagram-only render took 0.584 seconds without simulation. The compact layout
below supersedes that arrangement.

### Compact circuit diagram

The diagram follows exp160's mathematical labels, compact E/I pairs and recurrent
loops, with mirrored A/B pathways converging on C in the centre. Redraw only the
network from a validated presentation, preserving its result figures and numbers:

```bash
uv run python experiments/exp162/present.py --redraw-from PRESENT_ID
```
