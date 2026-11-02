# exp162 — Spiking competitive entrainment

One A/B → C trial with a constant-rate cue redirected at 1,200 ms. The selected
heterogeneous condition has 400 E / 100 I neurons per circuit, sensory and cue
fan-in 128, local E→I fan-in 80, sender→C fan-in 40, and fixed lognormal
excitatory target gain CVs 0.1 / 0.2 for E / I. Expanded reference matrices
preserve nominal incoming totals before masks, gains and the recorded E→I / I→I
multipliers 0.8 / 0.1. A/B→C I projections remain at zero execution gain.

1. `uv run python experiments/exp162/compute.py --parameter-source exp161-r002-compute`
   runs the fixed 2,400 ms CPU trial. Default sensory / cue / C background rates
   are 24 / 6 / 28 Hz/channel; nominal sender→C E strength is 0.25 µS. Optional
   `--coupling-us`, `--cue-hz`, `--background-hz` support explicit comparisons.
2. `uv run python experiments/exp162/analyse.py --source COMPUTE_ID` authenticates
   transformed native start weights, cue routing and recording coordinates, then
   measures switching with the unchanged phase/rhythm criteria and raw-spike
   recruitment/diversity diagnostics.
3. `uv run python experiments/exp162/present.py --source ANALYSE_ID` renders the
   experiment, all-neuron raster, and native network diagram. All stages accept
   `--run-id`; inputs are explicit and no stage launches another stage.
4. `uv run pytest experiments/exp162/test.py -q` runs non-simulation checks.

Selected compute `exp162-r031-compute` passes both epochs: cued PLVs 0.991 / 0.973,
uncued PLVs 0.119 / 0.178. E participation averages 59–78% per complete cycle
across the measured populations and epochs; I participation remains high, but
99–100% of I trains are distinct and maximum same-bin coincidence is 49%.
No spikes are moved, removed, smoothed or subsampled in the rasters.

The earlier r027 and r029 heterogeneous conditions failed switching. Sender and cue input trains, and sender spikes, remain identical across all three.
The r029/r031 coupling comparison also has identical C inputs; changing
background between r027 and r029 changes only X_C input bins. Every condition is immutable in Pingstore.
These are exploratory selections, not independent seed validation. The cue changes
sender rate and frequency together; low inhibition is only a conductance proxy.
No preferential-transmission, rate-matched, probe or classifier claim is made.

The original homogeneous runs and migration archives remain historical evidence.
Native execution-start arrays are authoritative; nominal constant graph
initializers precede the documented sparse and gain interventions.
