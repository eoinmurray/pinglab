# exp163 — Spatial pattern selection with partial recruitment

A/B → C spiking competition with Gaussian sensory fields and topographic sender
excitation. A constant-rate cue targets A before 1,200 ms and B afterward.
The selected main trial preserves both phase switching and spatial selection,
with incomplete E recruitment and distinct I spike trains.

1. `uv run python experiments/exp163/compute.py --parameter-source exp162-r001-compute`
   runs the selected 2,400 ms CPU trial. Defaults: 400 E / 100 I per circuit;
   sensory / cue / C background 24 / 8 / 28 Hz/channel; sensory contrast 0.08;
   Gaussian width 0.1, centres 0.25 / 0.75, cross kernel width 0.06.
2. `uv run python experiments/exp163/analyse.py --source COMPUTE_ID` authenticates
   actual native start weights and recordings, then applies the unchanged phase
   and spatial criteria. It also measures raw-spike recruitment and diversity.
3. `uv run python experiments/exp163/present.py --source ANALYSE_ID` renders the
   experiment, all-neuron rasters and native network diagram. All stages accept
   `--run-id`; scientific inputs are explicit and no upstream stage runs implicitly.
4. `uv run pytest experiments/exp163/test.py -q` runs non-simulation checks.

Fixed masks retain 128 sensory/cue sources, 80 local E→I sources and 40 pooled
cross sources per target. Expanded weights preserve nominal incoming totals
before interventions. Excitatory target gain CVs are 0.1 for A/B E, zero for C E,
and 0.2 for I; C E still has private stochastic input. Local E→I / I→I multipliers
are 0.8 / 0.1. Each sender supplies nominal 0.6 / 0.39 µS to C E / I.
Gaussian sensory fields are normalized to preserve total incoming sensory drive.
Native start snapshots retain the actual physical matrices.

Selected compute `exp163-r050-compute` reproduces the passing r044 main trial.
Cued PLVs are 0.973 / 0.982 and C spatial contrasts +0.225 / −0.235. E participation
averages 52–62% per complete cycle; 96–100% of measured I trains are distinct.
I remains strongly rhythmic. This is not a biological normality test.

## Control limitation

`compute.py --parameter-source exp162-r001-compute --swap` moves the Gaussian
centres while holding masks, gains and all non-sensory weights fixed.
`analyse.py --source MAIN_ID --mirror-source CONTROL_ID` authenticates both and
records the control outcome separately from the main trial. The retained fixed-
heterogeneity centre-swap r049 fails the task criteria in its second epoch.
The main result therefore does not establish robustness to spatial reassignment.

An earlier complete-network reflection r046, including reflection of the random
heterogeneity, mirrored spikes exactly. That checks numerical reflection symmetry;
it is not the fixed-weight stimulus control. Later bilateral-weight diagnostics
were not selected. Their native data and the earlier failed variants remain
immutable, rather than being substituted for the selected main trial.

All settings were selected on one realization. Spatial selection is weaker than
in the original homogeneous example; timing was not isolated from rate effects.
No classifier, arbitrary-pattern decoding or cross-seed robustness is claimed.
