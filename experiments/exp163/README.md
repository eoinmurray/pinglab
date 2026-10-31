# exp163 — Spatial pattern selection

One spiking A/B → C trial with fixed Gaussian sensory-drive fields and topographic
sender excitation. A separate constant-rate cue targets A E before 1,200 ms and B E
afterward. The five-panel figure retains exp162's rates, phases and conductances,
then adds measured A/B/C spatial output profiles before and after the switch.
Default settings reproduce the selected 8 Hz/channel cue case: sensory
24 Hz/channel, C background 22 Hz/channel, incoming cross weight 0.6 µS per pathway,
Gaussian centres 0.25 / 0.75, width 0.1, spatial contrast 0.4, kernel width 0.06.

1. `recipe.py` owns resolved settings, native graph, constant-rate spike input
   generation, spatial coordinates and explicit checkpoint transformations.
2. `compute.py --parameter-source COMPUTE_ID` runs one native conductance-LIF
   simulation on CPU. The parameter reference is `exp162-r001-compute`. Optional
   overrides: `--coupling-us`, `--cue-hz`, `--background-hz`, `--contrast`, `--width`,
   `--kernel`, `--swap`. Migration options: `--exploration-source PHASE_ID`,
   `--mirror-source PHASE_ID`, `--preserve-exploration DIRECTORY`.
3. `analyse.py --source COMPUTE_ID` checks actual native start arrays against the
   prescribed transforms, validates recordings and spike coordinates, and measures
   phase locking and spatial contrast. If supplied, the retained mirror control
   is authenticated and checked against the new trajectory without rerunning it.
4. `present.py --source ANALYSE_ID` produces the five-panel figure, native network
   diagram and report numbers. Each stage accepts `--run-id`.
5. `uv run pytest experiments/exp163/test.py -q` runs seven non-simulation checks
   covering preserved thresholds/defaults, spatial matrix normalization, reflection
   symmetry/input invariance, rejecting fixed-side spatial selection, native
   pathway/cue topology, phase discrimination and standalone imports.

```sh
uv run python experiments/exp163/compute.py --parameter-source exp162-r001-compute
uv run python experiments/exp163/analyse.py --source COMPUTE_ID
uv run python experiments/exp163/present.py --source ANALYSE_ID
```

The formal compute is `exp163-r001-compute`, analysis `exp163-r002-analyse`,
presentation `exp163-r003-present`. Every input spike, all six neuron spike arrays
and every recorded conductance sample exactly reproduce
`exp162-r021-compute-phase-scratch-spatial-competition`. The separately retained
`exp162-r022-compute-phase-scratch-spatial-competition` centre-swap control mirrors
all E spikes and measured spatial profiles exactly; complete input trains and
I-cell spikes remain identical. These are explicit scientific inputs, not code
imports or dependencies on scratch files.

Sensory weights are multiplied by a Gaussian-plus-pedestal target gain normalized
to mean one. Sender E→C E weights use a per-recipient normalized Gaussian distance
kernel; E→C I remains pooled. All incoming cross sums match, and local recurrence
is unchanged. Native start parameter arrays are authoritative. The graph's nominal
constant initializers describe the uniform pre-transform reference. Report matrix
cells combine those bound native base constants with recorded spatial rules to
give nominal per-connection formulas, not statistics of the executed arrays.
Analysis authenticates every executed array exactly. Sensory gain is included once.

C spatial contrasts are +0.620513 and −0.728261, with cued PLVs 0.977317 and
0.973955 in the predeclared 650–1,100 and 1,850–2,300 ms windows. Spatial profiles
are raw per-cell spike counts per measured time, with no spatial smoothing or
Gaussian fitting. Population-rate plotting/phase estimation uses a symmetric
1 ms SD Gaussian; native conductances are unsmoothed. Strong synchronization and
neuron thresholds produce stepped output bumps despite Gaussian sensory fields.

Simulation took 16.323 seconds, analysis 0.590 seconds, presentation 0.758 seconds.
The selected and mirror exploration runs pin the nineteen-attempt history, including
unsuccessful receiver-only changes, earlier 6 Hz cue results and the documented
corrected pass-flag error. Those explorations took 306.143 seconds of simulation.
The byte-verified `export/exploration.zip` preserves original scratch source, tests,
README, settings and iteration history. The migrated scratch directory was removed
after successful verification; no completed scientific run was rewritten or removed.

The report is in the CTC attention collection and linked from plan 4 of
`notes/ctc-attention/index.mdx`. One input realization and its mirrored control support
cue-dependent spatial output selection, not a causal separation of gamma timing
from rate effects or robustness across seeds. No timing interventions, probes,
classifier or arbitrary-pattern decoding were added.

### Compact circuit diagram

The diagram follows exp160's mathematical labels, compact E/I pairs and recurrent
loops, with mirrored A/B pathways converging on C in the centre. Redraw only the
network from a validated presentation, preserving its result figures and numbers:

```bash
uv run python experiments/exp163/present.py --redraw-from PRESENT_ID
```
