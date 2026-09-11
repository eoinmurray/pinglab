# EXP099: independent afferent excitation

This is a modified replication of the PING-network experiment in [Susin and Destexhe (2021)](https://doi.org/10.1371/journal.pcbi.1009416).

## Scientific configuration

The network has 1,600 excitatory and 400 inhibitory conductance-based LIF cells,
independent Bernoulli recurrent connections with probability 0.10, fixed
excitatory edge weights of 0.001 µS and inhibitory weights of 0.00334 µS.
Possible self-connections are included. AMPA/GABA decay times are 1.5/7.5 ms;
every pathway has a 1.5 ms delay. The timestep is 0.1 ms, preserving 3 ms E and
1.5 ms I refractory durations (30 and 15 steps).

Both populations use the paper's passive scale: capacitance 0.15 nF and leak
0.01 µS, giving a 15 ms passive membrane time constant. Both populations start at -65 mV, reset to
-65 mV and have threshold -50 mV. There is no adaptation or training.

Each neuron receives 400 independent Poisson afferents with 0.004 µS AMPA
weights. Their superposition is generated as a separate Poisson count stream
for every target neuron. Counts above one are preserved: these are not binary
Bernoulli samples. The population streams use separate child seeds. A diagonal
projection maps each aggregate stream to its own target; there is no shared
source and no separate external GABA or conductance-background process.

Rates are per individual afferent. The current protocol runs seed 7 for 1,100 ms, with a hidden 500 ms baseline before the visible 600 ms window,
matching the reference Figure 4 display duration and pulse shape: 200 ms baseline,
50 ms rise, 100 ms plateau, 50 ms fall, and 200 ms recovery. Both E- and I-targeted
rates follow 0.8 → 1.2 → 0.8 Hz. Initial voltage remains −65 mV. The video shows
the final 600 ms, rebased to 0–600 ms, in 625 frames (25 seconds); the raster uses the same interval and a
250–350 ms close-up. Earlier calibration and diagnostic runs remain documented
below as history.

## Independent stages

```sh
uv run python experiments/exp099/compute.py
uv run python experiments/exp099/analyse.py --source <compute-run-id>
uv run python experiments/exp099/present.py --source <analyse-run-id>
```

Each command completes one v4 run. An optional `--run-id` must name a previously
reserved, unused identity. Analyse and present validate explicit input digests;
neither launches upstream work. Publication is a separate operation.

Compute uses `snnsim.GraphExecutor` in chunks, carrying voltage, conductance,
refractory and delay history across boundaries. It records all E/I spikes and
private input counts at every timestep, plus population mean voltages and mean
conductances for every pathway. Individual voltage traces are not recorded.

The graph executor normalizes initializer values by source population size and
compensates sparsity. The recurrent initializer explicitly compensates these
operations to obtain the specified physical nonzero edge weights. The authored
bundle contains zero placeholders for external projections; compute binds the
actual diagonal matrices before execution. The full physical `[source, target]`
matrices in `weights.npz`, together with the bundle, are the executed model.
Do not execute the placeholder bundle alone and claim it reproduces this run.
The diagram describes the bound physical model, not placeholder initializers.

Analysis retains causal 20 ms population-rate traces, mean conductances onto E,
mean voltages, prescribed source rates, and baseline/plateau/recovery firing
rates. Additional diagnostics measure median per-cell ISI CV (at least five
spikes), mean pair correlation from 10 ms counts in at most 100 sampled cells,
SNNSIM autocorrelation contrast, and the population spectrum. These are
descriptive checks, not a PING classifier. Silence is reported explicitly.
Presentation consumes these measurements without recomputing estimators.
The single presentation command now emits the video, network diagram, full-population
spike raster, reference-paper crop and `numbers.json`; the former raster and
reference augmentation commands have been consolidated into this stage.

## Shared tools

SNNLang authors and validates the graph and lowers it to its structural diagram.
SNNSim GraphExecutor runs the network and its rhythmicity helper supplies the
autocorrelation measurement. SNNViz owns the Recording contract, FigureGrid
composition and nested panels, grid_layout cell positions, FrameTimeline,
animation encoding and Graphviz diagram rendering. Matplotlib remains only
the experiment-specific composition layer for pistons, traces and activity
marks, as permitted by SNNViz's composition boundary. NumPy generates exact
Poisson counts because the existing SNNSim Poisson binding is Bernoulli-based.

## Visual conventions

The video retains the previous white background, monospace titles, black E/red I
palette, six-panel composition and piston means. Titles sit outside panels;
mean labels sit inside above their pistons. D and E show the entire visible
interval with ten integer time ticks, revealing only elapsed data. F labels
sit inside the panel and its bottom axis is µS. Constant weights are shown as
point masses, without invented distribution widths.

Panel A shows 400 of the 1,600 E cells and 100 of the 400 I cells, evenly sampled,
and at most 100 actual recurrent edges per pathway. Input dots represent
aggregate target-specific streams, not all individual afferents. Input
projections are excitatory even when their target label is red. Edge flashes
use delayed arrivals; dots use recent events for visibility. Panel C shows a
40 ms conductance trail. The 25 s video plays at 25 fps, with approximately 0.96 ms of source time per frame.

Validation:

```sh
uv run pytest experiments/exp099/test.py -q
```

Tests check physical edge weights, refractory durations, count multiplicity and
independence, the exact 15-step arrival of a two-event pulse, and equality of
continuous versus chunked execution across a delay boundary.


## Calibration history (2026-09-08)

Susin and Destexhe Table 1 specifies 150 pF and 10 nS for RS and FS cells.
Their AdEx thresholds, adaptation and 5 ms refractory periods differ from this
LIF approximation. The passive scale is borrowed; the calibrated network is
not a paper replication or a validated model of cortex.

The search rejected silence, refractory-ceiling firing, and strongly correlated
baseline bursts. Source seed 7 was held fixed while parameters were changed.
All trials retain their complete compute/analysis evidence; none was overwritten.
Rows below are successive exploratory configurations, not independent replicates.

| Analyse run | E / I weights (nS) | Baseline source Hz | Baseline E / I Hz | E pair correlation (10 ms) |
|---|---:|---:|---:|---:|
| exp099-r002-analyse | 12.5 / 8.35 | 2 | 0.00 / 0.00 | undefined (silent) |
| exp099-r006-analyse | 12.5 / 8.35 | 2 | 323.10 / 624.12 | 0.125 |
| exp099-r008-analyse | 5 / 3.34 | 2 | 319.73 / 577.99 | 0.308 |
| exp099-r010-analyse | 5 / 8.35 | 2 | 36.45 / 59.74 | 0.849 |
| exp099-r013-analyse | 1.25 / 3.34 | 2 | 35.22 / 44.81 | 0.762 |
| exp099-r014-analyse | 1.25 / 2.0875 | 2 | 44.65 / 64.14 | 0.798 |
| exp099-r017-analyse | 1.25 / 3.34 | 0.8 | 10.47 / 11.52 | 0.369 |
| exp099-r018-analyse | 1.25 / 3.34 | 1 | 21.61 / 25.91 | 0.646 |
| exp099-r020-analyse | 1.25 / 3.34 | 0.6 | 2.07 / 2.11 | 0.018 |

The original silent trial used the previous larger membrane parameters. All
subsequent rows use the 150 pF / 10 nS scale. Short screening protocols had one
second of visible baseline and recovery; the final checks extend both to three
seconds to improve irregularity estimates and expose intermittent bursts.


### Historical longer checks of the selected configuration

The current single-seed protocol (2026-09-09) reports only seed 7. The earlier
seed-8 check remains here as calibration history and is not part of the current
article's results. Compute already executes exactly one seed per invocation;
the default command above uses seed 7, with no seed sweep or additional run.

| Seed | Compute / analyse | Baseline E / I Hz | Plateau E / I Hz | Recovery E / I Hz | Baseline E median ISI CV |
|---|---|---:|---:|---:|---:|
| 7 | exp099-r021-compute / exp099-r023-analyse | 3.09 / 3.22 | 23.56 / 30.01 | 2.53 / 2.53 | 0.82 |
| 8 | exp099-r022-compute / exp099-r024-analyse | 2.79 / 3.04 | 23.49 / 30.53 | 2.70 / 2.85 | 0.81 |

The longer baseline exposes intermittent population bursts: mean E pair
correlations are 0.178 and 0.159, not the 0.018 found in the earlier one-second
screen. This is a low-rate, individually irregular working configuration, not
a clean asynchronous-irregular regime. Plateau correlations rise above 0.81.
The two-seed check supports reproducible spiking and input responsiveness, not
biological validation, robustness across a parameter neighbourhood, or a causal
PING-mechanism claim. No simulation was selected for its best-looking frame.


### Transition-only runtime (2026-09-09)

The default compute now uses 15,000 steps instead of 100,000. Default presentation
covers 0–1,500 ms in 625 frames. Analysis omits empty recovery epochs; its baseline
includes initialization. The retained earlier 10-second runs remain unchanged.

`exp099-r028-compute` completed simulation and recording assembly in 8.141 s;
the complete compute command took 9.901 s including startup and export. Runtime
is recorded in the run's execution metadata. The previous seed-7 compute had
87 s between its recorded start and completion timestamps (a historical timing,
not a controlled benchmark). Analysis `exp099-r029-analyse` measured baseline
E/I rates of 2.54875/2.655 Hz and plateau rates of 25.345/32.55 Hz.

### Both-population drive comparison (2026-09-09)

The default now ramps independent excitatory afferents onto both populations
from 0.6 to 0.9 Hz. All other settings, including seed 7 and the 1.5 s duration,
match the preceding E-only trial. Retained configurations without an I stimulus
rate retain their fixed I-input schedule when analysed or rendered.

Compute `exp099-r032-compute` took 7.851 s for simulation; analysis
`exp099-r033-analyse` measured plateau E/I rates of 14.01625/15.62 Hz and
10 ms pair correlations of approximately 0.481/0.465, compared with
25.345/32.55 Hz and 0.814/0.890 in the preceding E-only trial.

### Reference-duration pulse (2026-09-09)

Compute `exp099-r047-compute` used 6,000 steps and took 3.012 s for simulation.
Analysis `exp099-r048-analyse` measured baseline E/I rates 1.60625/1.825 Hz,
plateau 10.175/11.925 Hz, and recovery 2.3375/2.2875 Hz. The shortened window
includes initialization; the earlier 280 ms pre-ramp burst note describes the
previous 1.5 s protocol, not this pulse trial.

### Hidden baseline (2026-09-09)

Compute `exp099-r053-compute` adds 500 ms of baseline before the visible pulse
protocol, without resetting voltages, conductances, delayed events or random streams
at the display boundary. Absolute simulation times are 0–1,100 ms; the visible
window is 500–1,100 ms and is labelled 0–600 ms in both video and raster.
Analysis excludes burn-in from the visible baseline and stimulus measurements.
Simulation took 6.793 s. Analysis `exp099-r054-analyse` measured visible baseline
E/I rates 2.503125/2.575 Hz, plateau 10/10.85 Hz and recovery 6.034375/6.8125 Hz.

### Selected c20 configuration (2026-09-09)

The author selected c20 from the completed 27-combination exploratory search in
`.scratch/exp099-lif-search-20260909/`. The grid combined baseline afferent rates
{0.4, 0.6, 0.8} Hz, recurrent E weights {1.0, 1.25, 1.5} nS and recurrent I
weights {2.7, 3.34, 4.0} nS. All cases shared seed 7 and recurrent connectivity;
cases with equal input rates shared afferent counts. Selection considered rasters,
firing rates, baseline correlations and 10 ms burst-window participation jointly.

The new defaults are 0.8 Hz baseline (1.2 Hz pulse), 1.0 nS E and 3.34 nS I.
The committed recipe fixes the baseline rate at 0.8 Hz, recurrent scale at
0.08, and inhibitory scale at 5.0; both scale factors
contribute to the inhibitory physical weight.

Fresh compute `exp099-r059-compute` took 6.092 s. Every recorded field matches
the retained scratch c20 recording bit for bit, including afferent counts, E/I
spikes, mean voltages and conductances. The scratch search record and verification
retain the selected case, payload hashes and the new v4 source reference.
Analysis `exp099-r060-analyse` measured visible baseline E/I rates
3.528125/3.825 Hz, plateau 11.50625/11.925 Hz and recovery 4.275/4.375 Hz.

Search measurements remain in the scratch search's full results and raw records.
The seven Methods steps describe that calibration and selection alongside the
selected trajectory. Earlier no-I-input timing diagnostics used the preceding
calibrated parameters and remain explicitly historical in the article.
