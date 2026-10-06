# exp122 — Gamma frequency and evidence accumulation

## Proposal — 2026-10-06

Map the accuracy–speed–energy trade-off of the gamma network while varying a
modulatory parameter that raises its gamma frequency. Measure accuracy, time to
a decision, and total network spikes as a proxy for energy, across different
input signal-to-noise ratios (SNRs), which we use input encoding rate as a
proxy for here. Estimate a separate Pareto frontier at each
input SNR: identify settings where improving one outcome requires sacrificing
another. Spike count is an activity-cost proxy, not a measurement of physical
energy consumption.

For many individual examples, trace confidence against elapsed time under each
modulation setting. Check whether these trajectories align in ordinary time, or
when time is expressed as the number of gamma cycles. Alignment by cycle count
would suggest that modulation changes the pace of evidence accumulation while
preserving its progression per cycle. This is a hypothesis to test, not an
assumed property of the network.

Examine input SNR versus the slope of the confidence staircase: does clearer
input produce larger confidence gains per cycle, faster cycles, or both?
Separate step size from step frequency to distinguish better use of evidence
from simply processing it faster. Measure the resulting gamma frequency at
each modulation setting rather than assuming the parameter changes frequency
alone.

Give an artificial neural network (ANN) the Poisson input summed over the first
`t` timesteps, repeating this for increasing `t`, where `t` is the number of
elapsed simulation steps. Use the same input realisations as the gamma network
to benchmark classification from accumulated evidence. This retains all input
spike counts; calling it perfect information integration would additionally
require showing that those counts preserve all task-relevant information and
that the ANN uses it effectively.

## Cycle participation

Measure the participation fraction in each complete gamma cycle: the number of
distinct neurons that spike at least once during that cycle divided by the full
population size. Report it as a percentage, separately for excitatory and
inhibitory populations; a neuron that spikes multiple times in one cycle counts
only once. Keep the population denominator fixed across modulation settings.

Record per-cycle values for each example and input realisation, then summarize
their distribution at each modulation setting and input SNR. Compare
participation with gamma frequency, spike cost and evidence gained per cycle to
distinguish faster cycles from recruitment of more neurons. Define cycle
boundaries consistently across conditions, exclude partial boundary cycles,
and mark participation as undefined when no valid cycles can be identified.

## Executed calibration — 2026-10-06

The maintained experiment is a standalone three-stage calibration. It imports
only maintained shared helpers and supported library interfaces. It neither
imports scratch code nor reads scratch outputs. The broader proposal above is
future work: accuracy, confidence, SNR and Pareto frontiers are not measured.

1. `compute.py --source exp022-r001-compute` pins final epoch-50 weights from
   `ping__tg6__seed42`, validates their SHA-256 and settings, and authors the
   reciprocal E/I graph with `experiments/helpers/ping.py`. The network is defined with
   `snnlab.lang` and executed exclusively through the compiled graph.
2. Compute presents official MNIST test image 0 (digit 7), maximum input rate
   25 Hz, with encoding seeds 42–51. It reuses each encoding across ten
   geometrically spaced inhibitory decay constants from 3 to 30 ms. Every
   trial lasts 200 ms at 0.1 ms steps, with no burn-in and reset initial state.
   All weights are frozen. E/I sizes are 1024/256, refractory holds 1.2/0.6 ms,
   capacitances 1/0.5 nF, leak 0.05/0.1 µS, rest/reset −65 mV, threshold −50 mV,
   AMPA decay 2 ms and recurrent delay one step. Readout dynamics remain fixed.
3. `analyse.py --source <compute-run-id>` measures E Welch PSD peaks (full
   200 ms Hann window, 5–150 Hz search, 5 Hz bins, parabolic interpolation
   clamped to half a bin). The condition estimate is the peak of the mean of
   ten spectra, not the mean of their individually selected peaks. Harmonics
   are not rejected and the search interval does not define gamma.
4. Participation uses I volleys detected after Gaussian smoothing with 1 ms
   standard deviation, minimum spacing 2 ms, prominence at least max(1 count,
   20% of trace maximum). Midpoint boundaries exclude partial edge cycles;
   acceptance requires one interior cycle and mean I participation ≥50%.
   E participation counts distinct spiking neurons divided by all 1024 cells,
   averaged within draws and then equally across draws. Invalid values are null.
5. Total E spikes/s counts all E spikes divided by 0.2 s, excluding input/I/output.
   Rhythmicity is `snnlab.analysis.rhythmicity_scalars` contrast from the full E raster,
   1 ms bins and 100 ms maximum lag. Volley interval CV is a separate diagnostic.
6. `present.py --source <analyse-run-id>` renders one compound figure with the
   four aggregate-only metrics on the left and four E/I rasters on the right using the shared
   theme, print sizing, black E/red I and flat PNG/SVG/PDF exports. Presentation
   does not simulate, estimate metrics or publish.

## Commands

Run from the repository root; each stage prints its own immutable v4 run ID.
Supply that exact ID to the next stage rather than choosing the newest run.

```sh
uv run python experiments/exp122/compute.py --source exp022-r001-compute
uv run python experiments/exp122/analyse.py --source <compute-run-id>
uv run python experiments/exp122/present.py --source <analyse-run-id>
```

## Maintained execution — 2026-10-06

1. Compute: `exp122-r037-compute`; all three sweeps use graph execution only.
2. Analysis: `exp122-r038-analyse`; SNNLab estimators, all 300 draws accepted.
3. Presentation: `exp122-r039-present`; aggregate-only compound figures for decay, leak and capacitance.

With ten encodings, the decay-sweep mean-PSD peak broadly declined from 104.9
to 15.4 Hz; mean total E rate fell from 21,834 to 4,320 spikes/s. Mean E
participation remained near one fifth of the population. These measurements
concern one frozen network and one image with ten stochastic encodings.
Most 30 ms trials contain only one complete interior cycle. Largest-peak
selection can switch to harmonics, and contrast alone does not establish periodicity.

Exploratory runs remain immutable historical evidence. These maintained runs
were computed anew rather than imported from scratch. Run metadata provides
full checkpoint, input, code and helper hashes and explicit stage lineage.

## Graph-only execution — 2026-10-06

Removed the comparison-model implementation, extra 6 ms recording and comparison
metadata. Recomputed the full sweep using only the `snnlab.lang` definition and
graph executor. All retained measurements matched the preceding execution
exactly. Earlier completed runs remain immutable historical evidence.

## Pruning — 2026-10-06

Removed 22 superseded exp122 runs through the hash-bound Pingstore prune command.
Only `exp122-r023-compute`, `exp122-r024-analyse` and `exp122-r025-present`
remain for this experiment: the latest presentation and its required ancestry.
Runs belonging to other experiments were outside the pruning scope.

## Combined presentation — 2026-10-06

Rendered `exp122-r026-present` from the existing `exp122-r024-analyse`, without
new simulation or measurements. Panels A–D are metrics on the left; E–H are
rasters on the right. The article now contains one figure and one caption.

Pruned the superseded `exp122-r025-present` using the exact dry-run plan hash,
retaining only the latest presentation and its compute/analyse ancestry.

## Leak calibration — 2026-10-06

Added a second independent ten-setting sweep: a geometrically spaced common
E/I leak multiplier from 0.5 to 2, at fixed 6 ms inhibitory decay. E/I default
conductances are 0.05/0.10 µS; capacitance is fixed, so the graph membrane time
constants also follow C/gL. All other neuron settings and weights are unchanged.
The exact same image and three encoded spike trains are reused in both sweeps.
Compute executes 60 presentations; analysis uses identical estimators for both.
The tau sweep reproduces its previous measurements exactly. The two lowest
leak settings select mean-PSD peaks near 131 Hz, whereas most settings select
near 60 Hz: largest-peak selection does not establish a fundamental frequency.
The presentation includes `calibration_compound` (Figure 1) and `leak_compound`
(Figure 2), each in PNG/SVG/PDF, plus scientific numbers for both sweeps.

After validating the two-figure presentation, pruned four superseded exp122
runs through the hash-bound prune command. The retained execution chain is
`exp122-r027-compute` → `exp122-r028-analyse` → `exp122-r030-present`.

## SNNLab analysis — 2026-10-06

Moved Welch PSD calculation and parabolic peak estimation to
`snnlab.analysis.power_spectrum` and `spectral_peak`. Explicit settings retain
full-trial Hann windows, temporal centering, no additional detrending, linear
power interpolation using neighbouring spectrum bins, half-bin displacement
clamping and no final search-band clamp. Condition frequency remains the peak
of the mean PSD, rather than the mean of individual peak locations.

Binning, rhythmicity, midpoint boundaries, cycle spike counts and firing rates
also use SNNLab analysis functions. Partial edge cycles are explicitly removed.
The I-volley Gaussian smoothing and prominence detector remain local because
SNNLab's burst detector uses different boundary smoothing. The demeaned,
zero-lag-normalized diagnostic correlation also retains its distinct definition.
All 60 measurements agree with the previous analysis within 1e-12, and every
plotted summary is exactly identical. Reused the existing compute evidence;
no new simulation was required.

## Ten encoding seeds — 2026-10-06

Expanded both sweeps to encoding seeds 42–51, for 200 presentations. The
original three input spike trains and E/I/output rasters are exactly unchanged.
Metrics retain the same SNNLab estimators. Presentation draws only the peak of
the mean PSD and the mean draw-level participation, E spike rate and contrast;
individual seed lines and their legend are removed. Illustrative rasters still
use seed 42. The leak sweep's mean-PSD peaks now range approximately 50–61 Hz,
illustrating the sensitivity of largest-peak selection to encoding aggregation.

Pruned four superseded runs after validating the ten-seed presentation. The
retained chain is `exp122-r033-compute` → `exp122-r034-analyse` →
`exp122-r036-present`.

## Capacitance calibration — 2026-10-06

Added ten geometrically spaced common E/I capacitance multipliers from 0.5 to
2.0, at fixed 6 ms inhibitory decay and default E/I leak conductances
0.05/0.10 µS. Default E/I capacitances are 1.0/0.5 nF; membrane time constants
follow C/gL. Other neuron settings, synapses, weights and input encoding remain
fixed. All three sweeps reuse seeds 42–51 and 200 ms presentations without
burn-in. The first two sweeps' measurements are exactly unchanged.

All 300 presentations have accepted cycle measurements. The added sweep has
mean-PSD peaks around 52–61 Hz; mean E participation rises from 18.1% to 26.0%
and mean total E rate from 11,710.5 to 15,903.5 spikes/s across the endpoints.
Figure 3, `capacitance_compound`, uses the existing house-style layout,
aggregate-only curves, illustrative seed-42 rasters and PNG/SVG/PDF exports.
Its Results subsection is “Membrane capacitance sweep”; Methods stays brief.

Pruned the three superseded exp122 runs after validating the added figure.
Retained chain: `exp122-r037-compute` → `exp122-r038-analyse` →
`exp122-r039-present`.

## House-style presentation — 2026-10-06

Selected `exp122-r040-present`, reusing `exp122-r038-analyse` and its compute
ancestor. All three compound figures use the shared paper typography and
canonical panel labels, black primary metric curves, and black E/red I rasters.
Scientific measurements and aggregation are unchanged; captions match the
revised curve colour.
