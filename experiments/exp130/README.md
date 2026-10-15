# EXP130: theta-modulated afferent excitation

Standalone recreation of the private-afferent E/I circuit and Figures 1–3,
with 5 Hz rate modulation and a constant-input comparison. There are no imports
from exp099, no dependencies on its runs, and no reused scientific exports.
The graph, Poisson generator, simulator driver, analysis, figure rendering,
browser replay and report are owned by this directory; installed SNNLab and
shared Pingstore helpers supply their normal supported interfaces.

## Scientific protocol

1. Two fixed, matched circuits: 1,600 E / 400 I conductance LIF neurons;
   all four recurrent pathways; independent Bernoulli edges with probability
   0.10 between distinct neurons; autapses are disabled. Excitatory/inhibitory recurrent
   increments are 1 / 3.34 nS. There is no training or adaptation.
2. Each neuron receives 400 independent excitatory Poisson afferents, represented
   exactly by a target-private aggregate count stream. Each event adds 4 nS.
   Multiplicities greater than one per timestep are retained.
3. Both conditions start directly at initialization with no burn-in. Theta follows
   `2 + 1.8*sin(2*pi*5*t)` Hz per afferent, where `t` is seconds from initialization;
   constant remains at 2 Hz. E- and I-targeted envelopes coincide but their
   samples are independent. The entire one-second simulation is recorded and contains five periods.
4. Seed 8 is paired across conditions, as are connectivity, physical weights,
   initial state and random-generator streams. Compute verifies identical
   matrices exactly. No burn-in is executed. Realised afferent counts
   and neuronal output rates after modulation onset are not matched.
5. Timestep 0.1 ms; delay 1.5 ms; capacitance 150 pF; leak 10 nS; rest/reset
   and initial voltage −65 mV; threshold −50 mV; E/I refractory times 3/1.5 ms;
   AMPA/GABA decay 1.5/7.5 ms.
6. Analysis uses 200–800 ms, three complete periods, excluding 200 ms at each recording
   edge from measurements. The complete simulation, including startup, is displayed. It retains 1 ms population-rate
   spectra, 30–90 Hz Hilbert amplitude and 18-bin phase profiles. Welch uses
   500 ms Hann segments with 50% overlap; peaks are raw bin maxima, 2 Hz spacing.
   Phase concentration is descriptive; no surrogate test or mechanistic
   isolation of gamma is claimed. Constant input uses the same reference clock.

## Independent stages

```sh
.venv/bin/python experiments/exp130/compute.py
.venv/bin/python experiments/exp130/analyse.py --source <compute-id>
MPLCONFIGDIR=/private/tmp/exp130-matplotlib XDG_CACHE_HOME=/private/tmp/exp130-cache .venv/bin/python experiments/exp130/present.py --source <analyse-id>
```

Each command creates one v4 run. Compute executes both conditions in one run,
without launching downstream stages. Analyse and present validate explicit
sources and payload digests. Bundle directories use canonical `SourceRun.unit`
resolution. Actual execution matrices, including private diagonal projections,
are retained alongside raw recordings. The graph bundle alone has zero external
initializer placeholders; replay/reproduction requires the retained matrices.

## Figures

1. A fresh expanded structural diagram from the compiled SNNLab graph.
2. A selectable theta/constant PixiJS replay from fresh recorded events,
   sampled real edges and full-population means. Both conditions share piston
   scales. The full 1 s protocol plays over 25 s at 1×; pause, seek and speed
   controls affect display only. Events preserve integer simulation steps and
   input multiplicities. The canonical `circuit.json` payload contains both
   conditions and is cached once per browser module, avoiding extra asset routes.
3. Paired prescribed rates, complete 2,000-neuron rasters and unsmoothed
   1 ms population count profiles. Every panel spans the full 0–1,000 ms
   simulation, with common axes across conditions. The former
   C/G close-ups are removed; A/E, B/F and D/H retain their labels.

## Current protocol without autapses, 15 October 2026

The report selects `exp130-r026-present`, derived from `exp130-r025-analyse`
and `exp130-r024-compute`. Individual self-connections are explicitly disabled
on all recurrent projections. E→E and I→I recurrence between distinct cells is
retained. The population-level diagram therefore still has loops; its caption
now explains that they do not represent individual self-connections.

Both conditions retain seed 8, 5 Hz theta, 0.2–3.8 Hz afferent modulation,
constant 2 Hz input, exactly one second and no burn-in. Saved weights were
checked directly: zero E autapses and zero I autapses in both conditions;
255,736 E→E and 15,585 I→I off-diagonal edges remain. Against the previous
`exp130-r021-compute`, all off-diagonal masks agree exactly, surviving physical
weights agree within floating-point tolerance, cross-population/external
matrices match exactly, and both private input arrays match exactly.

Disabling autapses changes the library's eligible-source normalization to N−1.
The initializer compensates for that denominator so removing self-edges does
not increase the remaining edge strengths. Scientific tests check this and
zero diagonals in the executed matrices. In the 200–800 ms analysis window,
theta/control E rates are 16.756/15.822 Hz; I rates are 19.667/18.288 Hz.
The 30–90 Hz maxima are 44 Hz for theta and 40 Hz for control; E phase
concentration is 0.526/0.077 and I is 0.527/0.079. These short-recording
measurements remain descriptive.

## Historical illustrative five-Hz protocol with autapses, 15 October 2026

The earlier report selected `exp130-r023-present`, derived from `exp130-r022-analyse`
and `exp130-r021-compute`. This is exactly 1,000 ms total simulation, with
zero burn-in, no hidden prelude and initial voltages −65 mV. Theta begins at
the rising mean crossing at time zero: `2 + 1.8*sin(2*pi*5*t)` Hz per afferent,
range 0.2–3.8 Hz. Constant input remains 2 Hz. Seed 8, connectivity, physical
weights and all neuron/synaptic settings are retained. Both figures and replay
show the complete 0–1,000 ms simulation, including its initial transient.

Analysis uses only 200–800 ms to exclude startup/filter endpoints from the
measurements: three complete theta cycles. This does not simulate or hide a
burn-in. The analysis-window E rates are 16.419/16.731 Hz (theta/constant),
I rates 19.346/19.317 Hz. E phase concentration is 0.541/0.027 and I is
0.542/0.036. The 30–90 Hz peak is 40 Hz for theta and 36 Hz for constant,
with 2 Hz bin spacing. These are descriptive observations from one short draw,
not a physiological calibration or surrogate-tested mechanistic inference.

## Historical ten-Hz-mean input, 15 October 2026

The earlier strong-drive report selected `exp130-r020-present`, derived from `exp130-r019-analyse`
and `exp130-r018-compute`. Both conditions now have mean 10 Hz per afferent;
theta follows `10 + 3*sin(2*pi*8*t)` Hz (7–13 Hz, ±30%) and constant stays at
10 Hz. The theta envelope is retained at 8 Hz. Burn-in is 500 ms at 10 Hz,
followed by the same one-second recording, six-panel Figure 3 and seed 8.
Graph connectivity and synaptic weights are unchanged; all paired checks pass.

Mean E/I firing rates are 58.764/78.127 Hz with theta and 60.087/79.467 Hz
with constant input. Both control populations have a dominant spectral peak
at 66 Hz; a direct check of the retained spectrum over 10–300 Hz confirms
this peak, with harmonics near 132 and 200 Hz. The reported 30–90 Hz maximum
is also 66 Hz in theta. Amplitude-weighted phase concentration is 0.091/0.024
(theta/constant) for E and 0.106/0.021 for I. These remain single-seed,
short-recording descriptive measurements. The input-rate change accelerates
the collective rhythm and substantially increases individual firing rates.
Earlier completed runs are preserved unchanged.

## Historical one-Hz-mean seed-8 evidence, 15 October 2026

The earlier low-input report selected `exp130-r017-present`, derived from `exp130-r016-analyse`
and `exp130-r015-compute`. This reruns both complete conditions using seed 8
for graph initialization and the private Poisson inputs. Initial voltages and
all non-seed scientific settings are unchanged; E/I populations and both
conditions retain their paired weights and burn-in checks. The previous seed-7
recordings remain available as immutable evidence.

A direct descriptive comparison of the constant recordings in
`exp130-r012-compute` (seed 7) and `exp130-r015-compute` (seed 8) used the
complete visible second in ten nonoverlapping 100 ms bins. Per-neuron E output
rates ranged from 5.3625 to 9.8625 Hz for seed 7, peaking at 400–500 ms;
seed 8 ranged from 4.49375 to 10.2625 Hz, peaking at 800–900 ms. I output ranges
were 5.775–9.8 and 5.7–11.05 Hz, respectively, with the same peak windows.
Every prescribed afferent rate stays at 1 Hz in the constant condition;
realised E-targeted rates across the bins stayed within 0.9945–1.0061 Hz.
These two draws show fluctuations at different times, not a repeated timed
intervention. They do not establish a multi-seed population result or isolate
input randomness from changed connectivity, since the whole seed changed.

## Historical one-second seed-7 evidence, 15 October 2026

The earlier seed-7 report selected `exp130-r014-present`, derived from `exp130-r013-analyse`
and `exp130-r012-compute`. Each condition executes a separate 500 ms burn-in
followed by exactly 1,000 ms of recorded continuation (1,500 ms total). The
8 Hz / 0.5–1.5 Hz afferent envelope and constant 1 Hz comparison are unchanged.
Figures 2 and 3 now show this one-second recording, containing eight cycles.
Figure 3 removes C/G; all six remaining panels share 0–1,000 ms limits,
including the unsmoothed 1 ms population counts. No close-up is retained.

Analysis uses absolute 625–1,375 ms, six complete theta cycles, leaving 125 ms
at each end of the visible recording for onset/filter-edge exclusion. E mean
rates are 8.352 / 7.522 Hz (theta / constant); I rates are 8.453 / 7.617 Hz.
E phase concentration is 0.396 / 0.127 and I is 0.404 / 0.113. These short,
single-seed measurements remain descriptive. Old completed runs are unchanged.
The Methods table, captions, replay and parameter footer reflect the shorter
recording under Writing Guide 36.0.0.

## Historical three-second 8 Hz evidence, 15 October 2026

The earlier three-second report selected `exp130-r011-present`, derived from `exp130-r010-analyse`
and `exp130-r009-compute`. The recipe now uses 8 Hz and 0.5–1.5 Hz per
source; the comparison is still constant 1 Hz. The prior 6 Hz evidence below
remains immutable. Every constant-control weight and recorded signal matches
`exp130-r005-compute` exactly. Both new conditions also passed the internal
weight and burn-in equality checks.

E mean rates are 7.657 / 6.911 Hz (theta / constant); I rates are 7.930 / 7.063 Hz.
Amplitude-weighted phase concentration is 0.427 / 0.070 for E and 0.425 / 0.075
for I, referenced to the new 8 Hz clock. This changed clock explains why the
constant condition's phase concentration differs from its previous 6 Hz value
despite identical recorded activity. Theta search-band maxima are 32 Hz for
both populations; constant E/I maxima are 30/32 Hz. A 32 Hz component is also
the fourth harmonic of 8 Hz, so these measurements alone do not establish an
independent gamma oscillator. The three-second recording contains 24 cycles;
the fixed 250 ms raster close-up contains two. Methods retain the requested
parameter-table format; labels, captions, equation and parameter footer agree.

## Historical 6 Hz evidence, 15 October 2026

The earlier report selected `exp130-r008-present`, derived from
`exp130-r006-analyse` and `exp130-r005-compute`. Both conditions use freshly
simulated activity. E mean rates are 6.538 / 6.911 Hz (theta / constant);
I rates are 6.778 / 7.063 Hz. E amplitude-weighted phase concentration is
0.287 / 0.098; I is 0.259 / 0.084. Search-band maxima are 30/30 Hz for E
and 30/32 Hz for I. Boundary maxima do not demonstrate an isolated gamma peak.
These are single-seed observations, not population-level or significance claims.

The first paired run (`exp130-r001-compute`) tested the installed library's
implicit no-autapse default; `r002-analyse` and `r004-present` describe it.
Inspection of the compiled graph showed that reproducing the documented
possible self-connections required `allow_autapses=True`, now explicit.
The scientific rerun is `r005-compute`. `r007-present` was an intermediate
render with separate JSON assets; `r008-present` uses the supported canonical
replay asset route. Completed evidence remains immutable. The failed `r003`
presentation used an incorrect pre-canonicalization bundle path and is not a
scientific input; its task-owned temporary directory was cleaned up after the
replacement succeeded.

Writing Guide 36.0.0 governs the new report, captions and parameter/reference
footers. The report is not author-reviewed. The router's repository-guide.md and
experiment-runner-guide.md paths are absent in this checkout; the current
experiment-rules.md, writing-rules.md and pingstore-rules.md were followed.

## Verification

1. Four scientific integration checks passed: envelope bounds/mean/phase,
   Poisson multiplicity and channel independence, chunked versus uninterrupted
   execution, and physical weights plus explicit self-connection declarations.
2. Ruff checks and formatting passed for this experiment. No runtime Python,
   JavaScript or Astro import points into exp099.
3. Pingstore discovery succeeded and the selected compute, analysis and
   presentation payload digests were independently revalidated.
4. The complete Astro build succeeded: 43 pages, including the new report and
   its three asset routes. The report explicitly binds its numbers and figures
   to the same selected presentation.
5. The final raster was visually inspected. Browser checks verified both replay
   conditions, one canvas after switching, pause, seeking and playback speed;
   no console warnings or errors occurred. Both replay states were visually
   inspected. Local preview for the original render: `http://localhost:3002/exp130/`.
   The refreshed 8 Hz preview is `http://localhost:3003/exp130/`.
6. No remote execution, publication, commit or push was performed. Other
   experiments and shared site source files were not edited by this task.

### 8 Hz verification

1. All four scientific checks passed with the new envelope; lint and whitespace
   checks passed. Selected v4 inputs and payload digests validated, and discovery
   confirmed the new presentation.
2. The full site build succeeded (43 pages). The refreshed raster was visually
   inspected; browser checks confirmed the new label and selected run, theta
   input extrema of approximately 0.500002/1.499998 Hz at the 0.1 ms sampling
   grid, constant 1 Hz input, pause/seek and condition switching with one canvas.
   The continuous prescribed envelope has exact extrema 0.5/1.5 Hz.
3. The initial preview reported a stale Vite dependency response for the shared
   command palette; restarting the task-owned preview cleared it, and the fresh
   page had no console warnings or errors. The Methods table, equation,
   captions, footer and replay selector now specify the 8 Hz protocol under
   Writing Guide 36.0.0. Previous completed runs were preserved.

### One-second verification

1. All four scientific checks passed, along with lint and whitespace checks.
   The new compute, analysis and presentation payloads were validated; both
   replay payloads contain exactly 1,000 ms of visible activity.
2. The full 43-page site build passed. Figure 3 was visually checked: all six
   axes span 0–1,000 ms, no close-up or shading remains, and C/G are absent.
   The live report selects the new raster and a 1,000 ms replay scrubber.
3. The existing 500 ms burn-in remains separate from the one-second recording;
   total simulated time is 1,500 ms per condition. Previous completed evidence
   was preserved. No remote execution or publication was performed.

### Seed-8 verification

1. All four scientific checks and lint passed. New paired weights and burn-in
   matched exactly within seed 8; all three selected stage payloads validated.
2. The full 43-page site build passed. The new six-panel raster was visually
   inspected and the live report's raster, replay and caption select seed 8.
3. The 100 ms fluctuation comparison above is a separate read-only comparison
   of two completed raw recordings. It does not alter either recording or
   imply that input noise and connectivity were varied independently.
4. The scientific settings and one-second visible interval were retained.
   Writing Guide 36.0.0 was applied to the seed labels and parameter tables.

### Ten-Hz-mean verification

1. All four scientific checks passed with the 10 Hz mean and 7–13 Hz envelope;
   the Poisson sampling check now uses its expected standard error at the higher
   count rate. Lint and whitespace checks passed.
2. Compute, analysis and presentation configurations and payload digests validated.
   The full site build passed (43 pages). The raster was visually inspected;
   the live report selects the new data and correctly labels both input regimes.
3. Methods and footer retain Writing Guide 36.0.0, 8 Hz theta, seed 8, a 500 ms
   burn-in and one second of visible activity. Previous completed evidence
   remains immutable. No remote execution or publication was performed.

### Five-Hz no-burn-in verification

1. All four scientific checks passed, explicitly checking 1,000 ms total,
   zero burn-in and theta onset at time zero. Lint and whitespace checks passed.
2. Raw recordings have exactly 10,000 steps per condition at 0.1 ms. Run
   execution metadata records zero burn-in steps, and both replay payloads
   start at zero and last exactly 1,000 ms. All three stage digests validated.
3. The full 43-page site build passed. Figure 3 was visually inspected and
   shows repeated packets of fast population cycles separated by quiet theta
   troughs. The live report selects the new raster and replay and labels the
   5 Hz envelope, 0.2–3.8 Hz afferents and 2 Hz constant comparison.
4. Methods, equation, captions and parameter footer agree under Writing Guide
   36.0.0. Measurements explicitly exclude 200 ms at each end while the entire
   simulation remains visible. Previous completed runs remain immutable.

### No-autapse verification

1. All four scientific tests passed, including zero executed recurrent diagonals
   and preservation of the specified nonzero physical edge strengths. Lint and
   whitespace checks passed. Both full-sized conditions were independently
   checked against the previous saved matrices and private input streams.
2. All new v4 payload digests validated and discovery succeeded. The full
   43-page site build passed. The regenerated six-panel raster was visually
   inspected and retains visible theta-organized packets.
3. The live report selects the new raster/replay. Figure 1's caption explains
   the remaining population loops. Methods and parameter footer now explicitly
   exclude autapses under Writing Guide 36.0.0; previous runs remain unchanged.

## Local pruning, 15 October 2026

At the author's request, the validated experiment-scoped, hash-bound prune
removed 22 superseded exp130 runs (approximately 0.20 GiB). Only
`exp130-r024-compute`, `exp130-r025-analyse` and `exp130-r026-present` remain
locally. Their payloads and ancestry validate, and the current report pin remains
unchanged. Historical run IDs above describe prior work, not locally available
evidence. Every out-of-scope run was retained. An initial attempt was refused
while another experiment held the writer lock; the successful retry used the
same unchanged plan after the lock became available. No remote data was deleted.
