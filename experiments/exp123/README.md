# exp123 — Output evidence against time and gamma cycles

## Question

Does changing inhibitory decay mainly alter the pace of evidence accumulation,
so that cumulative-output-spike softmax trajectories align more closely by
network cycle position than by elapsed time? Alignment is a hypothesis, not an
assumed property. Softmax scores are uncalibrated evidence summaries.

## Independent implementation

The experiment owns its recipe, graph definition, computation, measurement and
presentation code. Its only upstream scientific input is the validated final
checkpoint in the training bank, supplied explicitly to compute. Shared
maintained PING and checkpoint helpers and public SNNLab interfaces are used.
No other experiment's implementation or exploratory recordings are imported.
The checkpoint data are reused; there is no retraining and no comparison model.

## Protocol

1. Pin final epoch-50 `ping__tg6__seed42` weights from `exp022-r001-compute`,
   checking the registered checkpoint role and exact SHA-256 in `recipe.py`.
   Validate the training configuration; copy four matrices directly into graph
   projections in their stored [source, target] orientation. All weights stay frozen.
2. Define 784 input channels, 1024 E neurons, 256 I neurons and ten spiking
   output units with `snnlab.lang`, using `experiments/helpers/ping.py` for the
   reciprocal E/I circuit. E/I capacitance is 1/0.5 nF, leak 0.05/0.10 µS,
   rest/reset −65 mV, threshold −50 mV, refractory holds 1.2/0.6 ms. AMPA decay
   and readout/filter time constants are 2 ms; recurrent delay is one 0.1 ms
   step. The output threshold is one with subtractive reset; no E→E/I→I paths.
3. Use the first nine official MNIST test images, indices 0–8. For each
   image, make 100 stochastic encodings with draw identifiers 42–141; actual
   generator seeds are `1000 * image_index + draw_identifier`. Normalize
   pixels by 255 and use maximum pixel rate 25 Hz. Reuse each full encoded
   tensor at 3, 6 and 12 ms GABA decay. Simulate 200 ms without burn-in,
   resetting graph state at every presentation. This gives 2700 trajectory presentations.
4. Analyse softmax of cumulative output spike counts at fixed temperature one.
   Scores are calculated independently for each draw before averaging. Retain
   all cumulative class counts, true-class scores, maximum scores and predicted
   classes. An all-zero initial count vector gives scores 0.1; later times
   denote completed simulation updates. Lowest class index wins an argmax tie.
   The checkpoint was trained with mean-voltage evidence; the cumulative-count
   decoder here is a separate inference measurement, not a newly trained decoder.
5. Detect I volleys from population counts after Gaussian smoothing with
   sigma 1 ms, using reflected boundaries, spacing ≥2 ms and prominence
   ≥max(1 smoothed count, 20% of maximum). Require a complete interior midpoint
   cycle and mean I participation ≥50%. Use SNNLab for cycle construction and
   counting. Volley sequences do not by themselves prove periodicity.
6. Figure 2 counts completed I volleys from one, defining completion as the
   first minimum of smoothed I activity strictly between an accepted peak and
   the next peak. Exclude the final peak without a following boundary. Sample
   cumulative softmax at each completion's last completed simulation update;
   scores include all spikes since onset. Use only counts available in all 300
   presentations per image, averaging all 100 draws. Count zero shows the
   initial score 0.1. This trough is an operational completion boundary, not a
   guarantee that all associated output spikes have arrived. Retain interpolated
   cycle-position evidence separately; Figure 3 uses the same completion boundaries.
7. Render matching nine-image 3 × 3 physical-time and volley-count grids
   (indices 0–8), with one mean curve per decay setting. Retain PNG,
   SVG and PDF figures and their scientific numbers in a v4 presentation run.
   Diagnose saturation at maximum score ≥0.99, reporting onset time and final
   correctness. No temperature fitting, confidence calibration or numerical
   alignment statistic is included in this first visual comparison.
8. Measure the earliest update at which the true class becomes the unique
   largest cumulative-count winner and stays so through 200 ms. Ties interrupt
   this retrospective stability criterion; final ties or errors have no decision.
   Use the first trough completion at or after that update, numbered from one: the
   earliest Figure 2 sample at which the decision is already stable. Require accepted sequences, a decision no later
   than the final completion, and Figure 2's shared count range for count summaries.
   Figure 3 shows median points for the same nine images: time 0–200 ms on x
   above and completed volley count with Figure 2's count limits below; GABA
   decay on y. Lines connect median points; no uncertainty bars are displayed. Trial counts are
   integers, while medians can be fractional. The two rows can use different
   successful-trial subsets; failures have no decision point.

## Complete three-stage execution

Each stage creates exactly one run. Compute includes the nine-image trajectory
sample (100 draws each at 25 Hz) and the 500-image balanced benchmark (three draws
each at 12.5, 25 and 50 Hz). Both use three inhibitory-decay settings, for 16,200
presentations total. Baseline 25 Hz benchmark recordings are shared by the
matched-coordinate and Pareto measurements within the same run.

Analyse takes this single compute run and produces all trajectory, decision,
matched-coordinate accuracy, deadline-cost and Pareto measurements. Present takes
this single analysis run and always generates all six figures. Stages remain
independent: no stage launches another stage or chooses its input automatically.

Run each stage separately from the repository root, using the exact identity
printed by the preceding stage.

```sh
uv run python experiments/exp123/compute.py --source exp022-r001-compute
uv run python experiments/exp123/analyse.py --source <compute-run-id>
uv run python experiments/exp123/present.py --source <analyse-run-id>
```

## Historical execution

The dated records below describe earlier runs and the former branched workflow.
Its multi-source commands are historical; use the three commands above for the
current complete experiment. Historical run IDs may have been pruned.

## Initial twenty-image execution — 2026-10-06

1. `exp123-r002-compute`: 600 completed graph presentations. The first allocation
   did not execute because the article collection declaration was not yet present.
2. `exp123-r003-analyse`: cycle coordinates available for all twenty images;
   common complete-interval coverage is four to six cycles per image.
3. `exp123-r004-present`: twenty-panel time and cycle grids, plus scientific numbers.

At 3/6/12 ms decay, the final argmax is correct in 93.5/93.0/91.0% of the
200 presentations per setting. These repeated encodings of twenty fixed images
are not a representative accuracy benchmark or independent training replicates.
Maximum scores reach 0.99 by the end in 99.0/98.5/97.5% of presentations;
median first-crossing times among crossing trials are 23.45/34.5/47.8 ms.
No crossing precedes the first detected I volley. High scores include wrong
predictions, and late saturation can create apparent trajectory agreement.
Assess early trajectories before interpreting a cycle-coordinate collapse.

## Ten-image presentation — 2026-10-06

Reduced the active protocol to the first ten official test images and reran
300 presentations, retaining ten paired encodings and three decay settings.
All retained images reproduce their previous measurements exactly. The two
figures now use five rows and two columns, doubling each panel's horizontal
space. Selected execution: `exp123-r005-compute` → `exp123-r006-analyse` →
`exp123-r007-present`. Historical twenty-image outputs remain unchanged.

## One hundred encodings per image — 2026-10-06

Expanded the active protocol to 100 paired encoding draws per image, identifiers
42–141, retaining ten images and three decay settings. All 3000 presentations
completed. Cycle coordinates are available for every image, with four to six
complete intervals shared across all 300 presentations within each image.
Both figures average 100 draws and label the y-axis “True-class softmax score”.
Selected execution: `exp123-r010-compute` → `exp123-r011-analyse` →
`exp123-r012-present`. Earlier completed runs remain unchanged.

## Sustained decision measurement — 2026-10-06

Reused `exp123-r010-compute` without new simulations. `exp123-r013-analyse`
adds unique-winner decision times, cycle positions and conditional quartiles.
Selected presentation `exp123-r014-present` retains the two trajectory figures
and adds the decision comparison with explicit success and cycle-mapping counts.

## Decision figure readability — 2026-10-06

`exp123-r015-present` replaces the overlaid decision curves and separate coverage
table with ten pairs of time/cycle small multiples. Black medians and red
interquartile ranges isolate each image's trend. Included sample counts remain
inside each plot, in 3/6/12 ms order. Scales are shared within each coordinate
except image 08's single late physical-time decision. Analysis is unchanged.

## Decision rows — 2026-10-06

Selected `exp123-r017-present`: all ten physical-time plots occupy the top row,
with cycle plots directly underneath in matching image order. Panel labels and
the caption follow row order. The same analysis and conditional sample counts
are preserved; no simulation or measurement was rerun.

## Decision-axis cleanup — 2026-10-06

Selected `exp123-r018-present` removes the in-plot sample-count annotations.
Time ticks are spaced by 10 ms (25 ms for image 08's larger range), and cycle
ticks by 0.5 cycles. Conditional sample counts remain in the scientific numbers;
the caption retains the success and cycle-mapping selection definitions.

## House-style presentation — 2026-10-06

Selected `exp123-r019-present`, reusing `exp123-r013-analyse`. All three figures
use the shared paper theme, typography roles and canonical bold panel labels.
Trajectory figures use black/red/cyan for the three decay settings. The decision
figure retains black medians, red interquartile ranges, time above cycles and
the denser y ticks, without sample-count annotations. Measurements are unchanged.

## Decision row labels — 2026-10-06

Selected `exp123-r020-present`: Figure 3 uses A for the time row and B for the
cycle row, as requested, instead of separate letters for all twenty plots.
Image indices and digits remain above each plot; caption references match.

## Matched-coordinate accuracy protocol

1. Use a second independent compute protocol: 500 official MNIST test images,
   sampled without replacement as 50 per digit with selection seed 123. Use
   three encoding draw identifiers, 42–44, paired across 3/6/12 ms decay.
   The frozen weights, graph, input rate and 200 ms duration are unchanged.
   Batch up to 32 images for execution; images do not interact.
2. Measure instantaneous cumulative-count argmax accuracy, with lowest class
   index winning ties, on a 1 ms time grid, a 0.25-cycle grid and input-count
   budgets spaced by 25 spikes. This is not the retrospective sustained-winner
   measure used in Figure 3.
3. Cycle position zero is the first accepted I volley. Query the last completed
   output update at the interpolated physical time, without interpolating class
   predictions. For input budgets, query the first completed update reaching
   the budget; simultaneous events can overshoot it.
4. At each cycle coordinate, retain an image/seed only when all three decay
   settings have accepted sequences and reach that coordinate. Input-budget
   coverage is automatically paired because input tensors are identical.
   Include incorrect predictions; never condition on eventual correctness.
5. Average eligible draws within images, then give eligible images equal weight.
   Compute percentile 95% intervals from 1000 paired image bootstrap resamples,
   stratified by digit, with seed 12345. These quantify image sampling variation,
   conditional on this trained network and sampled encodings.
6. Show accuracy in three matched-coordinate panels, with paired-trial coverage
   underneath. Suppress accuracy below 80% coverage; show axes through the last
   coordinate retaining 50% plus one grid step. Report coverage and eligible
   image counts in the scientific numbers. High-budget and late-cycle subsets
   can change image composition even before the display cutoff.
7. Preserve the ten-image/100-draw trajectory and decision figures through an
   explicitly supplied separate analysis input. The new protocol does not
   replace their data or import another experiment's implementation.

```sh
uv run python experiments/exp123/compute.py --source exp022-r001-compute --protocol accuracy
uv run python experiments/exp123/analyse.py --source <accuracy-compute-run-id>
uv run python experiments/exp123/present.py --source exp123-r013-analyse --accuracy-source <accuracy-analyse-run-id>
```

## Matched-coordinate accuracy execution — 2026-10-06

Completed `exp123-r021-compute`: 500 images, exactly 50 per digit, three paired
encoding draws and three decay settings, giving 4500 graph presentations.
`exp123-r023-analyse` measures matched-coordinate accuracy and image-bootstrap
intervals. An earlier completed analysis had numerical backend warnings during
matrix multiplication; the selected analysis uses explicit contraction and
completed without those warnings. No simulations were repeated for this change.
Selected `exp123-r024-present` includes Figures 1–3 from `exp123-r013-analyse`
and the new accuracy/coverage Figure 4 from `exp123-r023-analyse`.

At 40 ms, accuracy is 81.5/74.3/58.8% for 3/6/12 ms decay. At cycle position
two it is 69.9/73.7/73.3%; at an input budget of 100 spikes it is
80.9/74.5/58.9%, with complete paired coverage there. Final 200 ms accuracy is
91.6/90.6/89.0%. Curves are closer in cycle coordinates, but do not establish
identical evidence per cycle or information creation. Accuracy is displayed
through cycle position five and 350 input spikes at the 80% coverage rule.
Whole-image uncertainty is conditional on the frozen network and three recorded
encodings, not independent training replicates.

## Accuracy–deadline–spike-cost protocol

1. Keep the balanced 500-image sample and three paired encoding draws, and use
   maximum-pixel encoding rates 12.5, 25 and 50 Hz at GABA decay 3, 6 and 12 ms.
   Reuse the validated 25 Hz compute input explicitly; the new compute protocol
   produces only the two added rates. There are 13,500 total presentations,
   9000 newly simulated. Each recording is 200 ms with no burn-in.
2. Reuse each image/draw's random uniforms across encoding rates. Increasing
   the rate changes the Bernoulli threshold, so low-rate input spikes must be
   nested within the baseline spikes and those within the high-rate spikes.
   Verify nested sets, pixels, labels, seeds, biological configuration and
   checkpoint identity before combining inputs. Rate is a sampling-quality
   proxy that also changes drive, not an isolated SNR intervention.
3. Read cumulative output-count predictions at 20, 40, 80, 120 and 200 ms;
   use the same lowest-class-index tie rule as the accuracy comparison.
   Measure E + I + output spikes through each deadline, excluding input spikes.
   Include every trial, including errors; average three draws per image then
   all 500 images equally. This is a spike-activity proxy, not physical energy.
4. Estimate separate empirical frontiers among the 15 decay/deadline candidates
   at each encoding rate. Maximize accuracy, minimize deadline and minimize mean
   network spikes. A competitor must weakly improve all objectives and strictly
   improve at least one. Compare integer correct totals and summed spike counts
   before division to avoid numerical ambiguity for tied accuracy.
5. Resample whole images 1000 times within digit strata, keeping all rates,
   decay settings and seeds paired. Retain percentile 95% accuracy/cost
   intervals and each point's bootstrap nondomination membership fraction.
   Empirical frontier membership is sample-dependent, not established dominance
   in the underlying population.
6. Measure full-trial E spectral frequency using SNNLab's full-200-ms Hann Welch
   spectrum at native 0.1 ms sampling, centering without detrending. Take the
   parabolically interpolated peak of the mean spectrum across all 1500
   presentations in a 5–150 Hz search, with spectrum-neighbor interpolation and
   no harmonic rejection. Also retain trial-level peak quartiles and mean PSDs.
7. Render accuracy/time, cost/time and accuracy/cost traces for each rate, then
   frontier bubbles with deadline on x, accuracy on y and a common area scale
   proportional to mean network spike count. Connect each decay setting's five
   deadlines; highlight empirical nondominated points and fade dominated ones.
   Show the measured spectral peaks alongside the frontier plots. Preserve
   previous figures through their explicitly supplied analysis inputs.

```sh
uv run python experiments/exp123/compute.py --source exp022-r001-compute --protocol pareto
uv run python experiments/exp123/analyse.py --source exp123-r021-compute --added-rates-source <added-rates-compute-run-id>
uv run python experiments/exp123/present.py --source exp123-r013-analyse --accuracy-source exp123-r023-analyse --pareto-source <pareto-analyse-run-id>
```

## Pareto execution — 2026-10-06

Completed `exp123-r025-compute`: 9000 new graph presentations at 12.5 and 50 Hz,
with the same 500 images, three draw identifiers and three decay settings as
`exp123-r021-compute`. `exp123-r026-analyse` combines those validated inputs:
13,500 presentations, 45 deadline candidates, and verified nested input-spike
sets for every image/draw. Baseline deadline accuracy agrees with the earlier
matched-coordinate analysis. All trial spike costs increase with deadline.

Selected `exp123-r028-present` preserves Figures 1–4 and adds Figure 5's
accuracy/deadline/cost traces and Figure 6's connected frontier bubbles plus
spectral-frequency diagnostics. Captions and Methods define the input-rate
proxy, population-total spike cost, exact empirical dominance and image-level
uncertainty. All scientific exports remain in immutable v4 run folders.

At 25 Hz and 200 ms, 3/6/12 ms decay gives accuracy 91.6/90.6/89.0% and mean
network costs 25,268/15,948/9,551 spikes. Shorter decay improves deadline accuracy
while costing more activity; differences narrow when accuracy is plotted against
spike count. Empirical frontier membership is 15/15, 15/15 and 14/15 candidates
at 12.5/25/50 Hz. The 50 Hz, 6 ms-decay, 200 ms candidate is dominated by the
3 ms-decay, 120 ms candidate. Many memberships change under bootstrap resampling.
Mean-PSD peaks for 3/6/12 ms decay are 77.8/46.9/27.2 Hz at 12.5 Hz input,
100.5/58.9/35.2 Hz at 25 Hz input and 126.2/77.4/45.8 Hz at 50 Hz input.
These are spectral estimates with possible harmonic ambiguity, not proof of a
single invariant gamma mechanism or measurements of physical energy.

## Nine-image physical-time figure — 2026-10-07

`exp123-r029-present` redraws Figure 1 as a 3 × 3 grid using only images 0–8;
its plotting path no longer loads or renders image 9. Figure 2 and decision
summaries retain their ten-image sample. The presentation reuses explicit
analysis inputs `exp123-r013-analyse`, `exp123-r023-analyse` and
`exp123-r026-analyse`; no simulations or measurements were rerun.

## Consolidated full execution — 2026-10-07

The complete experiment now uses one run per stage:

1. `exp123-r040-compute`: 16,200 fresh graph presentations, including the nine
   trajectory images and all 500 benchmark images at three encoding rates.
2. `exp123-r041-analyse`: all trajectory, decision, matched-coordinate accuracy,
   deadline-cost, Pareto and spectral measurements, with one compute input.
3. `exp123-r042-present`: all six figures in PNG, SVG and PDF, with one analysis
   input. Selected by the article.

All nine trajectory score/count arrays and decision summaries matched the
preceding evidence exactly. Matched-coordinate curves, Pareto rows and spectral
estimates also matched exactly; nested inputs were verified for all 500 images.
Each public command creates one run and accepts only its single explicit source;
no stage launches another stage. Measurement helpers write into the owning
analysis run rather than allocating additional runs.

## Uniform figure dimensions — 2026-10-07

`exp123-r031-present` renders all six figures at 180 × 170 mm, defined once in
`recipe.py`. PNG, SVG and PDF exports preserve that canvas without tight
cropping. Panel arrangements and measurements are retained, including Figure
1's nine-image grid. It reuses the same three analysis inputs as the preceding
presentation; no simulations or measurements were rerun.

## Caption-width 16:9 figures — 2026-10-07

`exp123-r034-present` renders all six figures at 180 × 101.25 mm (16:9),
preserving the canvas in every export. The article displays images at full
caption width without a viewport-height cap. Figure 1 retains its nine-image
3 × 3 grid; Figure 2 arranges its ten images in two rows of five. Compact labels
and tick spacing keep the shorter layouts legible. The same explicit analysis
inputs are reused; no simulations or measurements were rerun.

## Matched nine-image trajectory grids — 2026-10-07

`exp123-r035-present` uses the same images 0–8 for Figures 1 and 2, both in
3 × 3 grids. Neither trajectory plot loads or renders image 9. All figures
retain their caption-width 16:9 canvas. The existing analysis inputs and
measurements are reused.

## Matched nine-image decision figure — 2026-10-07

`exp123-r036-present` limits Figure 3 to the same images 0–8 as Figures 1 and 2.
Its two rows contain nine columns: decision time above, cycle position below.
The tenth image is excluded from the plotting code. All six figures retain
their caption-width 16:9 canvas; analysis inputs and measurements are reused.

## Completed cycle counts — 2026-10-07

`exp123-r037-analyse` reanalyses `exp123-r010-compute`, adding integer completed
inter-volley counts. Figure 2 samples scores at accepted volley completions,
starting with zero at the first volley. Figure 3 counts full intervals completed
at each sustained decision, within accepted volley coverage; fractional
medians and quartiles can arise from aggregation of integer trial values.
`exp123-r038-present` displays these measurements with integer cycle ticks,
retaining the same nine images and 16:9 layouts. Matched-coordinate accuracy
and Pareto inputs remain `exp123-r023-analyse` and `exp123-r026-analyse`.
No simulations were rerun.

## Decision figure axes — 2026-10-07

`exp123-r039-present` puts decision time on Figure 3A's x-axis and completed
cycles on Figure 3B's x-axis, with GABA decay on both y-axes. Medians and
interquartile ranges are shown horizontally for the same nine images.
It reuses `exp123-r037-analyse`, `exp123-r023-analyse` and
`exp123-r026-analyse`; no simulations or measurements were rerun.

After full validation, hash-bound exp123 pruning removed the seven superseded
branched runs (0.25 GiB). Only the consolidated compute, analyse and present runs
remain as completed exp123 evidence; the upstream training bank remains intact.


## Volley count starting at one — 2026-10-09

`exp123-r043-analyse` remeasures `exp123-r040-compute`, numbering accepted I
volleys from one for Figures 2 and 3. `exp123-r044-present` displays the new
coordinates. Trial decision counts and trajectory coordinates shift by exactly
+1; score arrays, decision times, matched-coordinate accuracy and Pareto
measurements are unchanged. Methods, captions and the illustrative diagram use
the same convention. No simulations were rerun.

## Initial score at volley zero — 2026-10-09

`exp123-r045-present` adds Figure 2's initial score at count zero, sampled from
the recorded time-zero state (all logits zero; score 0.1 for every digit).
This is an illustrative initial-state marker, not a detected volley. Detected
volleys still start at one; Figure 3 is unchanged. The same analysis input is
reused, without rerunning simulations or measurements.

## Evidence per completed volley — 2026-10-09

`exp123-r046-analyse` reanalyses `exp123-r040-compute`. Figure 2 now samples
cumulative true-class softmax at the first minimum of smoothed I activity
strictly between each accepted peak and the next. The final peak is excluded;
all 100 draws per setting share a common completed-volley count range per image.
The initial score remains at count zero. `exp123-r047-present` regenerates all
six figures. All 2700 trajectory trials were checked for ordered boundaries and
correct score sampling. Time trajectories, decision summaries and benchmark
arrays are unchanged. No simulations were rerun. The Methods illustration and
its standalone generating script have been removed.

## Stable decisions at completed volleys — 2026-10-09

`exp123-r049-analyse` reanalyses the same `exp123-r040-compute` recordings.
The completed-volley decision coordinate is the first trough boundary at or
after the retrospectively stable correct decision, within Figure 2's shared
count range and valid completion coverage. This measures the first Figure 2
sample at which the correct decision is already stable, rather than the count
of peaks already passed. `exp123-r051-present` shows median points only, for the
same nine images, with time 0–200 ms above and Figure 2's completed-volley count
ranges below. All 2700 decisions were verified against recorded troughs;
original decision times, trajectory and benchmark arrays are unchanged. No
simulations were rerun. `exp123-r048-analyse` is a superseded preliminary
measurement using counts of boundaries already passed and is not selected.

The final presentation angles time tick labels to keep the nine columns legible.

## Connected stable-decision medians — 2026-10-09

`exp123-r052-present` reuses `exp123-r049-analyse` and connects Figure 3's
median points across decay settings in both rows. Measurements and other
figures are unchanged; no simulations or analysis were rerun.
