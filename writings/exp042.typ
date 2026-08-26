#let meta = (
  title: "Breaking Gamma Releases the Rate Gate",
  date: "2026-06-02",
  description: "Overriding the I-stream of trained PING at inference shows what gates the E rate is the timing of inhibition, the rhythm, not its average level.",
  collection: "gamma-gated-sparsity",
)

// Provenance (HOUSESTYLE H9): every run number below is read from the run's
// numbers.json, never hand-typed, so a re-run updates the prose automatically.
#let run = json("/.artifacts/exp042/numbers.json")
#let cfg = run.config
#let mean(a) = a.sum() / a.len()

// Condition-level results (baseline / phase-shuffle / Poisson), averaged over seeds.
#let by_cond(cond, key) = mean(
  run.results.filter(r => r.condition == cond).map(r => r.at(key)),
)
#let base_e = calc.round(by_cond("baseline", "e_rate_hz"), digits: 1)
#let shuf_e = calc.round(by_cond("phase_shuffled_i", "e_rate_hz"), digits: 1)

// Sweep helpers: average a metric over seeds at a given σ.
#let cell_at(s, key) = mean(
  run.cell_jitter_sweep.filter(r => calc.abs(r.sigma_ms - s) < 0.001).map(r => r.at(key)),
)
#let cyc_at(s, key) = mean(
  run.jitter_sweep.filter(r => calc.abs(r.sigma_ms - s) < 0.001).map(r => r.at(key)),
)

// Per-I-cell jitter: E rate + accuracy along the collapse.
#let cell_e_half = calc.round(cell_at(0.5, "e_rate_hz"), digits: 1)
#let cell_e1 = calc.round(cell_at(1.0, "e_rate_hz"), digits: 1)
#let cell_e5 = calc.round(cell_at(5.0, "e_rate_hz"), digits: 1)
#let cell_acc05 = calc.round(cell_at(0.5, "acc"), digits: 1)
#let cell_acc1 = calc.round(cell_at(1.0, "acc"), digits: 1)
#let cell_acc2 = calc.round(cell_at(2.0, "acc"), digits: 1)
#let cell_acc5 = calc.round(cell_at(5.0, "acc"), digits: 1)
#let cell_acc9 = calc.round(cell_at(9.0, "acc"), digits: 1)

// Cycle-coherent jitter: baseline → high-σ E rise, accuracy plateau.
#let cyc_e_hi = calc.round(cyc_at(100.0, "e_rate_hz"), digits: 1)

// Rate-matched anchor for the compound figure and the strict same-mean claim.
#let anchor_sigma = 14
#let cyc_e_anchor = calc.round(cyc_at(14.0, "e_rate_hz"), digits: 1)
#let cyc_acc_anchor = calc.round(cyc_at(14.0, "acc"), digits: 1)
#let cyc_i_anchor = calc.round(cyc_at(14.0, "i_rate_hz"), digits: 1)

// Realised I rates for the mean-inhibition check.
#let base_i = calc.round(by_cond("baseline", "i_rate_hz"), digits: 1)
#let cell_i_anchor = calc.round(cell_at(14.0, "i_rate_hz"), digits: 1)
#let cyc_i_hi = calc.round(cyc_at(100.0, "i_rate_hz"), digits: 1)
#let cyc_i_drop_pct = calc.round(100 * (base_i - cyc_i_hi) / base_i)
#let cyc_i_anchor_drop_pct = calc.round(100 * (base_i - cyc_i_anchor) / base_i)

// Gamma period 1/f_γ: the predicted transition timescale.
#let period_ms = calc.round(1000 / cfg.f_gamma_reference_hz, digits: 1)

#let body = [
  == Abstract

  Precisely timed inhibitory bursts, rather than mean inhibition alone, suppress
  excitatory firing in trained pyramidal–interneuron network gamma (PING)
  classifiers. We replayed three frozen networks while either independently
  jittering inhibitory spikes or shifting whole inhibitory bursts. At the
  rate-matched comparison, per-cell jitter smeared the bursts and reduced the
  excitatory rate from #base_e Hz to approximately zero, whereas cycle-coherent
  jitter preserved burst synchrony and raised it to #cyc_e_anchor Hz. Within the
  tested regime, inhibitory temporal structure therefore gates excitatory rate.

  == Dependancies

  This experiment depends on #link("/exp022/")[exp022]. It replays exp022's
  final `TR-02` networks for seeds 42–44. The reference gamma frequency is the
  collection's shared canonical operating-point constant.

  == Results

  Three plots test the central claim, locate the temporal precision at which it
  fails, and show the contrasting effect of moving intact inhibitory bursts.

  === 1. Equal mean inhibition produces opposite excitatory rates

  We first compared two perturbations at the same jitter magnitude,
  $sigma = #anchor_sigma$ ms. Both keep the realised inhibitory rate close to the
  #base_i Hz baseline, but per-cell jitter smears each burst while cycle-coherent
  jitter moves the burst as a unit. A mean-inhibition account predicts comparable
  excitatory rates. A timing account instead predicts suppression when bursts are
  smeared and release when intact bursts are displaced.

  #figure(
    image(
      "/.artifacts/exp042/rhythm_compound.png",
      width: 100%,
      alt: "Matched per-cell and cycle-coherent inhibitory jitter. Per-cell jitter smears inhibitory bursts and silences excitatory cells; cycle-coherent jitter keeps bursts sharp, opens gaps, and raises excitatory firing at nearly the same realised inhibitory rate.",
    ),
    caption: [
      Matched inference-time perturbations at $sigma = #anchor_sigma$ ms. The
      top row shows one illustrative seed-42 trial. The bottom row shows mean
      excitatory rate, test accuracy, and realised inhibitory rate across three
      seeds; error bars are ±1 standard error of the mean for rate and accuracy.
    ],
  )

  The two arms move in opposite directions. Per-cell jitter leaves a realised
  inhibitory rate of #cell_i_anchor Hz but drives excitatory firing to
  approximately zero. Cycle-coherent jitter leaves #cyc_i_anchor Hz—within
  #cyc_i_anchor_drop_pct% of baseline—yet raises excitatory firing from #base_e Hz
  to #cyc_e_anchor Hz while accuracy remains #cyc_acc_anchor%. This contradicts
  the mean-only expectation and matches the prediction that inhibitory timing
  controls when excitatory cells can recover and fire.

  === 2. Millisecond-scale smearing collapses firing and accuracy

  We next varied the amount of independent jitter applied to every inhibitory
  spike. If narrow synchronous bursts create recovery gaps, then disrupting
  synchrony by only a few milliseconds should turn the inhibitory stream into a
  more continuous shunt, reducing excitatory firing even while its mean rate
  remains nearly fixed.

  #figure(
    image(
      "/.artifacts/exp042/cell_jitter_sweep.svg",
      width: 100%,
      alt: "Excitatory rate and accuracy fall steeply as independent inhibitory-spike jitter increases, while realised inhibitory rate remains nearly flat.",
    ),
    caption: [
      Per-I-cell jitter sweep across three frozen networks. Points show
      across-seed means; error bars are ±1 standard error of the mean. The grey
      trace is the realised mean inhibitory rate.
    ],
  )

  Excitatory firing falls from #base_e Hz to #cell_e_half Hz at
  $sigma = 0.5$ ms, #cell_e1 Hz at 1 ms, and #cell_e5 Hz at 5 ms. Accuracy
  follows from #cell_acc05% at 0.5 ms through #cell_acc1% at 1 ms and
  #cell_acc2% at 2 ms to #cell_acc5% at 5 ms, reaching approximately chance
  (#cell_acc9%) by 9 ms. The rapid collapse under nearly unchanged inhibitory
  rate matches the expected loss of narrow recovery gaps.

  === 3. Moving intact bursts releases excitatory firing

  Finally, we shifted each inhibitory burst bodily while preserving synchrony
  inside it. We expected displaced but intact bursts to open longer gaps in the
  inhibitory stream, so excitatory firing should rise rather than collapse. The
  strongest comparison remains the rate-matched range through
  $sigma = #anchor_sigma$ ms; larger offsets can push bursts beyond the finite
  trial window and reduce the delivered inhibitory rate.

  #figure(
    image(
      "/.artifacts/exp042/jitter_sweep.svg",
      width: 100%,
      alt: "Excitatory rate rises as coherent inhibitory bursts are displaced, accuracy declines gently, and realised inhibitory rate remains near baseline before falling at the largest offsets.",
    ),
    caption: [
      Cycle-coherent jitter sweep across three frozen networks. Points show
      across-seed means; error bars are ±1 standard error of the mean. The grey
      trace shows the realised mean inhibitory rate.
    ],
  )

  By $sigma = #anchor_sigma$ ms, excitatory firing has risen to
  #cyc_e_anchor Hz while the realised inhibitory rate remains within
  #cyc_i_anchor_drop_pct% of baseline. This matches the predicted release caused
  by moving intact bursts. The rise continues past the phase-shuffled reference
  of #shuf_e Hz to #cyc_e_hi Hz at 100 ms, but the inhibitory rate then falls to
  #cyc_i_hi Hz (#cyc_i_drop_pct% below baseline), so that extreme cannot support
  a strictly rate-matched interpretation.

  == Methods

  The experiment isolates inhibitory timing by replaying frozen networks,
  changing only when recorded inhibitory spikes arrive, and measuring the
  resulting excitatory activity and classification performance.

  + *Load the upstream networks.* We used the exp022 `TR-02` PING baselines for
    seeds 42–44 and selected `weights_final.pth` because the experiment tests
    final-epoch dynamics rather than validation-selected deployment performance.

  + *Fix the evaluation data.* Every intervention used the same fixed subset of
    1,000 images from the official MNIST test partition for each of the three
    trained seeds. No weights were retrained or selected during evaluation.

  + *Record the baseline inhibitory stream.* For each batch, a baseline forward
    pass recorded

    $ bold(s)^I_"base" in {0,1}^(T times B times N_I). quad "(1)" $

    Here $bold(s)^I_"base"$ is the inhibitory spike tensor, $T$ is the number of
    simulation timesteps, $B$ is the batch size, and $N_I$ is the number of
    inhibitory cells. Each entry is one when a cell spikes and zero otherwise.

  + *Construct the paired jitter interventions.* The reference gamma-cycle
    duration was

    $ P_gamma = 1000 / f_gamma. quad "(2)" $

    Here $P_gamma$ is the cycle duration in milliseconds and $f_gamma$ is gamma
    frequency in hertz; at the baseline operating point, $P_gamma approx
    #period_ms$ ms. Both interventions draw offsets

    $ Delta tilde cal(N)(0, sigma^2). quad "(3)" $

    Here $Delta$ is a temporal offset in milliseconds and $sigma$ is its standard
    deviation. Cycle-coherent jitter draws one $Delta$ for every trial and cycle
    and applies it to all inhibitory spikes in that cycle. Per-cell jitter draws
    an independent $Delta$ for every inhibitory spike.

  + *Construct the limiting controls.* Phase-shuffle applies one shared
    permutation of time to all inhibitory cells in a trial, preserving
    same-timestep co-firing while removing phase order. Rate-matched Poisson
    redraws each trial and cell from its observed spike count, removing temporal
    and cross-cell structure while preserving expected rate.

  + *Replay the perturbed stream.* A second forward pass replaced the network's
    inhibitory spikes through the exp037 hidden-perturbation hook. Excitatory
    cells received only the replacement stream through $W^(I E)$, the weight
    matrix from inhibitory to excitatory cells, and the frozen readout consumed
    the resulting excitatory spikes.

  + *Measure and aggregate the response.* For every condition we retained
    excitatory and inhibitory firing rates and test accuracy, then averaged each
    quantity over the three independently trained seeds. Sweep error bars are
    ±1 standard error of the mean across seeds.

  + *State the evidence boundary.* The compound raster is illustrative; all
    quantitative claims use the complete registered condition grids. Jitter
    moves spikes without intentionally changing their counts, but large
    cycle-coherent offsets can cross the finite trial boundary and be lost.
    Strict same-mean claims therefore use the shared $sigma = #anchor_sigma$ ms
    anchor, where realised inhibitory rates remain within
    #cyc_i_anchor_drop_pct% of baseline.
]
