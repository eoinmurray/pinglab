#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp042")

#let meta = (
  status: "Results available",
  title: "Breaking Gamma Releases the Rate Gate",
  date: "2026-06-02",
  description: "Overriding the I-stream of trained PING at inference shows what gates the E rate is the timing of inhibition, the rhythm, not its average level.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp042",)
#let preview-figures = (
  (path: "exp042/rhythm_compound.png", label: "rhythm compound"),
  (path: "exp042/cell_jitter_sweep.svg", label: "cell jitter sweep"),
  (path: "exp042/jitter_sweep.svg", label: "jitter sweep"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#set math.equation(numbering: "(1)")
// Provenance (HOUSESTYLE H9): every run number below is read from the run's
// numbers.json, never hand-typed, so a re-run updates the prose automatically.
#let run = data-json(data-file("exp042/numbers.json"))
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
#let cell_acc5 = calc.round(cell_at(5.0, "acc"), digits: 1)

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

  == Results

  === 1. Equal mean inhibition produces opposite excitatory rates

  #figure(
    data-image(data-file("exp042/rhythm_compound.png"),
      width: 100%,
      alt: "Matched per-cell and cycle-coherent inhibitory jitter. Per-cell jitter smears inhibitory bursts and silences excitatory cells; cycle-coherent jitter keeps bursts sharp, opens gaps, and raises excitatory firing at nearly the same realised inhibitory rate.",
    ),
    caption: [
      Matched inference-time perturbations at $sigma = #anchor_sigma$ ms. The
      top row shows one illustrative seed-42 trial. The bottom row shows mean
      excitatory rate, test accuracy, and realised inhibitory rate across three
      seeds across the full jitter sweeps. These compound-panel curves show
      means without error bars; the standalone sweeps show ±1 standard error.
    ],
  )

  At $sigma = #anchor_sigma$ ms, smearing individual inhibitory spikes almost
  silenced excitatory cells. Moving whole bursts instead raised firing from
  #base_e Hz to #cyc_e_anchor Hz, while accuracy remained #cyc_acc_anchor%.
  Mean inhibitory rates were similar: #cell_i_anchor and #cyc_i_anchor Hz,
  compared with #base_i Hz at baseline. Similar amounts of inhibition produced
  opposite outcomes: its timing matters.

  === 2. Millisecond-scale smearing collapses firing and accuracy

  #figure(
    data-image(data-file("exp042/cell_jitter_sweep.svg"),
      width: 100%,
      alt: "Excitatory rate and accuracy fall steeply as independent inhibitory-spike jitter increases, while realised inhibitory rate remains nearly flat.",
    ),
    caption: [
      Per-I-cell jitter sweep across three frozen networks. Points show
      across-seed means; error bars are ±1 standard error of the mean. The grey
      trace is the realised mean inhibitory rate.
    ],
  )

  Small timing changes were enough: excitatory firing fell to #cell_e_half Hz
  at $sigma = 0.5$ ms and #cell_e1 Hz at 1 ms. By 5 ms, firing was nearly zero
  and accuracy was #cell_acc5%, despite little change in mean inhibition.
  This supports the idea that gaps between inhibitory bursts let excitatory
  cells recover and fire.

  === 3. Moving intact bursts releases excitatory firing

  #figure(
    data-image(data-file("exp042/jitter_sweep.svg"),
      width: 100%,
      alt: "Excitatory rate rises as coherent inhibitory bursts are displaced, accuracy declines gently, and realised inhibitory rate remains near baseline before falling at the largest offsets.",
    ),
    caption: [
      Cycle-coherent jitter sweep across three frozen networks. Points show
      across-seed means; error bars are ±1 standard error of the mean. The grey
      trace shows the realised mean inhibitory rate.
    ],
  )

  Moving intact bursts was consistent with opening longer gaps for excitatory
  firing. The rate rose past the phase-shuffled control (#shuf_e Hz), reaching
  #cyc_e_hi Hz at $sigma = 100$ ms. But mean inhibition then fell to
  #cyc_i_hi Hz, #cyc_i_drop_pct% below baseline, so that extreme is no longer
  a clean test of timing alone. The clearest comparison is at #anchor_sigma ms, where mean
  inhibition remains close to baseline.

  == Methods

  The experiment isolates inhibitory timing by replaying frozen networks,
  changing only when recorded inhibitory spikes arrive, and measuring the
  resulting excitatory activity and classification performance.

  + *Select the frozen networks.* We used the three baseline PING classifiers
    from the #link("/exp022/")[shared training study], with seeds 42–44, at their
    final training epoch because the experiment tests final-epoch dynamics
    rather than validation-selected deployment performance. The reference gamma
    frequency is the shared canonical operating-point constant.

  + *Fix the evaluation data.* Every intervention used the same fixed subset of
    #cfg.evaluation_samples_per_condition images from the official MNIST test
    partition for each of the three
    trained seeds. No weights were retrained or selected during evaluation.

  + *Record the baseline inhibitory stream.* For each batch, a baseline forward
    pass recorded

    $ bold(s)^I_"base" in {0,1}^(T times B times N_I). $

    Here $bold(s)^I_"base"$ is the inhibitory spike tensor, $T$ is the number of
    simulation timesteps, $B$ is the batch size, and $N_I$ is the number of
    inhibitory cells. Each entry is one when a cell spikes and zero otherwise.

  + *Construct the paired jitter interventions.* The reference gamma-cycle
    duration was

    $ P_gamma = 1000 / f_gamma. $

    Here $P_gamma$ is the cycle duration in milliseconds and $f_gamma$ is gamma
    frequency in hertz; at the baseline operating point, $P_gamma approx
    #period_ms$ ms. Both interventions draw offsets

    $ Delta tilde cal(N)(0, sigma^2). $

    Here $Delta$ is a temporal offset in milliseconds and $sigma$ is its standard
    deviation. Cycle-coherent jitter draws one $Delta$ for every trial and cycle
    and applies it to all inhibitory spikes in that cycle. Per-cell jitter draws
    an independent $Delta$ for every inhibitory spike. Offsets are rounded to
    simulation timesteps and shifted times are clamped to the finite trial
    window; spikes that coincide at the same cell and timestep merge.

  + *Construct the limiting controls.* Phase-shuffle applies one shared
    permutation of time to all inhibitory cells in a trial, preserving
    same-timestep co-firing while removing phase order. Rate-matched Poisson
    redraws each trial and cell from its observed spike count, removing temporal
    and cross-cell structure while preserving expected rate.

  + *Replay and measure the perturbed stream.* A second forward pass replaced
    the network's inhibitory spikes with the controlled stream. Excitatory
    cells received only the replacement stream through $W^(I E)$, the weight
    matrix from inhibitory to excitatory cells, and the frozen readout consumed
    the resulting excitatory spikes.

    For every condition we retained excitatory and inhibitory firing rates and
    test accuracy, then averaged each
    quantity over the three independently trained seeds. Sweep error bars are
    ±1 standard error of the mean across seeds.

    The compound raster is illustrative; all quantitative claims use the
    complete registered condition grids. Jitter
    moves spikes without intentionally changing their counts, but large
    offsets can produce coincident spikes after shifting and boundary clamping,
    reducing the realised inhibitory rate.
    Strict same-mean claims therefore use the shared $sigma = #anchor_sigma$ ms
    anchor, where realised inhibitory rates remain within
    #cyc_i_anchor_drop_pct% of baseline.
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Does the timing of inhibition matter independently of its mean strength? Compare cell-wise jitter with shifts of intact inhibitory volleys.],
    preview-figures, json-inputs: ("exp042",),
  )
}
