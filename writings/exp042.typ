#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp042")

#let meta = (
  status: "[▦ DATA | v29.0.0]",
  title: "Breaking Gamma Releases the Rate Gate",
  created_at: "2026-06-02T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "Inference-time overrides of the inhibitory stream test whether inhibitory timing, rather than average level alone, gates excitatory firing in trained PING classifiers.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp042",)
#let preview-figures = (
  (path: "exp042/rhythm_compound.png", label: "rhythm compound"),
  (path: "exp042/cell_jitter_sweep.svg", label: "cell jitter sweep"),
  (path: "exp042/jitter_sweep.svg", label: "jitter sweep"),
)

#let result-card-style = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-result-card { margin: 1.25rem 0; padding: 1.2rem 1.35rem 1.3rem; border: 1px solid var(--rule-strong); border-radius: 3px; background: var(--paper); } "
      + ".pinglab-result-card > h4:first-child { margin-top: 0; } "
      + ".pinglab-result-card > :last-child { margin-bottom: 0; } "
      + "@media (max-width: 520px) { .pinglab-result-card { margin: 1rem 0; padding: .95rem 1rem 1.05rem; } }",
    )
  }
}

#let result-card(body) = context {
  if target() == "html" {
    html.elem("article", attrs: (class: "pinglab-result-card"), body)
  } else { body }
}

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

// Independent inhibitory-spike jitter: E rate + accuracy along the collapse.
#let cell_e_half = calc.round(cell_at(0.5, "e_rate_hz"), digits: 1)
#let cell_e1 = calc.round(cell_at(1.0, "e_rate_hz"), digits: 1)
#let cell_acc5 = calc.round(cell_at(5.0, "acc"), digits: 1)

// Cycle-coherent jitter: baseline → high-σ E rise, accuracy plateau.
#let cyc_e_hi = calc.round(cyc_at(100.0, "e_rate_hz"), digits: 1)

// Timing-matched anchor for the compound figure and the controlled comparison.
#let anchor_sigma = 14
#let cyc_e_anchor = calc.round(cyc_at(14.0, "e_rate_hz"), digits: 1)
#let cyc_acc_anchor = calc.round(cyc_at(14.0, "acc"), digits: 1)
#let cyc_i_anchor = calc.round(cyc_at(14.0, "i_rate_hz"), digits: 1)

// Realised I rates for the mean-inhibition check.
#let base_i = calc.round(by_cond("baseline", "i_rate_hz"), digits: 1)
#let cell_i_anchor = calc.round(cell_at(14.0, "i_rate_hz"), digits: 1)
#let cyc_i_hi = calc.round(cyc_at(100.0, "i_rate_hz"), digits: 1)
#let cyc_i_drop_pct = calc.round(100 * (base_i - cyc_i_hi) / base_i)

// Gamma period 1/f_γ: the predicted transition timescale.
#let period_ms = calc.round(1000 / cfg.f_gamma_reference_hz, digits: 1)

#let body = [
  == Abstract


  We asked whether excitatory-rate suppression depends on the timing structure
  of inhibition rather than only its average amount. We replayed frozen PING
  classifiers while either jittering inhibitory spikes independently or shifting
  intact inhibitory bursts.

  Smearing individual spikes collapsed excitatory activity, whereas moving
  coherent bursts preserved synchrony and released excitatory firing. The
  comparison shows that inhibitory temporal structure gates excitatory rate
  within the tested intervention regime.

  == Results

  #with-result-sections[

  #result-card-style

  #result-card[
  === Similar mean inhibitory rates produce opposite excitatory rates

  #figure(
    data-image(data-file("exp042/rhythm_compound.png"),
      width: 100%,
      alt: "Matched independent-spike and cycle-coherent inhibitory jitter. Independent-spike jitter smears inhibitory bursts and silences excitatory neurons; cycle-coherent jitter keeps bursts sharp, opens gaps, and raises excitatory firing at nearly the same realised inhibitory rate.",
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
  silenced excitatory neurons. Moving whole bursts instead raised firing from
  #base_e Hz to #cyc_e_anchor Hz, while accuracy remained #cyc_acc_anchor%.
  Mean inhibitory rates were similar: #cell_i_anchor and #cyc_i_anchor Hz,
  compared with #base_i Hz at baseline. Similar amounts of inhibition produced
  opposite outcomes: its timing matters.
  ]

  #result-card[
  === Millisecond-scale smearing collapses firing and accuracy

  #figure(
    data-image(data-file("exp042/cell_jitter_sweep.svg"),
      width: 100%,
      alt: "Excitatory rate and accuracy fall steeply as independent inhibitory-spike jitter increases, while realised inhibitory rate remains nearly flat.",
    ),
    caption: [
      Independent inhibitory-spike jitter sweep across three frozen networks.
      Points show
      across-seed means; error bars are ±1 standard error of the mean. The grey
      trace is the realised mean inhibitory rate.
    ],
  )

  Small timing changes were enough: excitatory firing fell to #cell_e_half Hz
  at $sigma = 0.5$ ms and #cell_e1 Hz at 1 ms. By 5 ms, firing was nearly zero
  and accuracy was #cell_acc5%, despite little change in mean inhibition.
  This supports the idea that gaps between inhibitory bursts let excitatory
  neurons recover and fire.
  ]

  #result-card[
  === Moving intact bursts releases excitatory firing

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
  ]

  ]

  == Methods

  === Compute

  + *Frozen networks.* We used the three baseline PING classifiers
    from the #link("/exp022/")[shared training study], with seeds 42–44, at their
    final training epoch because the experiment tests final-epoch dynamics
    rather than validation-selected deployment performance. The reference gamma
    frequency is the shared canonical operating-point constant.

  + *Evaluation data.* Every intervention used the same fixed subset of
    #cfg.evaluation_samples_per_condition images from the official MNIST test
    partition for each of the three
    trained seeds. No weights were retrained or selected during evaluation.

  + *Record the baseline inhibitory stream.* For each batch, a baseline forward
    pass recorded

    $ bold(s)^I_"base" in {0,1}^(N_t times B times N_I). $

    Here $bold(s)^I_"base"$ is the inhibitory spike tensor, $N_t$ is the number of
    simulation timesteps, $B$ is the batch size, and $N_I$ is the number of
    inhibitory neurons. Each entry is one when a neuron spikes and zero otherwise.

  + *Construct the paired jitter interventions.* The reference gamma-cycle
    duration was

    $ T_gamma = 1000 / f_gamma. $

    Here $T_gamma$ is the cycle duration in milliseconds and $f_gamma$ is gamma
    frequency in hertz; at the baseline operating point, $T_gamma approx
    #period_ms$ ms. Both interventions drew offsets

    $ Delta tilde cal(N)(0, sigma^2). $

    Here $Delta$ is a temporal offset in milliseconds and $sigma$ is its standard
    deviation. Cycle-coherent jitter drew one $Delta$ for every trial and cycle
    and applied it to all inhibitory spikes in that cycle. Independent-spike
    jitter drew a separate $Delta$ for every inhibitory spike. Offsets were rounded to
    simulation timesteps and shifted times were clamped to the finite trial
    window; spikes that coincided at the same neuron and timestep merged.

  + *Construct the limiting controls.* Phase-shuffle applied one shared
    permutation of time to all inhibitory neurons in a trial, preserving
    same-timestep co-firing while removing phase order. Rate-matched Poisson
    redrew each trial and neuron from its observed spike count, removing temporal
    and cross-neuron structure while preserving expected rate.

  + *Replay and measure the perturbed stream.* A second forward pass replaced
    the network's inhibitory spikes with the controlled stream. Excitatory
    neurons received only the replacement stream through $W^(I E)$, the weight
    matrix from inhibitory to excitatory neurons, and the frozen readout consumed
    the resulting excitatory spikes.

    For every condition we recorded per-neuron excitatory and inhibitory firing
    rates over the full presentation and test accuracy on the fixed test subset.

  === Analyse

  #set enum(start: 7)

  + *Aggregate the measurements.* We averaged each rate and accuracy over the
    three independently trained networks. Sweep uncertainty is ±1 standard
    error of the mean across those three training replicates.

  + *Identify the timing-matched comparison.* Jitter moves spikes without
    intentionally changing their counts, but boundary clamping and collisions
    at the same neuron and timestep can reduce the realised inhibitory rate.
    We therefore restricted the timing-matched comparison to the shared
    $sigma = #anchor_sigma$ ms anchor, where the realised inhibitory rate in
    both interventions remained close to baseline.

  === Present

  #set enum(start: 9)

  + *Expose the matched intervention.* We displayed the fixed seed-42 trial at
    $sigma = #anchor_sigma$ ms beside the complete recorded sweep summaries for
    both interventions. The raster is illustrative; the quantitative comparison
    uses all three independently trained networks and the full evaluation set.

  + *Expose the separate sweeps.* We displayed recorded condition means for
    excitatory rate, inhibitory rate and test accuracy across independent-spike
    and cycle-coherent jitter. The standalone sweeps show ±1 standard error
    across training replicates; presentation did not rerun the interventions or
    recompute the measurements.
]
#body
  #run-view("exp042", inputs)

]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Does the timing of inhibition matter independently of its mean strength? Compare independent inhibitory-spike jitter with shifts of intact inhibitory volleys.],
    preview-figures, json-inputs: ("exp042",),
  )
}

#let meta = meta + (assets: input-assets("exp042", inputs))
#let body = with-datasets("exp042", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
