#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp037")

#let meta = (
  status: "Ready for review",
  title: "Dropped Spikes vs Added Noise",
  date: "2026-05-30",
  updated_at: "2026-08-28",
  description: "Both trained networks tolerated substantial spike deletion, but PING accuracy fell more sharply under added spikes. The perturbations changed both recurrent feedback and readout input, so they do not isolate gamma gating.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp037",)
#let preview-figures = (
  (path: "exp037/perturbation_curves.svg", label: "perturbation curves"),
  (path: "exp037/perturb_rasters__drop__ping.png", label: "perturb rasters drop ping"),
  (path: "exp037/perturb_rasters__add__ping.png", label: "perturb rasters add ping"),
  (path: "exp037/perturb_rasters__drop__coba.png", label: "perturb rasters drop coba"),
  (path: "exp037/perturb_rasters__add__coba.png", label: "perturb rasters add coba"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp037/numbers.json"))
#let cfg = run.config
#let rounded(value) = calc.round(value, digits: 1)
#let mean(values) = values.sum() / values.len()
#let pert = run.perturbation_summary
#let point(model, mode, level) = pert.filter(r => r.model == model and r.mode == mode and calc.abs(r.level - level) < 0.001).first()
#let acc(model, mode, level) = rounded(point(model, mode, level).acc)
#let reference-rate(model) = mean(run.baseline_results.filter(r => r.model == model and r.rate_target_hz == none).map(r => r.rate_e))
#let eval_n = cfg.evaluation_samples_per_seed.first()
#let add_max = calc.max(..pert.filter(r => r.mode == "add").map(r => r.level))
#let knee-points = pert.filter(r => r.model == "ping" and r.mode == "add" and r.acc < 80).sorted(key: r => r.level)
#let knee = if knee-points.len() > 0 { knee-points.first().level } else { none }
#let labels = run.at("illustrative_labels", default: ())
#let trial-description = if labels.len() > 0 and labels.all(label => label == labels.first()) {
  [the same test image of digit #labels.first()]
} else { [the same test-image index in each condition] }
// HTML has no paged layout context; keep its images native and responsive.
#let report-image(path, alt, ratio: 0.58) = context {
  if target() == "html" {
    data-image(data-file(path), width: 100%, alt: alt)
  } else {
    // Fix the page frame so a short remainder cannot crop the image.
    layout(size => {
      let width = size.width
      box(width: width, height: width * ratio,
        data-image(data-file(path), width: width, height: width * ratio, fit: "contain", alt: alt))
    })
  }
}

  == Abstract

  Both trained networks tolerated substantial spike deletion, whereas added
  spikes reduced PING accuracy much more sharply than COBA accuracy. We
  reanalysed retained MNIST evaluations from three independently trained seeds
  per model on #eval_n test images. At 80% deletion probability, PING scored
  #acc("ping", "drop", 0.8)% and COBA #acc("coba", "drop", 0.8)%; deleting all
  spikes reduced both to #acc("ping", "drop", 1.0)%. At the largest nominal
  added rate (#add_max Hz per neuron), PING scored #acc("ping", "add", add_max)%
  and COBA #acc("coba", "add", add_max)%. This asymmetry constrains robustness
  of these trained networks, but does not separate recurrent timing from
  firing-rate or readout effects.

  == Results

  === 1. Deletion and insertion have different effects

  #figure(
    report-image("exp037/perturbation_curves.svg",
      "Mean test accuracy with sample standard deviation: both models tolerate substantial deletion; PING declines more steeply under added spikes.", ratio: 0.55),
    caption: [
      Means ± sample SD across seeds 42–44, #eval_n test images per seed.
      Unperturbed PING/COBA accuracies were #acc("ping", "drop", 0)%/
      #acc("coba", "drop", 0)%. The dashed line is nominal 10% chance.
      The added-rate axis divides by final-epoch reference-image E rates
      (#rounded(reference-rate("ping"))/#rounded(reference-rate("coba")) Hz),
      not test-set baseline rates. The same 0–#add_max Hz sweep therefore covers
      different normalized ranges; it does not match relative perturbation doses.
      #if knee != none [PING's first sampled mean below 80% occurred at #knee Hz;
        this is a grid crossing, not an estimated critical threshold.]
    ],
  )

  === 2. PING rasters retain bands under partial deletion

  #figure(
    report-image("exp037/perturb_rasters__drop__ping.png",
      "Three PING rasters at deletion probabilities 0, 0.5 and 1: visible bands remain at partial deletion; full deletion leaves no recorded spikes."),
    caption: [
      Retained seed-42 trials of #trial-description at deletion probabilities
      0, 0.5 and 1. E spikes (black) are below I spikes (red). The display samples
      200 E and 64 I cells; annotated E rates use the full population.
      Banding is visible at partial deletion, but these illustrative rasters
      provide no quantitative phase-coherence or gamma-frequency estimate.
    ],
  )
  #figure(
    report-image("exp037/perturb_rasters__add__ping.png",
      "Three PING rasters at nominal added rates 0, 20 and 40 Hz: dense inserted spikes increasingly obscure the unperturbed banding."),
    caption: [
      The same seed and test image at nominal added rates 0, 20 and #add_max Hz
      per neuron, applied independently to E and I. The displayed stream
      includes inserted spikes, which increasingly obscure its bands; this
      alone does not establish that an underlying oscillator disappeared.
      Display sampling and E-rate annotation follow the preceding figure.
    ],
  )

  === 3. COBA activity thins or increases with the intervention

  #figure(
    report-image("exp037/perturb_rasters__drop__coba.png",
      "Three COBA rasters at deletion probabilities 0, 0.5 and 1: E activity thins and becomes silent at full deletion."),
    caption: [
      Retained seed-42 COBA trials of #trial-description, with the same
      display sampling. E activity thins as deletion increases and vanishes at
      full deletion. This qualitative pattern does not imply accuracy is
      unchanged across the deletion sweep.
    ],
  )
  #figure(
    report-image("exp037/perturb_rasters__add__coba.png",
      "Three COBA rasters at nominal added rates 0, 20 and 40 Hz: E activity grows and imposed I spikes appear despite disabled recurrent coupling."),
    caption: [
      The same COBA trial at nominal added rates 0, 20 and #add_max Hz per neuron.
      Inserted E spikes increase recorded activity; imposed I spikes are also
      visible even though COBA's E→I→E coupling is disabled. The unperturbed
      population is denser than PING's, so equal nominal rates are not equal
      fractional perturbations.
    ],
  )

  #block(sticky: true)[
    == Methods

    We reused trained networks and retained inference trials to compare deletion
    and insertion of hidden spikes, without retraining.
  ]
  #set math.equation(numbering: "(1)")

  + *Select trained classifiers.* MNIST handwritten digits #cite(1) supplied
    a 7,000-image training pool, split into 6,300 optimization and 700 validation
    images. Networks had 1,024 excitatory and 256 inhibitory hidden neurons,
    with the E→I→E loop enabled for PING and disabled for COBA; input and readout
    weights were learned, while recurrent weights stayed fixed. We used the
    unregularized conditions from seeds 42–44 after #cfg.epochs training epochs,
    selecting the minimum-validation-loss epoch, not the maximum-accuracy epoch.
    During training, voltage-increment gradients were divided by 1,000 for PING
    and 1 for COBA; these are different trained recipes, not an isolated loop control.
    The wider retained activity-penalty comparison is described in
    #link("/exp025/")[the training activity-frontier study].

  + *Perturb emitted spikes.* Each trial lasted #cfg.t_ms ms at timestep
    #cfg.dt ms, with no warm-up or excluded interval. After membrane integration
    and spike emission, the intervention independently modified every E and I
    spike slot before rate recording, readout input and subsequent recurrent
    feedback (see Appendix).
    $ tilde(s)_j = s_j bb(1)[u_j >= p_"drop"] $ <eq-drop>
    $ tilde(s)_j = min(s_j + bb(1) lr([u_j < frac(r_"add" dot Delta t, 1000)]), 1) $ <eq-add>
    Here $s_j$ and $tilde(s)_j$ are the raw and modified binary spikes for a
    neuron–image–timestep slot $j$; $u_j$ is an independent uniform draw on
    $[0,1)$ and $bb(1)$ is an indicator. Deletion probability $p_"drop"$ ran
    from 0 to 1 in steps of 0.1; nominal insertion rate $r_"add"$ ran from
    0 to #add_max Hz in steps of 2, with $Delta t$ in ms.
    Insertion is a Bernoulli approximation to Poisson arrivals, capped at one
    spike per slot; collisions with existing spikes add nothing.

  + *Evaluate the selected networks.* Pixel intensity controlled Poisson input
    encoding, with a maximum rate of 25 Hz. An independent perturbation generator
    preserved the input-encoding stream across conditions. Each of 192
    model–seed–condition evaluations used #eval_n of the 10,000 official-test images;
    prediction selected the largest time-averaged output membrane potential.
    Accuracy and full-population E rates were aggregated over images and then
    over three seeds; curve envelopes show sample SD, not confidence intervals.
    Illustrative trials used seed 42 and test-image index 0, independently of
    digit-class selection.

  + *Normalize the nominal added rate.* We retained the original normalization:
    $ x_"add" = frac(100 r_"add", overline(r)_(E,"ref")) $ <eq-normalize>
    Here $x_"add"$ is the displayed percentage and $overline(r)_(E,"ref")$ is the
    model's three-seed mean E rate in Hz from final-epoch reference-image
    diagnostics. This differs in image and sometimes epoch from the selected
    classifiers' test evaluations; it is not a test-set firing-rate baseline.
    The full sampled range is displayed, without treating normalization as a
    matched-dose experiment.

  == Appendix. Within-step dynamics

  Conductances first decay and receive the previous timestep's modified spikes.
  For example, excitatory drive to E follows
  $ g_E^((t)) = d_A g_E^((t-1)) + tilde(bold(s))_E^((t-1)) W_(E E) + bold(s)_"in"^((t)) W_"in". $ <eq-conductance>
  Here $t$ indexes simulation steps, $g_E$ is the excitatory conductance vector,
  $d_A$ is its AMPA decay factor, $bold(s)_"in"$ is the input-spike vector,
  and $W_"in"$ and $W_(E E)$ are input and E→E conductance weights.
  The analogous E→I and I→E terms use $W_(E I)$ and $W_(I E)$, with inhibitory
  GABA decay for I→E. E→E weights were zero in these networks.

  Membrane integration and threshold/reset operations then emit raw E/I spikes.
  Deletion or insertion replaces those emitted spikes, which enter the recorded
  stream and the current readout update, and affect recurrent conductances at
  the next timestep. It does not undo a membrane reset or cause an inserted
  spike to trigger one. Thus deletion preserves connectivity but changes
  feedback activity; it does not leave loop dynamics intact. These interventions
  jointly affect feedback, spike counts and readout drive, so the observed
  asymmetry cannot by itself distinguish a dynamical activity floor from an
  informational requirement of the classifier.

  #reference-list((
    (text: [Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
      “Gradient-based learning applied to document recognition.”
      _Proceedings of the IEEE_ 86(11), 2278–2324 (1998).],
      doi: "10.1109/5.726791"),
  ))
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs,
    [How do trained COBA and PING networks respond to deletion and insertion of hidden spikes? Compare retained inference trials from validation-selected classifiers.],
    preview-figures, json-inputs: ("exp037",))
}
