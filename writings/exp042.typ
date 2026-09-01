#import "contents.typ": with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp042")

#let meta = (
  status: "[▦ DATA | v31.1.0]",
  title: "Inhibitory Replay Perturbations Change Excitatory Firing",
  created_at: "2026-06-02T00:00:00Z",
  updated_at: "2026-08-31T18:23:16Z",
  description: "Recorded inhibitory spike streams produced contrasting excitatory responses under independent-spike and fixed-window replay perturbations, but the experiment does not isolate synchrony.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp042",)
#let preview-figures = (
  (path: "exp042/rhythm_compound.png", label: "inhibitory replay perturbations"),
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
  } else {
    body
  }
}

// Calculations remain lazy so missing inputs cannot become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp042/numbers.json"))
#let cfg = run.config
#let anchor-sigma = 14

#let body = [
  == Abstract

  We asked how perturbing inhibitory spike timing changes trained PING
  classifiers. We reused three frozen MNIST classifiers and their retained
  inhibitory-spike recordings.

  We replayed each stream after either shifting every spike independently or
  shifting all spikes originating in the same fixed clock window together.
  Independent-spike jitter nearly silenced excitatory neurons, whereas
  fixed-window shifts increased their firing.

  Accuracy declined under both perturbations. Because rounding, boundary
  clamping and event collisions also changed realised inhibitory delivery,
  these measurements do not isolate synchrony or a gamma-specific mechanism.

  == Results

  #with-result-sections[

  #result-card-style

  #result-card[
  === Excitatory responses diverge across replay perturbations

  #figure(
    data-image(
      data-file("exp042/rhythm_compound.png"),
      width: 100%,
      alt: "Independent-spike jitter suppresses excitatory firing, whereas fixed-window group shifts increase it. Realised inhibitory spike rate changes across both sweeps.",
    ),
    caption: [
      Independent-spike and fixed-window group jitter. Top: the illustrative
      first test presentation from the training replicate initialized with seed
      42 at $sigma = #anchor-sigma$ ms; 200 of 1,024 E neurons and 64 of 256 I
      neurons are displayed, while annotations report full-population rates over
      the 200 ms presentation. Bottom: per-neuron E-rate, realised I-spike-rate
      and test-accuracy means across three training replicates over the complete
      sweeps. Retained SEM across training replicates is not displayed.
    ],
  )

  ]

  ]

  == Methods

  === Compute

  + We reused three frozen PING classifiers from experiment 022.

  + We tested every condition on the same 1,000 MNIST images.

  + We recorded each network’s inhibitory spikes and replayed altered versions.

  + Spikes from the same fixed time window received one shared random shift.

  + Alternatively, every inhibitory spike received its own random shift.

  + Shifted spikes were kept within the presentation. Colliding spikes merged,
    sometimes reducing inhibitory delivery.

  + We tested progressively larger shifts, including an unchanged condition.

  === Analyse

  #set enum(start: 8)

  + We measured accuracy and population firing rates.

  + We averaged results across the three classifiers and reproducibly selected
    one example raster.

  === Present

  #set enum(start: 10)

  + We displayed the example rasters and complete condition means without
    rerunning the experiment.
]
#body
  #run-view("exp042", inputs)
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file,
    inputs,
    [How do frozen classifiers respond when recorded inhibitory spikes are perturbed before replay?],
    preview-figures,
    json-inputs: ("exp042",),
  )
}

#let meta = meta + (assets: input-assets("exp042", inputs))
#let body = with-datasets("exp042", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
