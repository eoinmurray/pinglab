#import "contents.typ": contents-here, with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp042")

#let meta = (
  status: "◉ REVIEWED",
  writing_guide: "33.0.0",
  title: "Inhibitory Replay Perturbations Change Excitatory Firing",
  created_at: "2026-06-02T00:00:00Z",
  updated_at: "2026-09-01",
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

  #contents-here()

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
      *(A)* Independent-spike and *(B)* fixed-window group jitter for the
      illustrative first test presentation from the training replicate
      initialized with seed 42 at $sigma = #anchor-sigma$ ms; 200 of 1,024 E
      neurons and 64 of 256 I neurons are displayed, while annotations report
      full-population rates over the 200 ms presentation. *(C)* Independent-spike
      and *(D)* fixed-window sweep summaries show per-neuron E rate, realised
      I-spike rate and test-accuracy means across three training replicates.
      Retained SEM across training replicates is not displayed.
    ],
  )

  ]

  ]

  == Methods

  === Compute

  + *Models.* We reused three independently trained PING networks, initialized
    with training seeds 42–44.

  + *Trials.* We tested each network-condition combination on the same 1,000
    MNIST test images. The main figure represents 36,000 condition-level trials,
    or 33,000 distinct simulations because both arms shared the zero-jitter
    control.

  + *Simulation.* We presented each image for 200 ms using a 0.1 ms timestep.

  + *Independent jitter.* We gave every inhibitory spike an independent Gaussian
    time shift, using $sigma = 0, 0.5, 1, 2, 5, 9,$ and $14$ ms.

  + *Group jitter.* We divided the timeline into fixed 25 ms windows and gave
    every inhibitory spike within a window the same shift, using
    $sigma = 0, 1, 3, 7,$ and $14$ ms.

  + *Boundaries.* We reflected independent spikes that left the 200 ms interval
    back into it individually. For group jitter, we reflected the shared
    displacement once so that the group remained intact.

  + *Collisions.* We moved same-neuron spikes landing on the same timestep to the
    nearest unused in-range timestep. No spikes were removed.

  === Analyse

  #set enum(start: 8)

  + *Measurements.* We recorded accuracy and mean excitatory and inhibitory
    firing rates. Curves show the mean across the three training seeds.

  === Present

  #set enum(start: 9)

  + *Range restriction.* We retained larger jitter values as boundary-sensitivity
    results but excluded them from the main figure because reflection became a
    substantial part of the intervention.
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
