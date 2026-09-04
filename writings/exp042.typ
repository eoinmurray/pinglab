#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-card, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp042")

#let meta = (
  tags: ("reviewed", "v35.4.0"),
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

// Calculations remain lazy so missing inputs cannot become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp042/numbers.json"))
#let cfg = run.config
#let anchor-sigma = 14

#let body = [
  #journal-abstract(body: [
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
  ])

  == Results

  #with-result-sections[

  #result-card[
  === Replay perturbations diverge

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
  ) <fig:exp042-result-1>

  ]

  ]

  #journal-methods(
    compute: [
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
    ],
    analyse: [
  #set enum(start: 8)

  + *Measurements.* We recorded accuracy and mean excitatory and inhibitory
    firing rates. Curves show the mean across the three training seeds.
    ],
    present: [
  #set enum(start: 9)

  + *Range restriction.* We retained larger jitter values as boundary-sensitivity
    results but excluded them from the main figure because reflection became a
    substantial part of the intervention.
    ],
  )
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
#let body = journal-article("exp042", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
