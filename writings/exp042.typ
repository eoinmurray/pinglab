#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp042")

#let meta = (
  status: "[▦ DATA | v30.0.0]",
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
#set math.equation(numbering: "(1)")
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

  + *Starting classifiers.* We reused three PING training replicates from
    #link("/exp022/")[experiment 022], initialized with seeds 42–44. We selected
    their final checkpoints because this experiment measures endpoint dynamics,
    and we kept every weight fixed. Each classifier contained 1,024 E neurons,
    256 I neurons and ten output LIF neurons. Class scores were the output
    neurons' mean pre-reset voltages across each 200 ms presentation; neuronal
    and readout states were initialized anew for every presentation.

  + *Test data.* The retained campaign tested every condition on the same
    #cfg.evaluation_samples_per_condition images from the official MNIST test
    set. Each image was presented for 200 ms using a simulation timestep of
    $Delta t_"sim" = 0.1$ ms.

  + *Record inhibition.* The retained campaign first ran each image normally and
    recorded when every inhibitory neuron fired. It then ran the network again,
    replacing its natural inhibitory spikes with altered versions of this
    recording.

  + *Move groups of spikes together.* We divided each recording into fixed time
    windows based on the reference frequency. Their length in simulation steps
    was

    $ L = max(1, op("round")(1000 / (f_"ref" Delta t_"sim"))), $

    where $L$ is the window length, $f_"ref" = #cfg.f_gamma_reference_hz$ Hz is
    the reused three-training-replicate mean gamma estimate at the shared
    operating point and $Delta t_"sim"$ is the simulation timestep in
    milliseconds. For each replicate, the frequency estimate was the
    parabolically interpolated 5–150 Hz peak of its trial-mean Welch spectrum.
    Every inhibitory spike originating in the same window received the same
    Gaussian random time shift, with standard deviation $sigma_"jitter"$. These
    were fixed clock windows, not gamma cycles detected from the recorded spikes.

  + *Move spikes independently.* In the comparison intervention, every recorded
    inhibitory spike received its own random shift

    $ Delta_j tilde cal(N)(0, sigma_"jitter"^2), $

    where $Delta_j$ is the shift applied to spike $j$ and $sigma_"jitter"$
    controls the amount of timing disruption, both in milliseconds.

  + *Keep spikes inside the presentation.* We rounded every shift to the 0.1 ms
    simulation grid and assigned a spike at step $k_j$ to

    $ k'_j = op("clamp")(k_j + op("round")(Delta_j / Delta t_"sim"), 0, N_t - 1), $

    where $k_j$ and $k'_j$ are the original and shifted simulation steps,
    $Delta_j$ is the shift assigned under either intervention and $N_t$ is the
    total number of steps. This prevented spikes from moving outside the 200 ms
    presentation. Two spikes from the same neuron that landed on the same step
    became one spike, so realised inhibitory delivery could change.

  + *Jitter sweeps.* We tested group shifts at
    $sigma_"jitter" = 0, 1, 3, 7, 14, 21, 28, 42, 60,$ and $100$ ms, and
    independent-spike jitter at $0, 0.5, 1, 2, 5, 9, 14, 21,$ and $50$ ms.
    The zero-jitter arms were identical; all other shifts were reproducible.

  === Analyse

  #set enum(start: 8)

  + *Responses.* For each training replicate and jitter level, we measured test
    accuracy and final-population mean per-neuron E and I firing rates across all
    1,000 test presentations of 200 ms each. Realised I rate counts delivered
    spikes, not inhibitory conductance.

  + *Aggregation and raster.* At each jitter level, we calculated the mean and
    SEM across three training replicates, where $"SEM" = s / sqrt(3)$ and $s$ is
    the sample standard deviation. The raster uses the first test presentation
    from the seed-42 replicate at $sigma_"jitter" = #anchor-sigma$ ms. We used
    full-population rates before reproducibly selecting 200 E and 64 I neurons.

  === Present

  #set enum(start: 10)

  + *Expose retained evidence.* The upper panels show the illustrative rasters;
    the lower panels show complete condition means on symmetric-logarithmic
    jitter axes. SEM values remain in the retained analysis table but are not
    plotted. This presentation re-rendered retained campaign measurements
    without recomputing them.
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
#let body = with-contents(body)
