#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-card, with-result-sections
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": data-json, data-image, cite
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp044")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Firing Rate Across the Timestep Sweep",
  created_at: "2026-06-02T00:00:00Z",
  updated_at: "2026-09-09",
  description: "Compares final-epoch firing rate, classification accuracy and illustrative rasters across a twelvefold integration-timestep sweep.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp044",)
#let preview-figures = (
  (path: "exp044/dt_sweep.svg", label: "dt sweep"),
  (path: "exp044/raster_strip.png", label: "raster strip"),
  (path: "exp044/training_curves.svg", label: "training curves"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp044/numbers.json"))
#let c = run.config.training_contract.common
#let cfg = run.recipe
#let summary = run.summary
#let number(value) = calc.round(value, digits: 2)
#let n = cfg.seeds.len()

#let body = [
  #journal-abstract(body: [
  We asked how integration timestep affects the activity and accuracy of PING
  classifiers. We compared separately trained networks at matched training and evaluation
  timesteps, with fixed physical refractory periods, and evaluated their MNIST performance.

  Excitatory firing and accuracy varied across the sweep, showing that the
  timestep is part of the learned operating regime rather than a neutral solver
  setting. The comparison tests pipelines rather than holding weights fixed; the rasters do not
  establish invariance relative to the gamma period.
  ])

  == Results

  #with-result-sections[

  #result-card[
  === Timestep rate and accuracy

  Mean test accuracy ranged from #number(summary.acc_min_pct)% to
  #number(summary.acc_max_pct)%, spanning #number(summary.acc_span_pp) percentage
  points. Mean E rate ranged from #number(summary.e_rate_min_hz) to
  #number(summary.e_rate_max_hz) Hz. Accuracy persisted across the tested
  resolutions, while firing remained timestep-dependent (@fig:exp044-result-1).

  #figure(
    data-image(data-file("exp044/dt_sweep.svg"), width: 100%,
      alt: "Hidden excitatory firing rate and test accuracy against integration timestep, with uncertainty across training seeds."),
    caption: [Hidden E rate (black) and test accuracy (red) across the twelvefold
      timestep sweep. Markers show means over #n seeds; bars show ±1 standard
      error of the mean. Each network was evaluated on #cfg.evaluation_samples
      official-test images at its training timestep.],
  ) <fig:exp044-result-1>

  ]

  #result-card[
  === Timestep spike rasters

  #figure(
    data-image(data-file("exp044/raster_strip.png"), width: 100%,
      alt: "Single-trial excitatory and inhibitory spike rasters at five integration timesteps, plotted against physical time."),
    caption: [E (black) and I (red) rasters for the same official-test image,
      seed #cfg.raster.seed, at timesteps *(A–E)* 0.05, 0.1, 0.2, 0.3 and
      0.6 ms, respectively. Panels display
      #cfg.raster.n_e_plot E and #cfg.raster.n_i_plot I neurons over the first
      #cfg.raster.window_ms ms. These illustrative probes support visual cadence
      inspection, not a population estimate of gamma-period invariance.],
  ) <fig:exp044-result-2>

  ]

  #result-card[
  === Training-timestep response

  #figure(
    data-image(data-file("exp044/training_curves.svg"), width: 100%,
      alt: "Per-network validation accuracy and excitatory firing rate versus epoch, coloured by integration timestep."),
    caption: [Recorded training histories: *(A)* validation accuracy and *(B)* E
      rate, one line per timestep and seed. Each epoch averaged
      #c.validation_encoder_draws.count encoder draws per validation image.
      The final-epoch comparison is a finite-training snapshot, not an established
      fixed-point ceiling.],
  ) <fig:exp044-result-3>

  ]
  ]

  #journal-methods(
    orientation: [
  The audit reused separately trained networks and their learning histories,
  then measured endpoint dynamics at matched training and inference timesteps.
    ],
    compute: [
  + *Reuse the trained population.* One PING network was trained per
    $Delta t_"sim" in {0.05, 0.1, 0.2, 0.3, 0.6}$ ms and seed $in {42, 43, 44}$,
    giving fifteen networks. Each had #c.n_in inputs, #c.n_hidden excitatory
    neurons, #c.n_inh inhibitory neurons and #c.n_out class outputs.
    Network geometry, synaptic settings, readout and optimisation settings were
    checked for agreement across the comparison. E/I refractory holds were fixed
    at 1.2/0.6 ms: 24/12, 12/6, 6/3, 4/2 and 2/1 steps, respectively.
    Twelve networks were newly trained for this design; three 0.1-ms networks
    were reused unchanged after execution-equivalence checks.

  + *Keep data and nominal duration fixed.* The #(c.max_samples)-image MNIST
    training pool contained #c.dataset_split.optimizer_train_samples optimisation
    images and #c.dataset_split.validation_samples validation images; the official
    test partition was excluded from training. The nominal presentation duration
    was 200 ms; whole-step rounding gave 4,000, 2,000, 1,000, 666 and 333 steps.
    Realised $T_"present"$ was 199.8 ms at 0.3/0.6 ms and 200 ms otherwise. Image intensities drove Poisson input with peak rate
    #c.input_rate Hz.
    ],
    analyse: [
  #set enum(start: 3)

  + *Use the training endpoint.* Networks underwent #c.epochs epochs of
    surrogate-gradient training #cite(1), with batch size #c.batch_size and
    learning rate #c.lr. Class scores used the mean-membrane readout, and
    validation histories averaged #c.validation_encoder_draws.count encoder draws
    per image. The audit used the final epoch for rates, accuracy and rasters,
    rather than selecting the best validation epoch.
    ],
    present: [
  #set enum(start: 4)

  + *Measure held-out performance.* Each network was evaluated on the fixed
    #(cfg.evaluation_samples)-image subset of the official MNIST test partition,
    without retraining. Accuracy was the percentage of correctly classified
    images; population firing rate was total spikes divided by the number of
    evaluated images, population size and realised trial duration in seconds.
    Excitatory and inhibitory rates were recorded separately.
    ],
  )
  == Parameter summary

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t_"sim"$], [0.05–0.6 ms (swept)],
    [Presentation duration $T_"present"$], [#c.t_ms ms nominal; 199.8 ms at 0.3/0.6 ms],
    [Refractory hold, E/I], [1.2/0.6 ms at every timestep],
    [MNIST training pool], [#c.max_samples images: #c.dataset_split.optimizer_train_samples optimisation / #c.dataset_split.validation_samples validation],
    [Official-test evaluation], [#cfg.evaluation_samples images per network],
    [Epochs], [#c.epochs],
  )

  #run-view("exp044", inputs)

  #journal-references((
    (text: [E. O. Neftci, H. Mostafa, and F. Zenke.
      “Surrogate Gradient Learning in Spiking Neural Networks.”
      _IEEE Signal Processing Magazine_ 36(6), 51–63 (2019).],
      doi: "10.1109/MSP.2019.2931595"),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How sensitive are firing rate and classification accuracy to numerical timestep? Compare trained PING networks across integration timesteps at fixed physical refractory periods and nominal presentation duration.],
    preview-figures, json-inputs: ("exp044",),
  )
}

#let meta = meta + (assets: input-assets("exp044", inputs))
#let body = journal-article("exp044", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
