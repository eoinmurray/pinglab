#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp044")

#let meta = (
  status: "[▦ DATA]",
  title: "Firing Rate Across the Timestep Sweep",
  date: "2026-06-02",
  updated_at: "2026-08-27",
  description: "Compares final-epoch firing rate, classification accuracy and illustrative rasters across a twentyfold integration-timestep sweep.",
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
  == Abstract

  Final-epoch mean excitatory rates ranged from #number(summary.e_rate_min_hz) to
  #number(summary.e_rate_max_hz) Hz across a twentyfold integration-timestep sweep;
  mean test accuracy ranged from #number(summary.acc_min_pct) to
  #number(summary.acc_max_pct)%. We reused fifteen trained PING networks, with
  three seeds at each timestep and 200 ms presentations, then evaluated
  #cfg.evaluation_samples held-out MNIST images per network. Training and inference
  used matched timesteps. The comparison therefore tests the complete trained
  pipeline, not one network evaluated at different timesteps. Single-trial rasters
  permit qualitative inspection of burst cadence; they do not establish gamma-period invariance.

  #run-view("exp044", inputs)

  == Results: Mean firing spans #number(summary.e_rate_min_hz)–#number(summary.e_rate_max_hz) Hz; accuracy spans #number(summary.acc_min_pct)–#number(summary.acc_max_pct)%

  === Firing rate and classification accuracy

  #figure(
    data-image(data-file("exp044/dt_sweep.svg"), width: 100%,
      alt: "Hidden excitatory firing rate and test accuracy against integration timestep, with uncertainty across training seeds."),
    caption: [Hidden E rate (black) and test accuracy (red) across the twentyfold
      timestep sweep. Markers show means over #n seeds; bars show ±1 standard
      error of the mean. Each network was evaluated on #cfg.evaluation_samples
      official-test images at its training timestep.],
  )

  === Single-trial burst cadence

  #figure(
    data-image(data-file("exp044/raster_strip.png"), width: 100%,
      alt: "Single-trial excitatory and inhibitory spike rasters at five integration timesteps, plotted against physical time."),
    caption: [E (black) and I (red) rasters for the same official-test image,
      seed #cfg.raster.seed, at each timestep. Panels display
      #cfg.raster.n_e_plot E and #cfg.raster.n_i_plot I neurons over the first
      #cfg.raster.window_ms ms. These illustrative probes support visual cadence
      inspection, not a population estimate of gamma-period invariance.],
  )

  === Training trajectories

  #figure(
    data-image(data-file("exp044/training_curves.svg"), width: 100%,
      alt: "Per-network validation accuracy and excitatory firing rate versus epoch, coloured by integration timestep."),
    caption: [Retained training histories: validation accuracy (top) and E rate
      (bottom), one line per timestep and seed. Each epoch averages
      #c.validation_encoder_draws.count encoder draws per validation image.
      The final-epoch comparison is a finite-training snapshot, not an established
      fixed-point ceiling.],
  )

  == Methods

  The audit reused separately trained networks and their learning histories,
  then measured endpoint dynamics at matched training and inference timesteps.

  + *Reuse the trained population.* One PING network was trained per
    $Delta t in {0.05, 0.1, 0.25, 0.5, 1.0}$ ms and seed $in {42, 43, 44}$,
    giving fifteen networks. Each had #c.n_in inputs, #c.n_hidden excitatory
    neurons, #c.n_inh inhibitory neurons and #c.n_out class outputs.
    Network geometry, synaptic settings, readout and optimisation settings were
    checked for agreement across the comparison.

  + *Keep data and physical duration fixed.* The #(c.max_samples)-image MNIST
    training pool contained #c.dataset_split.optimizer_train_samples optimisation
    images and #c.dataset_split.validation_samples validation images; the official
    test partition was excluded from training. Each presentation lasted
    $T = 200$ ms, so the integration timestep $Delta t$ changed the step count
    from 4,000 to 200. Image intensities drove Poisson input with peak rate
    #c.input_rate Hz.

  + *Retain the training endpoint.* Networks underwent #c.epochs epochs of
    surrogate-gradient training #cite(1), with batch size #c.batch_size and
    learning rate #c.lr. Class scores used the mean-membrane readout, and
    validation histories averaged #c.validation_encoder_draws.count encoder draws
    per image. The audit used the final epoch for rates, accuracy and rasters,
    rather than selecting the best validation epoch.

  + *Measure held-out performance.* Each network was evaluated on the fixed
    #(cfg.evaluation_samples)-image subset of the official MNIST test partition,
    without retraining. Accuracy was the percentage of correctly classified
    images; population firing rate was total spikes divided by the number of
    evaluated images, population size and trial duration in seconds.
    Excitatory and inhibitory rates were retained separately.

  == Parameter summary

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.05–1.0 ms (swept)],
    [Trial duration $T$], [#c.t_ms ms],
    [MNIST training pool], [#c.max_samples images: #c.dataset_split.optimizer_train_samples optimisation / #c.dataset_split.validation_samples validation],
    [Official-test evaluation], [#cfg.evaluation_samples images per network],
    [Epochs], [#c.epochs],
  )

  #reference-list((
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
    [How sensitive are firing rate and classification accuracy to numerical timestep? Compare trained PING networks across integration timesteps at fixed physical presentation duration.],
    preview-figures, json-inputs: ("exp044",),
  )
}

#let meta = meta + (assets: input-assets("exp044", inputs))
#let body = with-datasets("exp044", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
