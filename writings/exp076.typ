#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp076")

#let meta = (
  status: "[▦ DATA]",
  title: "A bundle checkpoint replays",
  updated_at: "2026-08-28",
  date: "2026-08-02",
  description: "Checkpoint interchange and deterministic one-step equivalence in a spiking classifier.",
  collection: "snnlang-docs",
  order: 3,
)

#let inputs = ("exp076",)
#let preview-figures = (
  (path: "exp076/lifecycle.svg", label: "lifecycle"),
  (path: "exp076/training_curves.png", label: "training curves"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp076/numbers.json"))

#let pct(x) = str(calc.round(x, digits: 3)) + "%"
#let pp(x) = str(calc.round(x, digits: 4)) + " pp"
#let sec(x) = str(calc.round(x, digits: 2)) + " s"

#let body = [
  == Abstract

  I trained a compiled #(r.config.n_e)-excitatory/#(r.config.n_i)-inhibitory
  classifier and exercised checkpoint loading through compiled and explicit
  network descriptions. The selected checkpoint achieved
  #pct(r.replay.selected_checkpoint_accuracy_pct) on
  #r.replay.evaluation_samples official-test images, compared with
  #pct(r.replay.trainer_best_accuracy_pct) during validation. These evaluations
  used different partitions and encoding aggregation, so their difference is
  not a numerical replay error. A separate deterministic one-step gate passed
  exact comparisons of initial parameters, logits, loss, gradients and an
  AdamW update. The result supports checkpoint compatibility for this network
  family, not arbitrary graph equivalence.

  #run-view("exp076", inputs)

  == Results

  === Training and replay protocol

  #figure(data-image(data-file("exp076/lifecycle.svg"), width: 100%),
    caption: [Protocol schematic for compilation, training, checkpoint reload
      and a separate one-step equivalence test; the arrows describe operations,
      not measurements.])

  #figure(data-image(data-file("exp076/training_curves.png"), width: 100%),
    caption: [Training and validation trajectories across #r.config.epochs
      epochs. Validation averages #r.config.validation_encoder_draws.count
      encoder draws on #r.config.held_out_count images. The selected checkpoint
      came from epoch #r.trajectory.selected_epoch.])

  === Checkpoint evaluations

  #figure(table(columns: (1.6fr, 1fr, 1fr),
    [Checkpoint], [Validation], [Official-test replay],
    [Selected], [#pct(r.replay.trainer_best_accuracy_pct)], [#pct(r.replay.selected_checkpoint_accuracy_pct)],
    [Final], [#pct(r.replay.trainer_final_epoch_accuracy_pct)], [#pct(r.replay.final_checkpoint_accuracy_pct)],
  ), kind: table,
    caption: [Validation used #r.config.held_out_count images and multiple
      encoder draws; replay used #r.replay.evaluation_samples official-test
      images and one fixed encoding. The selected bundle checkpoint loaded
      through the explicit route achieved
      #pct(r.compatibility.legacy_route_accuracy_on_bundle_checkpoint_pct).
      A separately trained explicit-network checkpoint loaded through the
      bundle route achieved
      #pct(r.compatibility.bundle_route_accuracy_on_legacy_checkpoint_pct).])

  === Deterministic one-step equivalence

  #figure(table(columns: (1.6fr, 1fr),
    [Comparison], [Result],
    [Initial parameters], [#r.parity.initial_state_dict],
    [Forward logits], [#r.parity.forward_logits],
    [Cross-entropy], [#r.parity.cross_entropy_loss],
    [Gradients], [#r.parity.gradients],
    [AdamW update], [#r.parity.adamw_step],
  ), kind: table,
    caption: [Separately executed deterministic test with identical encoded
      spikes, labels and seeded initialisation. Relative and absolute
      tolerances were both zero. This gate does not compare accuracies obtained
      on different datasets.])

  == Methods

  I tested checkpoint interchange and numerical equivalence as separate
  properties of the same supported classifier family.

  + *Train and select states.* I split #r.config.max_samples MNIST training
    images into #r.config.train_count optimisation and
    #r.config.held_out_count validation examples. The network used a
    mean-voltage readout, #r.config.dt_ms ms timestep and #r.config.t_ms ms
    duration. I trained input and readout weights for #r.config.epochs epochs
    with AdamW#cite(1), learning rate #r.config.learning_rate, weight decay
    #r.config.weight_decay and gradient-norm clipping at one, while recurrent
    weights remained fixed. Selection minimised cross-entropy averaged over
    #r.config.validation_encoder_draws.count validation encodings, with accuracy
    and earliest epoch as tie-breakers.
  + *Reload and evaluate.* I evaluated selected and final states on
    #r.replay.evaluation_samples sampled official-test images using one fixed
    spike encoding. I also loaded the selected compiled-network state through
    the explicit network route, then trained an explicit network for one epoch
    with a separate seed and loaded that state through the compiled route.
    Checkpoint inspection compared parameter names and shapes; accuracy
    comparisons retained their dataset and encoding context.
  + *Test one-step equality.* In a separate deterministic fixture I seeded
    both descriptions identically and supplied the same encoded spikes and
    labels. I compared initial parameters, forward outputs, cross-entropy,
    gradients and one AdamW update with zero numerical tolerance. This bounded
    test isolates implementation equivalence from stochastic evaluation.

  #reference-list(((text: [Ilya Loshchilov and Frank Hutter: _Decoupled Weight Decay Regularization_. ICLR, 2019.], doi: "10.48550/arXiv.1711.05101"),))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Does a trained bundle replay consistently from its checkpoint? Compare training, reload, inference, and deterministic one-step parity.],
    preview-figures, json-inputs: ("exp076",),
  )
}

#let meta = meta + (assets: input-assets("exp076", inputs))
#let body = with-datasets("exp076", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
