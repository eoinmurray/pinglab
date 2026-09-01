#import "contents.typ": with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp075")

#let meta = (
  status: "[▦ DATA | v31.2.0]",
  title: "A compiled graph learns",
  updated_at: "2026-08-31T00:00:00Z",
  created_at: "2026-07-31T00:00:00Z",
  description: "A small compiled spiking classifier trains on MNIST with frozen recurrent weights.",
  collection: "snnlang-docs",
  order: 2,
)

#let inputs = ("exp075",)
#let preview-figures = (
  (path: "exp075/network_graph.svg", label: "network graph"),
  (path: "exp075/training_curves.png", label: "training curves"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp075/numbers.json"))

#let body = [
  == Abstract


  Asked whether a compiled graph can participate in gradient-based classifier
  training rather than only forward simulation. Trained an excitatory–inhibitory
  MNIST classifier through the compiled graph interface and selected a
  checkpoint using held-out validation loss.

  Training produced coherent loss and validation trajectories, showing that
  optimization traversed the compiled representation. Demonstrates the learning
  interface on a bounded example, not competitive classification performance or
  broad generalization.

  == Results

  #with-result-sections[

  #result-card[
  === Compiled classifier topology

  #figure(data-image(data-file("exp075/network_graph.svg"), width: 100%),
    caption: [Classifier with #r.config.n_e excitatory and #r.config.n_i
      inhibitory neurons. Input and readout projections were trainable;
      excitatory–inhibitory recurrence remained fixed.])

  ]

  #result-card[
  === Training loss and validation accuracy across epochs

  The loss-selected checkpoint came from epoch
  #(r.trajectory.selected_epoch); it need not be the epoch with maximum
  accuracy.

  #figure(data-image(data-file("exp075/training_curves.png"), width: 100%),
    caption: [Training and validation cross-entropy and validation accuracy
      over #r.config.epochs epochs. Validation values average
      #r.config.validation_encoder_draws.count fixed stochastic encodings.
      The dashed line marks ten-class chance accuracy.])

  ]
  ]

  == Methods

  We tested optimisation of a compiled network using a small, fixed
  handwritten-digit classification task.

  === Compute

  + *Prepare data and network.* We selected #r.config.max_samples images from
    the official MNIST training partition with a fixed subset seed, then made
    a stratified #r.config.train_count/#r.config.held_out_count training/validation
    split. The classifier contained #r.config.n_e excitatory and #r.config.n_i
    inhibitory neurons and a ten-class mean-voltage readout.
  + *Encode and train.* Pixel intensities controlled spike probability, with
    a maximum input rate of #r.config.input_rate_hz Hz. We simulated
    #r.config.t_ms ms per image at #r.config.dt_ms ms resolution and trained
    for #r.config.epochs epochs in batches of #r.config.batch_size, using
    seed #r.config.seed. AdamW#cite(1) minimised unit-weight cross-entropy with
    learning rate #r.config.learning_rate, weight decay #r.config.weight_decay
    and gradient-norm clipping at one; only input and readout weights changed.
  === Analyse

  #set enum(start: 3)

  + *Select the checkpoint.* After each epoch we evaluated the validation set
    using #r.config.validation_encoder_draws.count fixed encoder draws.
    Selection minimised mean validation cross-entropy, breaking ties by mean
    accuracy and then the earliest epoch. The official test partition did not
    participate in selection; both selected and final states were retained.

  === Present

  #set enum(start: 4)

  + *Display retained learning evidence.* We displayed the recorded learning
    curves and selected-versus-final checkpoint outputs with the validation
    selection rule stated above.

  #run-view("exp075", inputs)

  #reference-list(((text: [Ilya Loshchilov and Frank Hutter: _Decoupled Weight Decay Regularization_. ICLR, 2019.], doi: "10.48550/arXiv.1711.05101"),))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can a compiled network graph learn through the graph-native training path? Inspect its topology and recorded learning curves.],
    preview-figures, json-inputs: ("exp075",),
  )
}

#let meta = meta + (assets: input-assets("exp075", inputs))
#let body = with-datasets("exp075", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
