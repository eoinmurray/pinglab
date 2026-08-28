#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp075")

#let meta = (
  status: "[≡ TXT]",
  title: "A compiled graph learns",
  date: "2026-07-31",
  description: "A deliberately small MNIST run checks that an snnlang graph and training recipe can drive the existing tools/snnsim trainer without legacy structural flags.",
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

  This is an integration gate, not an MNIST benchmark. A Python `snnlang`
  program defines a #(r.config.n_e)-E/#(r.config.n_i)-I PING network and a standard
  training recipe. The compiler writes the graph and recipe into a portable
  bundle; `tools/snnsim train --bundle` authenticates both, maps the supported
  subset onto the existing optimised PyTorch trainer, and trains on only
  #r.config.max_samples MNIST examples for #r.config.epochs epochs. The
  experiment asks one intentionally unglamorous question: does the compiled
  network actually learn?

  #run-view("exp075", inputs)

  == Graph and training scope

  #data-image(data-file("exp075/network_graph.svg"), width: 100%)

  The graph fixes the PING topology, #r.config.dt_ms ms simulation step,
  mean-voltage classifier, initialisers, and trainable/frozen parameter scope.
  The recipe fixes unit-weight cross-entropy, AdamW with learning rate
  #r.config.learning_rate and weight decay #r.config.weight_decay, gradient
  clipping, and epoch count. The runner owns execution choices: the deterministic
  MNIST subset, batch size #r.config.batch_size, #r.config.t_ms ms presentation
  window, seed #r.config.seed, and artifact locations.

  The current backend is deliberately strict. It accepts trainable input and
  readout projections with frozen E↔I recurrence; unsupported objectives,
  parameter scopes, optimisers, or graph structures fail before training.

  #pagebreak()

  == Training trajectory

  #data-image(data-file("exp075/training_curves.png"), width: 100%)

  The #(r.config.train_count)-example training split and
  #(r.config.held_out_count)-example held-out split are tiny. Across the short
  run, training cross-entropy changed by
  #calc.round(r.trajectory.train_loss_change, digits: 3), held-out
  cross-entropy by #calc.round(r.trajectory.test_loss_change, digits: 3), and
  held-out accuracy by
  #calc.round(r.trajectory.accuracy_change_pct_points, digits: 2) percentage
  points. Best held-out accuracy was
  #calc.round(r.trajectory.best_accuracy_pct, digits: 2)% at epoch
  #r.trajectory.best_epoch. The trainer wrote both selected and final
  checkpoints in #calc.round(r.training.total_elapsed_s, digits: 1) seconds.

  == Conclusion

  #if r.trajectory.train_loss_change < 0 [
    The compiled graph trained: optimisation reduced the training objective and
    emitted ordinary `tools/snnsim` checkpoints and metrics. This does not establish
    competitive accuracy or good generalisation. It establishes the more basic
    vertical slice needed before migrating real experiments: Python graph →
    authenticated bundle → existing PyTorch training loop → inspectable evidence.
  ] else [
    The plumbing executed, but the training objective did not fall over this
    deliberately short run. The integration route works mechanically; this
    configuration is not yet evidence that useful optimisation occurs.
  ]
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can a compiled network graph learn through the graph-native training path? Inspect its topology and retained learning curves.],
    preview-figures, json-inputs: ("exp075",),
  )
}

#let meta = meta + (assets: input-assets("exp075", inputs))
#let body = with-datasets("exp075", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
