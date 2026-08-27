#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp076")

#let meta = (
  title: "A bundle checkpoint replays",
  date: "2026-08-02",
  description: "A small deterministic MNIST gate checks that snnlang bundle checkpoints replay through tools/snnsim and that the current bundle adapter is numerically equivalent to the matching legacy route.",
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

  This is an integration and equivalence experiment, not an accuracy benchmark.
  A Python `snnlang` program authors the current supported backend subset:
  MNIST input spikes, one #(r.config.n_e)-E/#(r.config.n_i)-I PING cell, and a
  mean-voltage classifier. The compiled bundle owns the graph topology,
  initialisers, unit-weight cross-entropy objective, AdamW optimiser, epoch
  count, and trainable/frozen parameter scope. The runner owns execution choices:
  #(r.config.max_samples) deterministic MNIST examples, batch size
  #r.config.batch_size, #r.config.t_ms ms presentation duration, #r.config.dt_ms
  ms timestep, and seed #r.config.seed.

  The selected bundle-trained checkpoint replayed through
  `tools/snnsim sim --bundle --infer --load-weights` at
  #pct(r.replay.selected_checkpoint_accuracy_pct), exactly matching the
  trainer's recorded best accuracy. The final checkpoint also replayed exactly.
  A focused deterministic unit gate separately shows exact bundle-vs-legacy
  equality for initial state dictionaries, forward logits, cross-entropy loss,
  gradients, and one AdamW step. This validates only the current MNIST PING +
  MeanVoltage adapter, not arbitrary `snnlang` graphs.

  == Lifecycle checked

  #data-image(data-file("exp076/lifecycle.svg"), width: 100%)

  The experiment stores the complete executable bundle at
  the selected run’s retained network bundle, including `graph.json`,
  `training.json`, and the manifest that authenticates both. Training writes a
  selected checkpoint and a final checkpoint. Replay then exercises four load
  paths: bundle checkpoint through bundle inference, final bundle checkpoint
  through bundle inference, selected bundle checkpoint through the equivalent
  explicit legacy route, and a separately trained legacy checkpoint through the
  bundle route.

  == Short training trajectory

  #data-image(data-file("exp076/training_curves.png"), width: 100%)

  The run used #r.config.train_count training examples and
  #r.config.held_out_count held-out examples from the deterministic split.
  Best held-out accuracy was #pct(r.trajectory.best_accuracy_pct) at epoch
  #r.trajectory.best_epoch; the final epoch was
  #pct(r.trajectory.final_accuracy_pct). The point is not that this tiny model is
  good at MNIST. The point is that an ordinary bundle-trained checkpoint is
  reloadable and produces reproducible held-out evaluation.

  == Replay and checkpoint compatibility

  #table(
    columns: (1.6fr, 1fr, 1fr),
    [Check], [Reference], [Fresh replay],
    [Selected checkpoint], [#pct(r.replay.trainer_best_accuracy_pct)], [#pct(r.replay.selected_checkpoint_accuracy_pct)],
    [Final checkpoint], [#pct(r.replay.trainer_final_epoch_accuracy_pct)], [#pct(r.replay.final_checkpoint_accuracy_pct)],
  )

  Selected replay differed from the trainer by
  #pp(r.replay.selected_delta_pct_points), and final replay differed by
  #pp(r.replay.final_delta_pct_points). The exact match is expected here because
  training-time evaluation and fresh inference use the same held-out split,
  presentation duration, seed, and deterministic Poisson evaluation generator.

  Structurally, every checked checkpoint loaded with no missing keys, no
  unexpected keys, and no shape mismatches. The state dictionary keys were
  `W_ff.0`, `W_ff.1`, `W_ee.1`, `W_ei.1`, `W_ie.1`, and `W_ii.1`. The equivalent
  legacy route loaded the bundle-produced selected checkpoint and reached
  #pct(r.compatibility.legacy_route_accuracy_on_bundle_checkpoint_pct). The
  bundle route loaded a checkpoint produced by a separate one-epoch legacy run
  and reached #pct(r.compatibility.bundle_route_accuracy_on_legacy_checkpoint_pct),
  matching that legacy checkpoint's own selected accuracy.

  == Deterministic one-step parity gate

  #table(
    columns: (1.4fr, 1fr),
    [Stage], [Result],
    [Initial state dictionary], [#r.parity.initial_state_dict],
    [Forward logits], [#r.parity.forward_logits],
    [Cross-entropy loss], [#r.parity.cross_entropy_loss],
    [Gradients], [#r.parity.gradients],
    [One AdamW step], [#r.parity.adamw_step],
    [Tolerance], [rtol #r.parity.tolerance.rtol, atol #r.parity.tolerance.atol],
  )

  The automated gate constructs the bundle and equivalent legacy configurations
  from their public descriptions, seeds both initialisations identically, feeds
  the exact same already-encoded spike tensor and labels, and compares tensors in
  order. If a future edit breaks parity, the test reports the first divergent
  stage and tensor name rather than a generic failure.

  The trainable parameter set is `W_ff.0` and `W_ff.1`: input and readout
  projections. The recurrent E/I matrices `W_ee.1`, `W_ei.1`, `W_ie.1`, and
  `W_ii.1` are frozen under the current bundle-training design. Unsupported
  objectives, parameter scopes, optimiser variants, graph shapes, and structural
  CLI overrides remain capability errors rather than silent fallbacks.

  == Runtime and scope

  The complete local experiment took #sec(r.runtime.total_elapsed_s), including
  bundle training, selected/final replay, legacy checkpoint loading, a tiny
  legacy checkpoint-production run, and bundle loading of that legacy checkpoint.
  Bundle training itself reported #sec(r.runtime.training_elapsed_s).

  This establishes a small but useful invariant: for the currently supported
  MNIST PING + MeanVoltage subset, `snnlang` bundle execution is not merely
  shape-compatible with the legacy route; it is numerically identical for the
  deterministic one-step training calculation and checkpoint-compatible across
  the bundle and legacy inference routes. It says nothing about unsupported
  graph topologies, objectives, recurrent plasticity scopes, custom readouts, or
  accelerator-specific execution.
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Does a trained bundle replay consistently from its checkpoint? Compare training, reload, inference, and deterministic one-step parity.],
    preview-figures, json-inputs: ("exp076",),
  )
}
