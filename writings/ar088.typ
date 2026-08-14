#let meta = (
  title: "Training recipes and graph-native learning",
  date: "2026-08-14",
  description: "Declare standard objectives, parameter groups, optimization, regularization, and the boundary of current training support.",
  collection: "snnlang",
  status: "draft",
  order: 6,
)

#let body = [
  == Contents <contents>

  + #link(<developer-guide>)[Developer guide]
  + #link(<training-separation>)[Training is separate from the graph]
  + #link(<current-boundary>)[Current boundary]
  + #link(<api-reference>)[API reference]

  == Developer guide <developer-guide>

  === Training is separate from the graph <training-separation>

  `TrainSpec` describes standard learning choices without embedding a training loop in graph structure. It refers to named outputs and parameters in one network.

  ```python
  from tools.snnlang import training

  recipe = training.TrainSpec(
      objectives=[training.CrossEntropy(
          prediction=readout,
          target="digit",
      )],
      parameter_groups=[training.ParameterGroup(
          name="feedforward",
          parameters=readout.parameters,
          lr=1e-3,
      )],
      optimizer=training.AdamW(weight_decay=1e-4),
      epochs=20,
  )

  bundle = snn.compile(net, training=recipe)
  ```

  The recipe vocabulary includes objectives, named parameter groups, learning rates, frozen parameters, optimizer settings, regularizers, stop-gradient boundaries, epoch count, gradient clipping, and surrogate configuration.

  === Current boundary <current-boundary>

  Training recipes compile, and a narrow one-layer MNIST PING recipe can be translated into the legacy trainer. Graph-native training is not implemented. A graph training request fails explicitly and never falls back silently to legacy execution.

  The complete implementation also needs deterministic initialization, differentiable-reachability checks, recurrent trainability, variable-rate training, spike-budget regularization, selected and final checkpoints, optimizer-state replay, exact resume, and layered legacy-versus-graph conformance tests.

  == API reference <api-reference>

  === `TrainSpec`

  ```python
  training.TrainSpec(
      objectives,
      parameter_groups,
      optimizer,
      regularizers=(),
      stop_gradients=(),
      epochs=1,
      gradient_clip=None,
      surrogate=None,
  )
  ```

  The specification is data only. `compile` verifies referenced outputs, parameters, regularizer signals, and stop-gradient signals before writing `training.json`.

  === Objectives and parameter groups

  ```python
  training.CrossEntropy(*, prediction, target, weight=1.0) -> Objective
  training.ParameterGroup(parameters, name, lr, frozen=False)
  ```

  `prediction` accepts a signal-like object or identifier. It must resolve to a named graph output. `target` is the external target identifier. A parameter may appear in only one group.

  === Optimizer and regularization

  ```python
  training.AdamW(**config) -> Optimizer
  training.UpperRatePenalty(*, signal, threshold, strength) -> Regularizer
  training.StopGradient.at(signal) -> StopGradient
  ```

  Optimizer configuration is serialized without running an optimizer. `UpperRatePenalty` stores an upper-rate threshold and strength against a named signal. `StopGradient.at` records a graph boundary by signal identifier.

  === Execution support

  `snn.compile(net, training=recipe)` validates and serializes the recipe. `tools.snn.execution.train(ExecutionSpec(executor="graph", ...))` currently raises `NotImplementedError`. Use the explicit legacy executor only for the supported compatibility adapter.

  #link("/ar089/")[Next: Runtime state, checkpoints, and provenance]
]
