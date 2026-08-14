#let meta = (
  title: "Compatibility, status, and extension",
  date: "2026-08-14",
  description: "Understand the legacy compatibility contract, current capability boundary, and rules for extending snnlang safely.",
  collection: "snnlang",
  status: "draft",
  order: 8,
)

#let body = [
  == Contents

  + #link("/ar090/#developer-guide")[Developer guide]
  + #link("/ar090/#compatibility")[Compatibility]
  + #link("/ar090/#current-support")[Current support]
  + #link("/ar090/#what-remains")[What remains]
  + #link("/ar090/#extending-snnlang")[Extending snnlang]
  + #link("/ar090/#api-reference")[API reference]

  == Developer guide

  === Compatibility

  _snnlang_ is additive. Historical commands and experiments use the legacy executor unless they explicitly select the graph executor. Bundles remain data-only, and _tools/snn_ does not import the authoring package when replaying them.

  Migration requires layered comparisons of structure, parameters, initialization, forward state, gradients, optimizer updates, checkpoints, resumed trajectories, learning curves, downstream measurements, and publication outputs. A successful smoke test does not establish equivalence.

  #link("https://github.com/eoinmurray/pinglab/issues/73")[Issue 73] is the active implementation and migration checklist. The independent exp022 gold-star campaign remains outside this documentation work.

  === Current support

  #table(
    columns: (1.3fr, 1fr, 2.2fr),
    align: (left, left, left),
    [*Area*], [*Status*], [*Current boundary*],
    [Graph authoring], [Implemented], [Typed populations, inputs, parameters, projections, groups, outputs, and observables.],
    [Bundles], [Implemented], [Canonical JSON, hashes, manifests, assets, reports, and diagrams.],
    [Graph simulation], [Implemented], [Dense COBA-LIF and leaky-integrator graphs with AMPA/GABA projections.],
    [Arbitrary topology], [Implemented], [Named feedforward, recurrent, and feedback projections with integral delays.],
    [Runtime continuation], [Implemented], [Validated save, load, and continuation of dynamic graph state.],
    [Readouts], [Implemented], [Mean voltage, final voltage, spike count, spike rate, cumulative potential, and valid-time masks execute through graph operations.],
    [Input bindings], [Partial], [Resolved dense-array, valid-time-mask, and sparse event-stream bindings emit a versioned execution protocol; portable dataset loaders and encoders are pending.],
    [Training recipes], [Partial], [The schema and narrow legacy adapter exist.],
    [Native training], [Not implemented], [Requires gradients, optimizer state, resume, checkpoints, and parity gates.],
    [Collection migration], [Not implemented], [Requires the complete issue 73 equivalence process.],
  )

  === What remains

  The following work is *not implemented*. #link("https://github.com/eoinmurray/pinglab/issues/73")[Issue 73] tracks it as an active checklist.

  ==== Complete graph and protocol vocabulary

  - Add portable dataset-loader and encoder bindings. Fixed and categorical variable-rate Poisson protocols are implemented.
  - Complete spike-budget regularization and variable-duration training vocabulary. Disabled loops, initializer metadata, parameter-group trainability, group-specific learning rates, surrogate gradients, backward dampening, and variable graph timesteps are implemented.
  - Record dataset identity, split, sample cap, batch size, shuffle behavior, and all stochastic seeds as execution protocol.

  ==== Graph-native training and checkpoints

  - Train the supported recurrent SNN subset through the graph executor.
  - Define stable parameter names, trainable and frozen sets, optimizer state, checkpoint schema, and an explicit legacy-to-graph parameter map.
  - Save selected and final checkpoints, reject partial mappings, and resume without changing data order, optimizer state, or stochastic streams.
  - Validate COBA, PING, trainable recurrence, fine timesteps, variable rates, MNIST, SHD, and deeper trained graphs.

  ==== Inference and interventions

  - Override inference duration, input rate, timestep, and supported projection strengths through an explicit execution contract.
  - Provide hidden-spike deletion and Poisson-addition interventions without reaching into backend internals.
  - Stabilize names and shapes for population spikes, membrane traces, rates, logits, accuracy, and rasters.
  - Preserve seed-labelled caches and fail closed when a compiled graph cannot support an intervention.

  ==== Artifacts and campaigns

  - Version the metric, checkpoint, and provenance schemas needed by collection campaigns.
  - Integrate graph and training digests with runstore validation, resumption, promotion, archive, and restore.
  - Select `legacy` or `graph` explicitly while keeping campaign ownership and publication behavior unchanged.

  ==== Conformance and migration

  - Compare topology, parameter tensors, initialized state, forward traces, loss, gradients, optimizer updates, checkpoint interchange, exact resume, learning trajectories, interventions, and aggregations.
  - Define exact fields and numerical tolerances before viewing the final comparison.
  - Run CPU and publication-accelerator conformance cases and record any limits on numerical determinism.
  - After the independent legacy gold-star campaign is complete, run the full snnlang campaign under the frozen scientific protocol and compare every result, uncertainty band, claim, and figure.
  - Retain both immutable campaigns and require deliberate review before changing the default executor or migrating the publication collection.

  === Extending snnlang

  Add a feature when it represents reusable computation or execution. Keep it in an experiment runner when its meaning depends on one hypothesis or figure.

  A capability should include a clear authoring API, versioned serialization, compiler validation, precise diagnostics, a hand-checkable execution fixture, explicit state and provenance semantics, and compatibility tests. Add an executable example once the feature is mature enough to teach.

  Avoid opaque callbacks, access to backend internals, positional checkpoint guesses, and silent fallbacks.

  == API reference

  === Validation and diagnostics

  ```python
  snn.validate_graph(graph) -> ValidationResult
  result.raise_for_errors() -> None
  ```

  `ValidationResult` exposes `.diagnostics`, `.errors`, and `.warnings`. Each `Diagnostic` contains `severity`, `code`, `message`, and optional `subject`; `.line()` renders a stable one-line form.

  Compilation raises graph and training errors. Target capability gaps remain diagnostics so a bundle can still describe computation unsupported by the selected backend.

  === Executor capability inspection

  ```python
  graph_capability_issues(graph) -> list[CapabilityIssue]
  plan_graph(graph) -> GraphPlan
  ```

  `CapabilityIssue` identifies the graph element, missing capability, and message. `plan_graph` performs execution-specific topology, shape, polarity, scheduling, and integral-delay validation before constructing a `GraphPlan`.

  `GRAPH_CAPABILITIES_V1` is the current machine-readable executor boundary. It lists supported neuron, synapse, operation, connection, recording, delay, and training capabilities.

  === Code map

  - `tools/snnlang/core.py`: authoring objects and primitive specifications.
  - `components.py`: reusable graph-building components.
  - `readouts.py` and `ops.py`: standard operations.
  - `training.py`: training recipes.
  - `compiler.py`: validation and bundle writing.
  - `visualize.py`: bundle reports.
  - `tools/snn/execution.py`: graph planning and execution.
  - `tools/snn/bundle.py`: the narrow legacy adapter.
  - `tools/snnlang/tests` and `tools/snn/tests/test_execution.py`: focused conformance fixtures.
]
