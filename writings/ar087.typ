#let meta = (
  title: "Inputs, outputs, and readouts",
  date: "2026-08-14",
  description: "Bind data to graph inputs, expose internal activity, and turn network signals into stable output values.",
  collection: "snnlang",
  status: "draft",
  order: 5,
)

#let body = [
  == Contents

  + #link("/ar087/#developer-guide")[Developer guide]
  + #link("/ar087/#outputs-and-observables")[Outputs and observables]
  + #link("/ar087/#standard-readouts")[Standard readouts]
  + #link("/ar087/#input-bindings")[Input bindings]
  + #link("/ar087/#api-reference")[API reference]

  == Developer guide

  === Outputs and observables

  An output is a named value returned by the graph. An observable exposes an internal signal for recording without changing the computation.

  ```python
  readout = snn.readouts.MeanVoltage(
      source=cell.E.spikes,
      classes=10,
      name="classifier",
  )
  net.output("class_logits", readout)
  net.expose(cell.E.spikes, cell.I.spikes, name="hidden")
  ```

  === Standard readouts

  The standard readouts are mean voltage, final voltage, spike count, duration-normalized spike rate, and cumulative potential. Spike-rate logits divide spike count by valid presentation duration, so changing duration does not silently rescale the classifier.

  All five standard readouts execute in the graph executor. Spike rate reports spikes per second from either an explicit duration in seconds or a `(time, batch)` valid-time mask; masked reductions reject empty windows rather than silently returning arbitrary values. The compiler infers shapes and units, checks linear readout parameter shapes, rejects malformed masks and ambiguous durations, and records operation requirements in the bundle capability manifest.

  === Input bindings

  The execution layer binds concrete data to graph inputs without placing datasets or stimuli in graph structure. Dense arrays and sparse event streams are implemented as separate named, data-only representations. Dense NPY replay binds to a graph with one input; dense NPZ arrays bind by input identifier.

  Before execution, the resolver requires exact input coverage, common time and batch axes, declared feature dimensions, finite numeric values, binary spike values, and boolean or zero/one masks. A mismatch names the offending input and fails before simulation.

  Event replay stores zero-based integer step, batch, and channel coordinates plus explicit step and batch counts. Coordinates must be ordered by step, batch, and channel. The resolver rejects duplicates, invalid graph input types, inconsistent durations, and out-of-bounds coordinates before materializing binary spikes.

  Every replay emits a versioned execution protocol containing the representation, source-file digest and array keys, resolved shape and data type, signal type and unit, dataset identity and split when supplied, sample cap, batch size, shuffle behavior, timestep, duration, masks, seeds, and representation-specific resolution semantics. The command-line contract accepts `--input-file` or `--event-file`, dataset metadata, and the explicit `--input-shuffle` or `--no-input-shuffle` pair.

  Fixed-rate and categorical variable-rate Poisson encoding belong to execution protocol rather than graph topology.

  Dense-array, valid-time-mask, and sparse event-stream bindings work today. Portable dataset-loader and encoder bindings are not implemented.

  == API reference

  === Outputs and observables

  ```python
  net.output(name, signal) -> SignalLike
  net.expose(*signals, name=None) -> None
  ```

  An output maps a public name to one graph signal. `expose` records one or more internal signals. A single exposed signal uses `name` directly; multiple signals use `<name>_0`, `<name>_1`, and subsequent indices.

  === Standard readouts

  ```python
  snn.readouts.MeanVoltage(*, source, classes, name, tau=20 * snn.ms, weight=Normal(1.0, 0.1))
  snn.readouts.FinalVoltage(*, source, classes, name)
  snn.readouts.SpikeCount(*, source, classes, name)
  snn.readouts.SpikeRate(*, source, classes, name, duration=None, mask=None, window="full")
  snn.readouts.CumulativePotential(*, source, classes, name)
  ```

  Every constructor returns a `Readout`, which delegates signal attributes and exposes its parameter identifiers. `classes` is the output width. `SpikeRate` requires either `duration` in seconds or a `(time, batch)` mask. The only implemented window is `"full"`.

  === Dense bindings

  ```python
  DenseArrayBinding(input_id, value, source={})
  load_dense_array_bindings(path, graph) -> tuple[DenseArrayBinding, ...]
  resolve_dense_array_bindings(
      graph, *, bindings=(), inputs=None,
      device="cpu", seed=0, protocol=None,
  ) -> ResolvedDenseInputs
  ```

  `value` is a PyTorch tensor. `source` is JSON-serializable provenance. NPY binds only to a graph with one input; NPZ keys bind by graph input identifier. Direct `ExecutionSpec.inputs` values are converted to in-memory bindings and validated through the same resolver.

  The resolved protocol uses schemas `tools/snn.dense-array-binding/v1` and `tools/snn.execution-protocol/v1`. Reserved protocol fields cannot be overridden by caller metadata.

  === Event-stream bindings

  ```python
  EventStreamBinding(
      input_id, steps, batches, channels,
      steps_count, batch_size, source={},
  )
  load_event_stream_bindings(path, graph) -> tuple[EventStreamBinding, ...]
  resolve_event_stream_bindings(
      graph, *, bindings,
      device="cpu", seed=0, protocol=None,
  ) -> ResolvedDenseInputs
  resolve_input_bindings(
      graph, *, dense_bindings=(), event_bindings=(), inputs=None,
      device="cpu", seed=0, protocol=None,
  ) -> ResolvedDenseInputs
  ```

  Event bindings support graph inputs with shape `("time", "batch", channels)` and signal type `"spikes"`. Coordinate arrays are one-dimensional integer tensors of equal length. Step coordinates lie in `[0, steps_count)`, batch coordinates in `[0, batch_size)`, and channel coordinates in the declared graph width.

  A single-input NPZ uses `steps`, `batches`, `channels`, `steps_count`, and `batch_size`. Multi-input files prefix each key with the input identifier and a period. Typed requests may combine event-stream spike inputs with dense masks or continuous inputs when their time and batch axes agree.

  The event schema is `tools/snn.event-stream-binding/v1`; a mixed request uses `tools/snn.mixed-input-bindings/v1`. The protocol records ordering, duplicate rejection, binary materialization, event counts, masks, and file provenance.

  #link("/ar088/")[Next: Training recipes and graph-native learning]
]
