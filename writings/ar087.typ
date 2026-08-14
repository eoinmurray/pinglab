#let meta = (
  title: "Inputs, outputs, and readouts",
  date: "2026-08-14",
  description: "Bind data to graph inputs, expose internal activity, and turn network signals into stable output values.",
  collection: "snnlang",
  status: "draft",
  order: 5,
)

#let body = [
  == Outputs and observables

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

  == Standard readouts

  The intended standard readouts are mean voltage, final voltage, spike count, duration-normalized spike rate, and cumulative potential. Spike-rate logits divide spike count by valid presentation duration, so changing duration does not silently rescale the classifier.

  Mean voltage is implemented in the graph executor. Final voltage, spike count, spike rate, masks, and cumulative potential exist in parts of the authoring API but are not implemented end to end.

  == Input bindings

  The complete execution layer will bind concrete data to graph inputs. A resolved binding should state representation, dataset identity, split, sample cap, batch size, shuffle behavior, duration, masks, and seeds.

  Fixed-rate and categorical variable-rate Poisson encoding belong to execution protocol rather than graph topology. Dense arrays and event streams need separate bindings because their storage and timing semantics differ.

  Explicit dense input tensors and legacy CLI dataset handling work today. Portable dataset, event-stream, mask, and encoder bindings are not implemented.

  #link("/ar088/")[Next: Training recipes and graph-native learning]
]
