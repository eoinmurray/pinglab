#let meta = (
  title: "Compiling and executing bundles",
  date: "2026-08-14",
  description: "Validate a graph, write its portable bundle, select an executor, supply inputs, and retrieve named outputs and recordings.",
  collection: "snnlang",
  status: "draft",
  order: 4,
)

#let body = [
  == Compilation

  `snn.compile` validates a network, validates an optional training recipe, reports target capabilities, and returns a `Bundle`. A target asks for diagnostics but never changes the graph.

  ```python
  bundle = snn.compile(net, target="tools/snn")
  root = bundle.write("network.bundle", visualise=True)
  ```

  A bundle contains canonical `graph.json`, optional `training.json`, a digest-bearing `manifest.json`, copied logical assets, a readable summary, and optional diagrams. Loading verifies hashes and rejects missing or altered files.

  Dataset paths, output directories, accelerator caches, and experiment-specific analysis do not belong in the bundle.

  == Execution requests

  `ExecutionSpec` selects the legacy or graph executor explicitly and supplies inputs, seed, device, recording profile, checkpoint, runtime state, and execution options.

  The graph executor plans the complete topology before stepping. It validates dimensions, scheduling, polarity, delays, supplied inputs, outputs, and backend capabilities. Historical CLI commands continue to select the legacy executor by default.

  == Recordings and results

  Recording profiles are:

  - `full`, retaining every supported population trace;
  - `observables`, retaining only explicitly exposed signals; and
  - `none`, minimizing recording overhead.

  Results contain named outputs, recordings, parameters, final voltages, runtime state, timing, device, and recording metadata.

  Forward graph execution is implemented for dense COBA-LIF and leaky-integrator graphs with AMPA and GABA projections. Unsupported capabilities fail explicitly before execution.

  #link("/ar087/")[Next: Inputs, outputs, and readouts]
]
