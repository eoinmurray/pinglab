# snnlang

`snnlang` is a standalone Python authoring library for graph-shaped spiking
circuits. It validates a network and optional standard training recipe, then
writes a deterministic, data-only bundle. It does not simulate or train.

```python
from tools import snnlang as snn

net = snn.Network("small_ping")
events = net.input(
    "events", shape=("time", "batch", 128), signal_type="spikes", unit="spike"
)
cell = snn.components.ping(net, name="cell", n_e=256, n_i=64, source=events)
scores = snn.readouts.MeanVoltage(
    source=cell.E.spikes, classes=10, name="classifier"
)
net.output("scores", scores)

bundle = snn.compile(net)
bundle.write("small_ping.bundle", visualise=True)
```

The bundle contains canonical `graph.json`, optional `training.json`, a
digest-bearing `manifest.json`, copied logical assets, a text summary, and
optional circuit/training/expanded SVG and PNG reports. Physical dataset and
checkpoint paths deliberately do not belong in the graph.

Inputs are graph contracts, not stimulus recipes. A time-varying spike input
uses the canonical `(time, batch, channels)` axis order and declares its signal
type and unit. Dataset selection, Poisson rates, encoders, seeds, durations, and
realised spike tensors belong to the experiment protocol. `tools/snn` may
generate a standard stimulus from CLI parameters or consume an exact replay
with `--input-file`; the latter is optional evidence, not part of the graph.

Run all examples:

```sh
uv run python -m tools.snnlang.examples.build_examples
```

Graph validity is checked independently of a simulator backend. Passing
`target="tools/snn"` adds capability diagnostics but never changes the graph.
See `writings/ar063.typ` for the architectural rationale and staged backend
integration plan.

The first additive `tools/snn` backend route accepts the deliberately narrow
single-layer MNIST PING subset:

```sh
uv run python tools/snn/tool.py sim \
  --bundle small_ping.bundle \
  --t-ms 200 \
  --out-dir run/

uv run python tools/snn/tool.py train \
  --bundle tools/snnlang/examples/generated/ping_classifier.bundle \
  --max-samples 1000 \
  --batch-size 64 \
  --out-dir train-run/
```

Execution choices such as duration, seed, input mode, output directory, and
recordings remain CLI concerns. Structural flags cannot override the bundle.
For the first training subset, `training.json` owns cross-entropy, AdamW,
epochs, learning rate, and a trainable input/readout plus frozen recurrent
scope; dataset cap and batch size remain execution choices.
Unsupported graph structures fail with an element-level capability error;
legacy commands that omit `--bundle` retain their existing defaults and
behaviour.

## Graph-native forward execution

The opt-in graph backend is exposed through typed requests. Bundle loading is
still data-only and does not import this authoring package.

```python
import torch
from tools.snn.execution import ExecutionSpec, simulate

result = simulate(ExecutionSpec(
    kind="simulate",
    executor="graph",
    bundle="small_ping.bundle",
    inputs={"events": torch.zeros(100, 1, 128)},
))
```

The planner lowers the complete dense topology before stepping. It supports
arbitrarily named COBA-LIF and leaky-integrator populations, independent spike
inputs, AMPA and GABA projections, feedforward/recurrent/feedback paths,
integral delay buffers, standard readout operations, and recordings from every
named population. Mean voltage, final voltage, spike count, spike rate, and
cumulative-potential readouts execute through the graph operation vocabulary.
Spike-rate readouts report spikes/s from either an explicit duration in seconds
or a `(time, batch)` valid-time mask whose duration is inferred from graph
`dt`. Zero-delay feedforward edges follow a deterministic topological order.
Recurrent and feedback spikes are causal, so zero additional delay means one
simulation step. Positive delays must be exact integer multiples of the declared
timestep. Zero-delay cycles, dimension errors, malformed masks, ambiguous
durations, polarity mismatches, and missing backend capabilities fail before
simulation.

Graph-native training is deliberately not enabled at this milestone. A graph
training request fails with the explicit future `training:v1` capability rather
than silently routing through the legacy trainer. The legacy CLI and bundle
adapter remain the default and retain their historical numerical contract.
