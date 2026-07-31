# snnlang

`snnlang` is a standalone Python authoring library for graph-shaped spiking
circuits. It validates a network and optional standard training recipe, then
writes a deterministic, data-only bundle. It does not simulate or train.

```python
from tools import snnlang as snn

net = snn.Network("small_ping")
events = net.input(
    "events", shape=("batch", "time", 128), signal_type="spikes", unit="spike"
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
```

Execution choices such as duration, seed, input mode, output directory, and
recordings remain CLI concerns. Structural flags cannot override the bundle.
Unsupported graph structures fail with a capability error; legacy commands
that omit `--bundle` retain their existing defaults and behaviour.
