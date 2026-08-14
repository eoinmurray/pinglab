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
with `--input-file` or `--event-file`; the replay is optional evidence, not part
of the graph.

Dense replay is resolved before graph execution. NPY files bind to the sole
graph input; NPZ arrays bind by input id. The resolver requires exact input
coverage, matching time and batch axes, declared feature shapes, finite values,
binary spikes, and boolean or zero/one masks. It records a versioned execution
protocol containing source-file digests, resolved shapes and dtypes, dataset
identity and split when supplied, sample cap, batch size, shuffle behavior,
duration, masks, and the execution seed.

Sparse event replay uses an NPZ coordinate contract. A single-input file stores
`steps`, `batches`, `channels`, `steps_count`, and `batch_size`; multi-input
files prefix each field with the graph input id. Coordinates are zero-based
integer simulation steps ordered by step, batch, and channel. Resolution rejects
duplicates and out-of-bounds coordinates, then materializes binary spikes for
the graph executor while retaining event counts and source identity in the
protocol. Typed requests may combine event-stream spike inputs with dense mask
or continuous inputs when every binding resolves to the same time and batch
axes.

```sh
uv run python tools/snn/tool.py sim \
  --executor graph \
  --bundle small_ping.bundle \
  --input-file replay.npz \
  --input-dataset-id mnist-test-sha256-... \
  --input-split test \
  --no-input-shuffle \
  --seed 17 \
  --out-dir run/
```

The resolved contract is written under `execution_protocol` in `metrics.json`.
Generated Poisson inputs are execution protocols rather than graph structure.
`PoissonInputBinding` supports a fixed rate or a rate sampled uniformly and
independently per presentation from a categorical set. The resolver owns an
explicit seed, records both configured and realized rates, and uses the graph
timestep to materialize Bernoulli-discretized homogeneous Poisson spikes.

```sh
uv run python tools/snn/tool.py sim \
  --executor graph \
  --bundle small_ping.bundle \
  --poisson-protocol categorical-rate \
  --input-rates 0.5 1 5 10 25 \
  --n-batch 64 \
  --t-ms 200 \
  --seed 17 \
  --out-dir run/
```

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
Parameter groups are exhaustive and non-overlapping. The compiled recipe also
contains resolved trainable/frozen parameter lists and a stable per-parameter
learning-rate map; frozen groups use zero and trainable groups require a
positive finite rate.
The standard backward contract uses an explicit fast-sigmoid surrogate and
positive per-population voltage-gradient dampening factors. Compilation records
both in `resolved_gradients`; the legacy adapter maps the supported shared
dampening case back to its established CLI settings.
Training recipes may declare a physical presentation duration independently of
graph `dt`, plus the collection's exact multi-layer spike-budget penalty. Its
stored aggregation contract is the mean over presentations and layers of each
population's mean-rate squared overshoot above a ceiling in Hz.
Before serialization, reverse reachability proves that every objective and
regularizer reaches at least one trainable parameter through enabled graph
elements. The check respects frozen groups and stop-gradient boundaries and
reports the exact reachable and trainable sets when a route is absent.
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

For explicit provenance, pass `DenseArrayBinding` objects through
`ExecutionSpec.input_bindings`; the original `inputs={...}` tensor mapping is a
compatible in-memory shorthand and is resolved through the same validator.

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

Projections authored with `enabled=False` remain in the graph with the same
parameter shape, initializer, name, and construction position, but contribute
zero conductance at runtime. Controlled cells can therefore disable a recurrent
loop without shifting later tensor identities or random initialization draws.

Initializer specifications distinguish lower-clamped and signed normals,
uniform, constant, and zero distributions. Lower-clamped normals can apply
seeded Bernoulli or exact-fan-in initial zeroing. Build metrics expose stable
per-parameter realized statistics together with the constraint, unit, runtime
shape, and fan-in scaling convention.

The typed graph API supports deterministic single-batch AdamW updates for the
validated cross-entropy and spike-budget vocabulary. `ExecutionSpec` supplies
resolved inputs and external targets; a bundle may authenticate and supply its
training recipe. Results expose per-update loss components, named gradients,
parameters, and optimizer state.

Graph training checkpoints use a versioned manifest plus a digest-verified
tensor payload. They key parameters and AdamW state by stable graph id and
record graph/training digests, completed updates, execution protocol,
initializer metadata, and CPU random state. The trainer can save final and
invocation-selected checkpoints and resume exactly after rejecting recipe,
protocol, initializer, shape, dtype, or parameter-set mismatches. An explicit
one-layer legacy parameter map fails closed when any graph parameter is
unrepresentable. Dataset iteration, CLI target loading, accelerator stochastic
state, and production trajectory parity remain separate gates. The legacy CLI
and bundle adapter remain the default and retain their historical numerical
contract.
