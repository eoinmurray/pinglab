#let meta = (
  title: "Networks, signals, and parameters",
  date: "2026-08-14",
  description: "The core snnlang authoring objects, their names, shapes, units, neuron populations, and parameter declarations.",
  collection: "snnlang",
  status: "draft",
  order: 2,
)

#let body = [
  == Networks

  `Network` is the mutable authoring object. Compilation turns it into immutable graph data. Every graph has a name and timestep.

  ```python
  net = snn.Network("example", dt=0.1 * snn.ms)
  ```

  Names must be unique and contain no whitespace. They become stable identifiers in graphs, recordings, and checkpoints.

  == Signals and inputs

  Signals carry an identifier, shape, physical unit, and signal type. Time-varying inputs use the canonical `(time, batch, channels)` axis order.

  ```python
  image = net.input(
      "image",
      shape=("time", "batch", 784),
      signal_type="spikes",
      unit="spike",
  )
  ```

  This declares what the graph accepts. It does not decide whether the spikes came from MNIST, SHD, a Poisson encoder, or a recorded event stream.

  == Populations

  A population contains equally configured neurons and exposes named ports such as `spikes`, `voltage`, `excitatory`, and `inhibitory`.

  ```python
  excitatory = net.population(
      "E",
      size=80,
      neuron=snn.COBA_LIF(
          capacitance_nf=1.0,
          leak_us=0.05,
          resting_mv=-65.0,
          threshold_mv=-50.0,
          reset_mv=-65.0,
          refractory_steps=12,
      ),
  )
  ```

  The authoring API describes COBA-LIF, LIF, and non-spiking leaky-integrator populations. The graph executor currently supports COBA-LIF and leaky integrators. General current-based LIF execution is not implemented.

  == Parameters

  Parameters have stable names, shapes, units, initializers, and optional constraints. Projections normally create their own dense weight parameter, but a named `ParameterRef` may be supplied explicitly.

  `Normal`, `Constant`, and `NonNegative` are currently available. More elaborate initialization and structural sparsity need explicit reusable representations rather than hidden tensor manipulation.

  #link("/ar085/")[Next: Components, projections, and delays]
]
