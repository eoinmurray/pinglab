#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "LIF voltage traces and a recurrent network raster",
  date: "2026-05-13",
  description: "A single leaky integrate-and-fire neuron under tonic input, then 200 of them wired together.",
  collection: "neuron-models",
  status: "final",
)

#let run = json("/artifacts/data/exp000/numbers.json")

#let body = [
  A single leaky integrate-and-fire (LIF) neuron driven by a steady supra-threshold
  current should fire periodically; wire 200 of them together with sparse excitation and
  noisy per-neuron bias and the population should sustain ongoing, irregular firing rather
  than fall silent or lock into synchrony. We ran both (one neuron, then the network) and
  report the voltage trace, the raster, and the measured firing rates.

  == Methods

  The LIF membrane voltage decays toward a resting potential and is pushed around by
  injected current; a threshold crossing emits a spike and resets the voltage.

  - *Subthreshold dynamics.* Integrate the leaky-membrane ODE
    $ tau_m (dif V) / (dif t) = -(V - V_"rest") + R_m I, $
    with $V$ the membrane potential, $tau_m$ the membrane time constant, $V_"rest"$ the
    resting potential, $R_m$ the input resistance, and $I$ the injected current.
  - *Spike-and-reset.* When $V(t) >= V_"th"$, record a spike at $t$ and set
    $V <- V_"reset"$, with $V_"th"$ the threshold and $V_"reset"$ the reset potential.
  - *Integration.* Forward Euler at timestep $Delta t$,
    $ V <- V + (Delta t) / tau_m [-(V - V_"rest") + R_m I]. $
  - *Single-neuron run.* Drive one neuron with a constant supra-threshold tonic current.
  - *Network run.* Wire 200 neurons with sparse, all-excitatory recurrence. Each neuron
    $i$ takes total current $I_i (t) = I_i^"bias" + s_i (t)$, with noisy bias
    $I_i^"bias" tilde.op cal(N)(mu, sigma^2)$ and synaptic current $s_i$ that decays
    between spikes and is kicked by incoming ones,
    $ s_i <- s_i e^(-Delta t \/ tau_"syn") + sum_(j in "spiked") W_(i j), $
    where $tau_"syn"$ is the synaptic time constant and $W_(i j) = w C_(i j)$ with
    connectivity $C_(i j) tilde.op "Bernoulli"(p)$, weight $w$, and no self-connections.
    Each neuron is held at $V_"reset"$ for a refractory period after spiking.

  Parameter values are tabulated below.

  == Results

  With a steady tonic input above threshold, the single neuron fires periodically at
  #calc.round(run.lif.firing_rate_hz, digits: 0) Hz. Each upward sweep is the membrane
  charging through its RC time constant; each vertical line is a reset after a spike.

  #figure(
    image("/artifacts/data/exp000/lif.svg", width: 100%),
    caption: [Single LIF neuron under tonic input: membrane potential (mV) against time (ms).
      The membrane charges toward its steady state and resets at each threshold crossing,
      giving periodic firing at #calc.round(run.lif.firing_rate_hz, digits: 0) Hz.],
  )

  The network sustains ongoing irregular firing: a population mean of
  #calc.round(run.net.mean_firing_rate_hz, digits: 0) Hz, spanning
  #calc.round(run.net.min_firing_rate_hz, digits: 0)–#calc.round(run.net.max_firing_rate_hz, digits: 0) Hz
  across neurons, no silence, no runaway synchrony. The raster shows every spike from every
  neuron over half a second; the panel beneath tracks the network's mean total input current.

  #figure(
    image("/artifacts/data/exp000/net.png", width: 100%),
    caption: [Network raster (top: spike time vs neuron index) and mean input current (bottom,
      nA vs ms) across #run.net.config.n recurrently coupled LIF neurons over
      #calc.round(run.net.config.duration, digits: 0) ms. Mean firing rate
      #calc.round(run.net.mean_firing_rate_hz, digits: 0) Hz; the shaded band is ±1 s.d. of the
      per-neuron total current across the population.],
  )

  #numbers-table(run.lif, title: "LIF run parameters")
  #numbers-table(run.net, title: "Network run parameters")

  #provenance-footer(run.lif.config)
]
