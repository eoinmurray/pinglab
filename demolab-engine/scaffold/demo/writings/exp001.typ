#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "EIF voltage traces and a recurrent network raster",
  date: "2026-05-13",
  description: "The same setup as exp000, but using exponential integrate-and-fire neurons in place of LIF.",
  collection: "neuron-models",
  status: "final",
)

#let run = json("/artifacts/data/exp001/numbers.json")
#let lif-rate = json("/artifacts/data/exp000/numbers.json").lif.firing_rate_hz

#let body = [
  The exponential integrate-and-fire (EIF) neuron replaces LIF's hard threshold with an
  explicit exponential term that models the upswing of a spike. Under the same tonic drive
  as #link("exp000.html")[exp000] it should fire at a different rate than LIF (the
  exponential accelerates spike initiation, but the upswing itself takes time) while a
  network of them should behave much like the LIF network with a softened spike onset. We
  ran the single neuron and the network and compare against exp000.

  == Methods

  The EIF neuron keeps LIF's one-variable spirit but grows an exponential term, so the
  threshold becomes a soft inflection point rather than a discontinuity.

  - *Subthreshold dynamics.* Integrate
    $ tau_m (dif V) / (dif t) = -(V - V_"rest") + Delta_T exp((V - V_T) / Delta_T) + R_m I, $
    with $V$ the membrane potential, $tau_m$ the membrane time constant, $V_"rest"$ the
    resting potential, $R_m$ the input resistance, $I$ the injected current, $V_T$ the soft
    threshold, and $Delta_T$ the slope factor.
  - *Peak-and-reset.* When $V(t) >= V_"peak"$, record a spike at $t$ and set
    $V <- V_"reset"$, with $V_"peak"$ the peak cutoff and $V_"reset"$ the reset potential.
  - *Threshold behaviour.* $V_T$ is no longer a discontinuity: it is the point at which the
    exponential term takes over and the voltage blows up on its own. Small $Delta_T$
    approximates LIF; larger $Delta_T$ smooths spike initiation.
  - *Integration.* Forward Euler at timestep $Delta t$,
    $ V <- V + (Delta t) / tau_m [-(V - V_"rest") + Delta_T exp((V - V_T) / Delta_T) + R_m I]. $
  - *Single-neuron run.* Drive one neuron with a constant tonic current.
  - *Network run.* Structurally identical to #link("exp000.html")[exp000]: 200
    all-excitatory neurons, sparse $p$ random connectivity, weight $w$,
    exponentially-decaying synaptic input with time constant $tau_"syn"$, noisy per-neuron
    bias $I_i^"bias" tilde.op cal(N)(mu, sigma^2)$, and a refractory period, with every
    neuron now obeying the EIF dynamics above.

  Parameter values are tabulated below.

  == Results

  Under the same tonic input, the EIF neuron fires at
  #calc.round(run.eif.firing_rate_hz, digits: 0) Hz, against
  #calc.round(lif-rate, digits: 0) Hz for the LIF neuron in exp000: the exponential term
  accelerates spike initiation, but the upswing takes time, so the inter-spike period
  shifts.

  #figure(
    image("/artifacts/data/exp001/eif.svg", width: 100%),
    caption: [Single EIF neuron under tonic input: membrane potential (mV) against time (ms).
      The soft exponential threshold produces a curved spike upswing rather than LIF's hard
      reset, and periodic firing at #calc.round(run.eif.firing_rate_hz, digits: 0) Hz.],
  )

  The EIF network sustains irregular firing at a population mean of
  #calc.round(run.enet.mean_firing_rate_hz, digits: 0) Hz, spanning
  #calc.round(run.enet.min_firing_rate_hz, digits: 0)–#calc.round(run.enet.max_firing_rate_hz, digits: 0) Hz
  across neurons, the same asynchronous-irregular regime as the LIF network, shifted in rate.

  #figure(
    image("/artifacts/data/exp001/enet.png", width: 100%),
    caption: [EIF network raster (top: spike time vs neuron index) and mean input current
      (bottom, nA vs ms) across #run.enet.config.n neurons over
      #calc.round(run.enet.config.duration, digits: 0) ms. Mean firing rate
      #calc.round(run.enet.mean_firing_rate_hz, digits: 0) Hz; the shaded band is ±1 s.d. of the
      per-neuron total current.],
  )

  #numbers-table(run.eif, title: "EIF run parameters")
  #numbers-table(run.enet, title: "EIF network run parameters")

  #provenance-footer(run.eif.config)
]
