#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp074")

#let meta = (
  status: "[▦ DATA | v28.0.0]",
  title: "From Python graph to spikes",
  updated_at: "2026-08-31",
  date: "2026-07-31",
  description: "A compiled excitatory–inhibitory network responds to a controlled spike input.",
  collection: "snnlang-docs",
  order: 1,
)

#let inputs = ("exp074",)
#let preview-figures = (
  (path: "exp074/network.svg", label: "network"),
  (path: "exp074/rasters.png", label: "rasters"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp074/numbers.json"))

#let body = [
  == Abstract

  Asked whether a Python-authored graph can be compiled into an executable
  excitatory–inhibitory spiking network. Supplied explicit input spikes to the
  compiled graph and retained aligned topology, input and population-activity
  evidence.

  The compiled description produced the expected excitatory and inhibitory
  simulation outputs from the supplied input. Demonstrates the
  graph-to-simulation integration path on a bounded example, not a
  neuroscientific mechanism.

  == Results

  #with-result-sections[

  === Compiled PING classifier topology

  #figure(data-image(data-file("exp074/network.svg"), width: 100%),
    caption: [Compiled topology with #r.config.n_e excitatory neurons,
      #r.config.n_i inhibitory neurons and a ten-class mean-voltage readout.
      The complete graph contains #r.graph.populations populations and
      #r.graph.projections projections.])

  === Aligned input, excitatory and inhibitory spike rasters

  #figure(data-image(data-file("exp074/rasters.png"), width: 100%),
    caption: [Input, excitatory and inhibitory spikes in trial
      #r.output.display_trial, with zero-based trial numbering. The displayed
      trial contained #r.output.display_trial_spikes.input input,
      #r.output.display_trial_spikes.e excitatory and
      #r.output.display_trial_spikes.i inhibitory events. Rates in the abstract
      aggregate all #r.config.n_batch trials, not only this illustrative raster.])

  ]

  == Methods

  We tested whether a compiled network description reproduced the requested
  spiking computation in a PyTorch-based simulator#cite(1).

  + *Define the network.* We authored a pyramidal–interneuron gamma circuit with
    #r.config.n_e excitatory and #r.config.n_i inhibitory neurons, driven by
    #r.config.n_input input channels. A ten-class mean-voltage readout received
    excitatory spikes; no weights were trained in this experiment.
  + *Generate and simulate input.* We generated independent Bernoulli spike
    events with probability equal to input rate times timestep in seconds,
    approximating a #r.config.input_rate_hz Hz Poisson drive. We used seed
    #r.config.seed, #r.config.n_batch trials, a #r.config.dt_ms ms integration
    step and a #r.config.t_ms ms duration, and supplied the exact generated
    tensor to the simulator.
  + *Measure the response.* We counted input events and divided by channel
    count, trial count and duration to obtain the realised input rate.
    Excitatory and inhibitory rates used their respective population sizes;
    we recorded aligned event times and cell indices for the specified
    illustrative trial. These measurements establish execution and activity,
    not oscillatory synchronisation or classification performance.

  #run-view("exp074", inputs)

  #reference-list(((text: [Adam Paszke et al.: _PyTorch: An Imperative Style, High-Performance Deep Learning Library_. NeurIPS, 2019.], doi: "10.48550/arXiv.1912.01703"),))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can a Python-authored network graph reproduce the intended spiking computation? Inspect the compiled topology and its response to a controlled input.],
    preview-figures, json-inputs: ("exp074",),
  )
}

#let meta = meta + (assets: input-assets("exp074", inputs))
#let body = with-datasets("exp074", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
