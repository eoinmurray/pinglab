#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp074")

#let meta = (
  status: "Implemented",
  title: "From Python graph to spikes",
  date: "2026-07-31",
  description: "The first snnlang vertical slice: author a PING network in Python, compile a portable bundle, execute it through tools/snnsim, and retain both the graph and its spike rasters.",
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

  This is the smallest useful end-to-end demonstration of `snnlang`. The
  experiment runner defines a PING circuit using the Python authoring API,
  compiles it into a deterministic data-only bundle, and invokes `tools/snnsim`
  through its command-line interface. The simulator receives an explicit,
  saved Poisson spike tensor rather than an implicit constant drive. The
  published record retains the graph, compiler reports, exact input, simulator
  rasters, and summary numbers. This entry tests plumbing, not a neuroscientific
  hypothesis.

  == The compiled network

  #data-image(data-file("exp074/network.svg"), width: 100%)

  The graph contains #r.graph.populations populations,
  #r.graph.projections projections, #r.graph.operations graph operations, and
  #r.graph.parameter_tensors parameter tensors. Its canonical graph digest is
  #(r.graph.digest_short + "..."). The simulator consumes this compiled description; the
  Python objects used to author it do not cross the process boundary.

  == Exact spiking input and response

  The saved input tensor has shape #r.input.shape_text and contains
  #r.input.total_spikes spikes. Its requested uniform Poisson rate was
  #r.config.input_rate_hz Hz and its realised rate was
  #calc.round(r.input.realised_rate_hz, digits: 2) Hz. The aligned raster below
  shows trial #r.output.display_trial: the exact input events, then the
  excitatory and inhibitory events produced by the compiled network.

  #data-image(data-file("exp074/rasters.png"), width: 100%)

  Across all #r.config.n_batch trials and #r.config.t_ms ms, the simulator
  measured mean E and I rates of
  #calc.round(r.output.rate_e_hz, digits: 2) Hz and
  #calc.round(r.output.rate_i_hz, digits: 2) Hz respectively. The displayed
  trial contains #r.output.display_trial_spikes.input input,
  #r.output.display_trial_spikes.e E, and
  #r.output.display_trial_spikes.i I spikes.

  == What this establishes

  The useful result is architectural: one short Python definition can be
  statically checked, visualised, serialised, handed to the existing optimised
  PyTorch simulator, and inspected as ordinary Demolab evidence. Execution
  settings remain with the experiment runner, while graph structure remains in
  the bundle. The next experiments can change the authored network without
  growing another pile of bespoke simulator flags—a small victory over
  configuration archaeology.
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can a Python-authored network graph reproduce the intended spiking computation? Inspect the compiled topology and its response to a controlled input.],
    preview-figures, json-inputs: ("exp074",),
  )
}
