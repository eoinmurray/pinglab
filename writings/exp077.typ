#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp077")

#let meta = (
  status: "[▦ DATA]",
  title: "Arbitrary coupled graphs execute natively",
  updated_at: "2026-08-31",
  date: "2026-08-05",
  description: "Four graph-defined coupling variants with separate numerical compatibility and runtime checks.",
  collection: "snnlang-docs",
  order: 4,
)

#let inputs = ("exp077",)
#let preview-figures = (
  (path: "exp077/reciprocal_delayed.svg", label: "reciprocal delayed"),
  (path: "exp077/matched_rasters.png", label: "matched rasters"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp077/numbers.json"))
#let short-digest(x) = x.slice(0, 19) + "..."

#let body = [
  == Abstract


  Asked whether the graph runtime can execute arbitrary coupling between
  independently driven excitatory–inhibitory circuits. Compared uncoupled,
  directional, reciprocal and delayed-reciprocal graphs, alongside a separate
  circuit-parity and runtime check.

  Graph execution matched the explicit circuit and reproduced the expected
  causal effects of coupling direction and delay within the tested workload.
  These are bounded architecture and timing checks, not evidence for a
  biological coupling mechanism or performance on unrelated workloads.

  == Results

  #with-result-sections[

  === Single-circuit compatibility, timing and memory measurements

  #figure(table(columns: (1.8fr, 1fr),
    [Measurement], [Result],
    [Maximum parameter discrepancy], [#r.parity_performance.parameter_max_abs],
    [Spike mismatches], [#r.parity_performance.spike_mismatch_count],
    [Maximum output discrepancy], [#r.parity_performance.output_max_abs],
    [Checkpoint replay discrepancy], [#r.parity_performance.checkpoint_replay_max_abs],
    [Explicit median runtime], [#calc.round(r.parity_performance.legacy_median_s, digits: 4) s],
    [Graph median runtime], [#calc.round(r.parity_performance.graph_median_s, digits: 4) s],
    [Graph overhead], [#calc.round(r.parity_performance.graph_overhead_percent, digits: 1)%],
    [Compiled first invocation], [#calc.round(r.parity_performance.compile_first_s, digits: 3) s],
    [Compiled warm median], [#calc.round(r.parity_performance.compiled_warm_median_s, digits: 4) s],
    [Compiled repeat discrepancy], [#r.parity_performance.compiled_replay_max_abs],
    [Explicit peak traced Python memory], [#r.parity_performance.legacy_peak_python_bytes bytes],
    [Graph peak traced Python memory], [#r.parity_performance.graph_peak_python_bytes bytes],
  ), kind: table,
    caption: [Matched eager execution used 100 timesteps and eight samples,
      with five timed calls after two warmups. CPU Inductor compilation used a
      separate #(r.parity_performance.compile_workload_steps)-step,
      #(r.parity_performance.compile_workload_batch)-sample input and three warm
      calls. Compiled repeat discrepancy compares two compiled calls, not
      compiled versus eager execution. Traced Python memory excludes native
      tensor storage; no uncertainty across independent timing sessions is estimated.])

  === Reciprocal delayed circuit graph

  #figure(data-image(data-file("exp077/reciprocal_delayed.svg"), width: 100%),
    caption: [Reciprocal delayed graph. Circuit A contains 16 excitatory and
      four inhibitory neurons; circuit B contains 12 excitatory and three
      inhibitory neurons. Each cross-circuit inhibitory projection targets the
      other circuit's excitatory population.])

  === Coupling variants and causal delay steps

  Separate pulse and causal-planning tests passed.

  #figure(table(columns: (1.8fr, 1fr, 1fr),
    [Variant], [Cross projections], [Delay steps],
    [Uncoupled], [#r.variants.uncoupled.coupling_projection_count], [none],
    [Unidirectional], [#r.variants.unidirectional.coupling_projection_count], [#r.variants.unidirectional.coupling_delay_steps.at(0)],
    [Reciprocal], [#r.variants.reciprocal_zero_additional.coupling_projection_count], [#r.variants.reciprocal_zero_additional.coupling_delay_steps.at(0)],
    [Delayed reciprocal], [#r.variants.reciprocal_delayed.coupling_projection_count], [#r.variants.reciprocal_delayed.coupling_delay_steps.at(0)],
  ), kind: table,
    caption: [Coupling changed graph data only. A recurrent or feedback
      connection with no additional delay receives spikes on the next causal
      step; the explicit delay was #r.delay_timing.explicit_delay_steps steps.])

  === Matched-input rasters across coupling variants

  #figure(data-image(data-file("exp077/matched_rasters.png"), width: 100%),
    caption: [Excitatory spikes in the first sample of each variant, with
      circuit A on the left and circuit B on the right. The time axis is in
      milliseconds. Identical input tensors were reused across variants;
      panels show the first sample only.])

  Sparse or silent responses in these illustrative samples do not establish a
  phase-coupling mechanism.

  ]

  == Methods

  We separated graph-defined coupling checks from a bounded implementation
  comparison using PyTorch#cite(1).

  + *Define and drive the coupled circuits.* We authored two circuits with
    16/four and 12/three excitatory/inhibitory neurons and eight/six input
    channels respectively. Deterministic input pulses arrived every ten and
    thirteen timesteps, with different offsets for the second sample. All
    variants used seed #r.config.seed, #r.config.steps timesteps,
    #r.config.batch samples and #r.config.dt_ms ms integration steps.
  + *Vary cross-circuit inhibition.* We compared no cross projections, inhibition
    from A to B, reciprocal inhibition with one-step causal delay and
    reciprocal inhibition with five-step delay. Cross-projection weights were
    fixed at three and inhibitory conductance decay at nine milliseconds.
    We recorded all exposed populations and computed zero-lag correlation and
    peak cross-correlation lag; silent traces received the existing zero-valued
    diagnostic convention and were not interpreted as phase estimates.
  + *Compare explicit and graph execution.* We initialised a separate
    256-excitatory/64-inhibitory classifier identically in both implementations,
    using seed 17 and the same 100-step, eight-sample spike tensor. We compared
    parameters, spikes and outputs, and reloaded the graph state into an
    independently initialised model. All comparisons used recorded tensors.
  + *Measure timing and causal boundaries.* We timed five eager calls after
    two warmups and calculated graph overhead relative to the explicit
    implementation's median, using the unchanged ten-percent threshold.
    We measured CPU Inductor's first invocation and three warm calls separately
    on a 20-step, two-sample input. Separate numerical tests checked delayed
    pulse arrival and causal planning; no coupling sweep or accelerator
    performance study was performed.

  #run-view("exp077", inputs)

  #reference-list(((text: [Adam Paszke et al.: _PyTorch: An Imperative Style, High-Performance Deep Learning Library_. NeurIPS, 2019.], doi: "10.48550/arXiv.1912.01703"),))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can arbitrary coupled network graphs execute natively? Compare reciprocal, delayed, unidirectional, and uncoupled graph variants under matched inputs.],
    preview-figures, json-inputs: ("exp077",),
  )
}

#let meta = meta + (assets: input-assets("exp077", inputs))
#let body = with-datasets("exp077", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
