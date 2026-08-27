#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp077")

#let meta = (
  status: "Implemented",
  title: "Arbitrary coupled graphs execute natively",
  date: "2026-08-05",
  description: "A cumulative snnlang validation establishes the typed execution seam, exact single-PING parity, and graph-only construction of two independently driven PING circuits with delayed reciprocal inhibition.",
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

  This validation completes the cumulative architecture gates through exp102
  Milestone 3. The new graph executor matches the active legacy single-PING
  reference exactly: seeded parameters, E and I spikes, mean-voltage output,
  and checkpoint replay all have zero recorded discrepancy. Its matched local
  steady-state runtime is below the legacy runtime, so the declared overhead
  gate passes without an exception. The arbitrary-graph fixture then executes
  two independently driven PING circuits in uncoupled, unidirectional,
  reciprocal, and explicitly delayed reciprocal forms. Every variant changes
  graph data only. This is an execution and causality result, not a scientific
  claim about inhibitory coupling.

  == Registered goal

  #raw(read(data-file("exp077/goal.txt")), block: true, lang: "text")

  == Methods

  + The compatibility adapter converts legacy flags and bundles into typed
    execution requests. Legacy remains the default executor.
  + The graph planner validates capabilities, dimensions, polarity, temporal
    causality, and integral delays before lowering named populations and dense
    projections into a fixed schedule.
  + The single-PING gate uses the same authored bundle, seed, input tensor, and
    initial parameters in legacy and graph-native execution. It compares active
    spikes and outputs, then reloads a graph checkpoint.
  + The coupled fixture authors two different-sized PING circuits with separate
    spike inputs. GABA projections connect each inhibitory population to the
    other circuit's excitatory population as required by each variant.
  + The runner calculates only compact cross-correlation diagnostics to show
    that named recordings are usable. It does not sweep coupling parameters.

  == Exact single-PING compatibility

  #table(
    columns: (1.8fr, 1fr),
    [Gate], [Recorded result],
    [Seeded parameter maximum absolute error], [#r.parity_performance.parameter_max_abs],
    [E/I spike mismatch count], [#r.parity_performance.spike_mismatch_count],
    [Named output maximum absolute error], [#r.parity_performance.output_max_abs],
    [Checkpoint replay maximum absolute error], [#r.parity_performance.checkpoint_replay_max_abs],
    [Legacy median runtime], [#calc.round(r.parity_performance.legacy_median_s, digits: 3) s],
    [Graph median runtime], [#calc.round(r.parity_performance.graph_median_s, digits: 3) s],
    [Graph overhead], [#calc.round(r.parity_performance.graph_overhead_percent, digits: 1)%],
    [Allowed overhead], [#r.parity_performance.performance_gate_percent%],
    [CPU Inductor first invocation (#r.parity_performance.compile_workload_steps steps × #r.parity_performance.compile_workload_batch samples)], [#calc.round(r.parity_performance.compile_first_s, digits: 3) s],
    [CPU Inductor warm median (same bounded shape)], [#calc.round(r.parity_performance.compiled_warm_median_s, digits: 3) s],
    [Compiled replay maximum absolute error], [#r.parity_performance.compiled_replay_max_abs],
    [Legacy peak traced Python memory], [#r.parity_performance.legacy_peak_python_bytes bytes],
    [Graph peak traced Python memory], [#r.parity_performance.graph_peak_python_bytes bytes],
  )

  The active parity result is exact at the recorded precision. The graph path is
  #calc.round(-r.parity_performance.graph_overhead_percent, digits: 1)% faster
  on this matched CPU reference. Matched steady state is measured eager for both
  paths because the established legacy boundary disables `torch.compile` on
  CPU. A separate bounded CPU Inductor probe records compilation and warm
  execution with exact compiled replay. The larger compile attempt was killed
  after five minutes, so accelerator compilation and large-shape CPU compile
  scaling remain limitations. The compilation boundary remains an internal
  backend concern rather than graph data.

  == Coupled acceptance fixture

  #figure(
    data-image(data-file("exp077/reciprocal_delayed.svg"),
      width: 100%,
      alt: "Circuit diagram of two PING components with reciprocal delayed inhibitory projections.",
    ),
    caption: [Reciprocal delayed acceptance graph. Two separately named PING
      components receive independent inputs. Each cross-circuit red projection
      carries spikes from one inhibitory population to the other excitatory
      population. The topology is compiled, planned, and archived as data.],
  )

  #table(
    columns: (1.7fr, 0.8fr, 0.9fr, 1.1fr),
    [Variant], [Cross edges], [Delay steps], [Graph digest],
    [Uncoupled], [#r.variants.uncoupled.coupling_projection_count], [none], [#short-digest(r.variants.uncoupled.graph_digest)],
    [Unidirectional], [#r.variants.unidirectional.coupling_projection_count], [#r.variants.unidirectional.coupling_delay_steps.at(0)], [#short-digest(r.variants.unidirectional.graph_digest)],
    [Reciprocal, zero additional delay], [#r.variants.reciprocal_zero_additional.coupling_projection_count], [#r.variants.reciprocal_zero_additional.coupling_delay_steps.at(0)], [#short-digest(r.variants.reciprocal_zero_additional.graph_digest)],
    [Reciprocal, explicit delay], [#r.variants.reciprocal_delayed.coupling_projection_count], [#r.variants.reciprocal_delayed.coupling_delay_steps.at(0)], [#short-digest(r.variants.reciprocal_delayed.graph_digest)],
  )

  A zero-additional-delay recurrent or feedback edge still receives its source
  spike on the next causal step. The explicitly delayed fixture receives it
  after #r.delay_timing.explicit_delay_steps steps. Non-integral delays and
  zero-delay algebraic cycles fail during planning.

  == Named population recordings

  #figure(
    data-image(data-file("exp077/matched_rasters.png"),
      width: 100%,
      alt: "Excitatory spike rasters for both circuits across four coupling graph variants.",
    ),
    caption: [Matched excitatory recordings from the four graph variants. Time
      in milliseconds is horizontal and cell index is vertical; the left and
      right columns show circuits A and B. Strong inhibition silences circuit B
      in two fixtures, which is a useful execution stress case but not evidence
      for a coupling mechanism. The retained input, E, I, voltage, and
      projection-conductance arrays permit later analysis without simulator
      access.],
  )

  The phase and synchrony calculations are deliberately not interpreted.
  Circuit B has no spikes in the unidirectional and reciprocal
  zero-additional-delay fixtures, so a scientific phase comparison would be
  meaningless. Milestone 4 remains unstarted.

  == Exit decision

  Milestones 1, 2, and 3 pass this local validation. Legacy routing remains the
  default, bundle loading remains independent of the authoring package, and the
  arbitrary coupling variants require no simulator edit. The committed record
  includes authenticated bundles, canonical diagrams, independent inputs,
  named recordings from both E/I circuits, delay evidence, parity evidence,
  performance timings, provenance, and a reproducer. Paid compute cost is
  \$#r.exit.paid_compute_usd.

  == Activity log

  #for event in r.activity [
    *#event.timestamp* \
    #event.event

  ]
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can arbitrary coupled network graphs execute natively? Compare reciprocal, delayed, unidirectional, and uncoupled graph variants under matched inputs.],
    preview-figures, json-inputs: ("exp077",),
  )
}
