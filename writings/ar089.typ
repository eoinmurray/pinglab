#let meta = (
  title: "Runtime state, checkpoints, and provenance",
  date: "2026-08-14",
  description: "Continue simulations safely, distinguish dynamic state from learned parameters, and record enough identity to replay a run.",
  collection: "snnlang",
  status: "draft",
  order: 7,
)

#let body = [
  == Runtime state

  Runtime state contains the dynamic values required to continue one trajectory: voltages, refractory counters, conductances, projection and input histories, and completed step count. It excludes static parameter values.

  A runtime-state signature covers timestep, populations, input shapes, projections, delays, synapses, and state layout. Incompatible state is rejected rather than partially loaded. This supports burn-in followed by a compatible intervention without restarting the circuit.

  Runtime-state save, load, validation, and continuation are implemented.

  == Checkpoints

  Parameter checkpoints and runtime state serve different purposes. A parameter checkpoint describes learned or initialized weights. Runtime state describes where one simulation trajectory currently is. A complete training checkpoint additionally requires optimizer state, data-order state, and stochastic-stream state.

  Parameter loading and graph runtime state exist today. A portable graph-training checkpoint and exact optimizer resume contract are not implemented.

  == Provenance

  A run should retain the bundle digest, resolved execution protocol, seeds, backend and version, realized initializer statistics, checkpoint identity, and produced artifacts. A resumed run must preserve optimizer state, data order, and stochastic streams when exact continuation is claimed.

  The bundle is network provenance, not the complete experiment record. Experiment runners still own conditions, orchestration, analysis, figures, and publication.

  #link("/ar090/")[Next: Compatibility, status, and extension]
]
