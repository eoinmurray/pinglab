// title: study.4-rate-encoding
// date: 2026-02-22
// description: Rate-coded signal injection baseline measuring propagation fidelity across layers.

#let config = json("_artifacts/study.4-rate-encoding/config.json")






= Summary


This study starts with a simple message source only (no carrier). The first
step is to verify and visualize the injected message signal before running
propagation/decoding metrics across layers.


= Config Snapshot


#table(
  columns: 2,
  [Key], [Value],
  [`sim.neuron_model`], [#config.sim.neuron_model],
  [`sim.dt_ms`], [#config.sim.dt_ms],
  [`sim.T_ms`], [#config.sim.T_ms],
  [`meta.message.message_freq_hz`], [#config.meta.message.message_freq_hz],
)



= Results


#figure(
  image("_artifacts/study.4-rate-encoding/input_row-1_ff_sweep_signal-components_dark.png"),
  caption: [Message envelope and resulting injected current (carrier removed).],
)



#figure(
  image("_artifacts/study.4-rate-encoding/raster_row-2_ff_sweep_layers_dark.png"),
  caption: [Layer-delineated raster with message input injected into E1.],
)



#figure(
  image("_artifacts/study.4-rate-encoding/rate_row-3_ff_sweep_layers_dark.png"),
  caption: [Population rates for E1, E2, and E3.],
)



#figure(
  image("_artifacts/study.4-rate-encoding/decode_row-4_ff_sweep_envelopes_dark.png"),
  caption: [Decoded envelopes from E1/E2/E3 rates against message reference.],
)


