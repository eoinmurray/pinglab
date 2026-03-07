// title: study.6-poisson-encoding
// date: 2026-02-25
// description: Poisson-coded graded input testing spike-count preservation in an E-only network.

#let config = json("_artifacts/study.6-poisson-encoding/config.json")





#figure(
  image("_artifacts/study.6-poisson-encoding/raster_row-1_pingloop_input-poisson_dark.png"),
  caption: [Input Poisson raster with linearly increasing channel rates.],
)



#figure(
  image("_artifacts/study.6-poisson-encoding/raster_row-2_pingloop_network-poisson_dark.png"),
  caption: [E-only network output raster after injecting the Poisson input.],
)



#figure(
  image("_artifacts/study.6-poisson-encoding/spikes_row-3_pingloop_total-spikes-vs-neuron-id_dark.png"),
  caption: [Total output spikes for each E neuron.],
)


