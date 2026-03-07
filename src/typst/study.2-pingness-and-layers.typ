#set document(
  title: "study.2-pingness-and-layers",
  date: datetime(year: 2026, month: 2, day: 18),
)
#metadata((
  title: "study.2-pingness-and-layers",
  date: "2026-02-18",
  description: "How PING rhythm tightness propagates through a two-layer E chain.",
)) <meta>

#let config = json("_artifacts/study.2-pingness-and-layers/config.json")





*Readers note:* click images to make them larger, you can then use arrow keys
to scroll through them.


= TODO

- [ ] plot population rates for each layer.
- [ ] calculate magnitude and tightness of PING bands for each layer and compare.


= Introduction


We run three ablation configs in a two-layer E network to test why PING
tightening appears in E2.


== Network architecture


The network is setup as follows:

1. 400 E neurons in layer E1.
1. 400 E neurons in layer E2.
2. 100 I neurons.
3. Noisy tonic input to E1 population only.
4. Feedforward E1 to E2 coupling.
5. E1 and I coupling.
6. With and without E2 and I coupling.

We compare:

1. `partially_connected`: E2-I removed.
2. `fully_connected`: E1-E2 present and E2-I present.
3. `independently_connected`: E1-I1 and E2-I2, with E1->E2 feedforward.


= Results



== Partially connected


In this one there is no E2-I coupling, so E2 is not oscillating on its own.
The PING bands are broad in layer E2.

#figure(
  image("_artifacts/study.2-pingness-and-layers/raster_row-1_raster_partially_connected_light.png", width: 60%),
  caption: [Layer-delineated rasters for partially connected variant.],
)




== Fully connected





#figure(
  image("_artifacts/study.2-pingness-and-layers/raster_row-1_raster_fully_connected_light.png", width: 60%),
  caption: [Layer-delineated rasters for fully connected variant.],
)



We see that the PING bands are much stronger and tighter in layer E2 than in
layer E1.

This is likely due to two reasons:

1. E2 is connected to the I population and thus has its own oscillating process.
2. E2 is receiving feedforward input from E1, which is already oscillating, thus
   reinforcing the oscillations in E2.

From this we deduce that the presence of E2-I coupling is crucial for the
emergence of strong and tight PING oscillations in layer E2, and that
feedforward input from E1 can further enhance these oscillations.


== Independently Connected





#figure(
  image("_artifacts/study.2-pingness-and-layers/raster_row-1_raster_independently_connected_light.png", width: 60%),
  caption: [Layer-delineated rasters for independently connected variant.],
)


