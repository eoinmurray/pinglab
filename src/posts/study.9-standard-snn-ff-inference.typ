// title: study.9-standard-snn-ff-inference
// date: 2026-02-28
// description: Inference-only MNIST evaluation loading study.8 trained SNN weights.

#let config = json("_artifacts/study.9-standard-snn-ff-inference/config.json")
#let results = json("_artifacts/study.9-standard-snn-ff-inference/results.json")






= Summary


This study loads the trained weights from study.8 and runs pure inference on
the MNIST test set. No training, no gradients — just the forward pass of the
conductance-based LIF network with spike-count decoding. The goal is to
confirm that the trained SNN can be deployed as a standalone classifier by
loading a weight checkpoint.


= Architecture


#figure(
  image("_artifacts/study.9-standard-snn-ff-inference/graph_dark.png"),
  caption: [Network topology with initialization parameters.],
)




= Config Snapshot


#table(
  columns: 2,
  [Key], [Value],
  [`sim.neuron_model`], [#config.sim.neuron_model],
  [`sim.dt_ms`], [#config.sim.dt_ms],
  [`sim.T_ms`], [#config.sim.T_ms],
  [`meta.batch_size`], [#config.meta.batch_size],
  [`meta.input_scale`], [#config.meta.input_scale],
  [`meta.test_subset_size`], [#config.meta.test_subset_size],
  [`meta.weights_source`], [#config.meta.weights_source],
)



= Method


The network topology and biophysics are identical to study.8: three
feedforward excitatory populations (784 → 64 → 10) with conductance-based LIF
neurons, 1 ms timestep, 100 ms simulation window.

The only difference: instead of randomly initializing the weight matrix and
training it, we load the `W_ee` tensor saved by study.8 after surrogate-gradient
training. The runtime is compiled from the same graph spec, then `W_ee` is
overwritten with the checkpoint via `copy_()`.

Inference uses the same spike-count decoding: each test image is rate-coded
into the input layer, the network runs for 100 ms, and the output neuron
with the most spikes wins.


= Results



== Results Snapshot


#table(
  columns: 2,
  [Metric], [Value],
  [Test accuracy], [#results.test_accuracy%],
  [Test loss], [#results.test_loss],
  [Test samples], [#results.test_samples],
  [Weights source], [#results.weights_source],
  [Inference time], [#results.elapsed_secondss],
)



== Overall accuracy


The trained SNN achieves *#results.test_accuracy% test accuracy* on #results.test_samples held-out MNIST
samples (test loss: #results.test_loss). This is consistent with study.8's best test
accuracy — the small difference is expected from evaluating a
different random subset.


== Per-class accuracy


#figure(
  image("_artifacts/study.9-standard-snn-ff-inference/accuracy_per_class_dark.png"),
  caption: [Per-class test accuracy across all 10 digit classes.],
)



Figure 1.1 shows wide variation across digit classes. Digits with simple,
distinctive strokes (like 1) tend to be classified most reliably, while
visually similar digits (like 8 vs 5) are frequently confused (see confusion
matrix).


== Confusion matrix


#figure(
  image("_artifacts/study.9-standard-snn-ff-inference/confusion_dark.png"),
  caption: [Confusion matrix.],
)



Figure 1.2 reveals the network's failure modes. Digit 5 is a common false
positive — digits 0, 3, and 8 are frequently misclassified as 5. This makes
sense: in spike-count decoding, the output neuron for class 5 tends to be
highly active across many inputs, creating a bias.


== Output layer rasters


#figure(
  image("_artifacts/study.9-standard-snn-ff-inference/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class. Each panel shows all 10 output neurons over 100 ms.],
)




== Hidden and output layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.9-standard-snn-ff-inference/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit (0–9). Top: E_hid (64 neurons). Bottom: E_out (10 neurons). Dashed line separates layers.],
)




= Discussion


This study confirms that trained SNN weights can be checkpointed and reloaded
for inference without any training infrastructure. The weight matrix is the
only learned artifact — all biophysics, topology, and simulation parameters
come from the graph spec.

The accuracy matches study.8's training results, validating that the
save/load pipeline preserves the learned representations faithfully. The
per-class analysis reveals that the hidden layer struggles most with
visually similar digits, suggesting that scaling the hidden layer would
help most.
