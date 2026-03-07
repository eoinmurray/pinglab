// title: study.10-ping-snn-ff-inference
// date: 2026-03-01
// description: Adding a global I population to the trained SNN for PING rhythms during inference.

#let config = json("_artifacts/study.10-ping-snn-ff-inference/config.json")
#let results = json("_artifacts/study.10-ping-snn-ff-inference/results.json")






= Summary


This study extends study.9's inference-only SNN classifier with a global
inhibitory population (I\_global, #config.nodes.at(4).size neurons) to observe PING
(Pyramidal-Interneuron Network Gamma) rhythms. The trained E→E weights from
study.8 are loaded unchanged — the only additions are the I population and
reciprocal E\_hid↔I\_global connections.


= Architecture


// Gallery: graph_*.png
// #figure(image("_artifacts/study.10-ping-snn-ff-inference/graph_dark.png"), caption: [Network topology with initialization parameters.])



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


The feedforward E pathway is identical to study.9: three excitatory
populations (784→64→10) with conductance-based LIF neurons, 1 ms timestep,
100 ms simulation window, and W\_ee loaded from study.8.

The new addition is a PING loop:
- *I\_global* (#config.nodes.at(4).size I-type neurons) receives excitation from E\_hid (EI edge)
- *I\_global→E\_hid* (IE edge) provides inhibitory feedback

This creates the classic PING circuit: E drives I, I inhibits E, E recovers
and fires again — generating gamma-band oscillations when the network is
sufficiently driven.

The E→I weights are randomly initialized (mean=0.3, std=0.1) and the
I→E weights (mean=0.15, std=0.05), both with 1 ms delays. The trained
W\_ee block is [N\_E, N\_E] and loads unchanged regardless of N\_I.


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


// Gallery: accuracy_per_class*.png
// #figure(image("_artifacts/study.10-ping-snn-ff-inference/accuracy_per_classdark.png"), caption: [Per-class test accuracy with the I_global PING loop active.])



== Confusion matrix


// Gallery: confusion*.png
// #figure(image("_artifacts/study.10-ping-snn-ff-inference/confusiondark.png"), caption: [Confusion matrix with PING inhibition.])



== Output layer rasters


// Gallery: raster_output_all_all*.png
// #figure(image("_artifacts/study.10-ping-snn-ff-inference/raster_output_all_alldark.png"), caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class.])



== Hidden, inhibitory, and output layer rasters


// Gallery: raster_layers_digit_*.png
// #figure(image("_artifacts/study.10-ping-snn-ff-inference/raster_layers_digit_dark.png"), caption: [Full signal path rasters for each digit (0–9). Top: E_hid. Middle: I_global. Bottom: E_out. Dashed lines separate layers.])



= Discussion


Adding a global inhibitory population creates the PING feedback loop that is
a hallmark of cortical gamma rhythms. The raster plots should show rhythmic
alternation between E\_hid and I\_global bursts, and the power spectrum
should reveal whether these oscillations fall in the gamma band.

The I→E inhibition suppresses some E\_hid activity, reducing the total spike
counts available for readout. Any accuracy loss relative to study.9 is expected,
and a modest drop suggests the PING loop doesn't catastrophically disrupt the
learned representations.

Digits with stronger E\_hid drive (e.g. digit 6, digit 4) show clear PING
bursting: E fires, I follows ~1–2 ms later with a population-wide volley,
then both are silent for ~30–40 ms until E recovers. Digits with sparser
E\_hid activity (e.g. digit 1) barely activate I at all.
