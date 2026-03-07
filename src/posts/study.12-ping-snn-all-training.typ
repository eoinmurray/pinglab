// title: study.12-ping-snn-all-training
// date: 2026-03-03
// description: Training a PING-SNN on MNIST with all weights (W_ee, W_ei, W_ie) trained jointly via surrogate gradient BPTT.

#let config = json("_artifacts/study.12-ping-snn-all-training/config.json")
#let results = json("_artifacts/study.12-ping-snn-all-training/results.json")






= Summary


This study extends study.11 by training all synaptic weights — W\_ee, W\_ei,
and W\_ie — jointly via surrogate gradient BPTT. In study.11, only the E→E
weights were trained while the E↔I connections remained at their random
initial values. Here, the optimizer can also learn the inhibitory dynamics:
how strongly E\_hid drives I\_global, and how strongly I\_global suppresses
E\_hid.

The question: does training the E↔I weights improve classification, or does
the additional flexibility lead to instability?


= Architecture


// Gallery: graph_*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/graph_dark.png"), caption: [Network topology with initialization parameters.])


The architecture is identical to study.11:

#table(
  columns: 4,
  [Layer], [Population], [Size], [Role],
  [Input], [`E_in`], [784], [One neuron per pixel, driven by rate-coded image],
  [Hidden], [`E_hid`], [#config.nodes.at(2).size], [Learned feature extraction],
  [Output], [`E_out`], [10], [One neuron per digit class],
  [Inhibitory], [`I_global`], [#config.nodes.at(4).size], [PING feedback inhibition on E\_hid],
)



= Config Snapshot


#table(
  columns: 2,
  [Key], [Value],
  [`sim.neuron_model`], [#config.sim.neuron_model],
  [`sim.dt_ms`], [#config.sim.dt_ms],
  [`sim.T_ms`], [#config.sim.T_ms],
  [`meta.epochs`], [#config.meta.epochs],
  [`meta.batch_size`], [#config.meta.batch_size],
  [`meta.lr`], [#config.meta.lr],
  [`meta.input_scale`], [#config.meta.input_scale],
  [`meta.subset_size`], [#config.meta.subset_size],
  [`constraints.nonnegative_weights`], [#config.constraints.nonnegative_weights],
)



= Method



== What is trained


Three weight matrices receive gradient updates:

#table(
  columns: 3,
  [Matrix], [Params], [Connection],
  [`W_ee`], [#results.trainable_params_breakdown.W_ee], [E\_in→E\_hid and E\_hid→E\_out (structural mask)],
  [`W_ei`], [#results.trainable_params_breakdown.W_ei], [E\_hid→I\_global],
  [`W_ie`], [#results.trainable_params_breakdown.W_ie], [I\_global→E\_hid],
  [*Total*], [*#results.trainable_params*], [],
)


Compared to study.11's #results.trainable_params_breakdown.W_ee trainable params (W\_ee only),
this adds (results?.trainable_params_breakdown?.W_ei ?? 0) + (results?.trainable_params_breakdown?.W_ie ?? 0) E↔I parameters — a ~3% increase. The optimizer can now learn
both the classification pathway and the inhibitory loop strength.


== Gradient flow


The surrogate gradient flows through the full E↔I loop. Unlike study.11
where W\_ei and W\_ie gradients were computed but discarded, here they drive
weight updates. The optimizer can strengthen or weaken the PING loop —
potentially learning to use inhibition constructively (e.g. sharpening
hidden representations) rather than treating it as fixed interference.


= Results



== Results Snapshot


#table(
  columns: 2,
  [Metric], [Value],
  [Best test accuracy], [#results.best_test_accuracy% (epoch #results.best_test_accuracy_epoch)],
  [Final test accuracy], [#results.final_test_accuracy%],
  [Best test loss], [#results.best_test_loss (epoch #results.best_test_loss_epoch)],
  [Final test loss], [#results.final_test_loss],
  [Trainable params], [#results.trainable_params],
  [Training time], [#results.elapsed_secondss],
)



== Comparison across studies


#table(
  columns: 4,
  [Study], [Architecture], [Trained weights], [Best Accuracy],
  [study.8 (standard SNN)], [E-only], [W\_ee], [64.2%],
  [study.10 (PING inference)], [E + I\_global], [None (loaded)], [55.3%],
  [study.11 (PING, W\_ee only)], [E + I\_global], [W\_ee], [61.9%],
  [*study.12 (PING, all weights)*], [*E + I\_global*], [*W\_ee + W\_ei + W\_ie*], [*#results.best_test_accuracy%*],
)



== Training dynamics


// Gallery: loss_train*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/loss_traindark.png"), caption: [Per-iteration training loss over 5 epochs.])


// Gallery: loss_test*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/loss_testdark.png"), caption: [Test loss evaluated at the end of each epoch on 1000 held-out samples.])


Like study.11, the test loss decreases monotonically from
#results.test_losses_per_epoch.at(0) to #results.final_test_loss across all
#results.epochs epochs with no overfitting.

// Gallery: accuracy_dark.png, accuracy_light.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/accuracy_dark.png"), caption: [Per-iteration training accuracy.])



== Per-class accuracy


// Gallery: accuracy_per_class*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/accuracy_per_classdark.png"), caption: [Per-class test accuracy with all weights trained.])



== Confusion matrix


// Gallery: confusion*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/confusiondark.png"), caption: [Confusion matrix on 1000 test samples.])



== Output layer rasters


// Gallery: raster_output_all_all*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/raster_output_all_alldark.png"), caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class.])



== Hidden, inhibitory, and output layer rasters


// Gallery: raster_layers_digit_*.png
// #figure(image("_artifacts/study.12-ping-snn-all-training/raster_layers_digit_dark.png"), caption: [Full signal path rasters for each digit (0–9). Top: I_global. Middle: E_hid. Bottom: E_out. Dashed lines separate layers.])



= Discussion



== Training E↔I weights does not help (yet)


At #results.best_test_accuracy%, study.12 performs comparably to study.11 (61.9%). Training
the additional 22,308 E↔I parameters does not meaningfully improve
classification. This suggests that at this scale (13 I neurons, 5 epochs,
5000 samples), the E↔I weights have limited influence on the classification
objective — the loss gradient with respect to W\_ei and W\_ie is small
compared to W\_ee.


== The PING loop may be too small to matter


With only #config.nodes.at(4).size inhibitory neurons acting on #config.nodes.at(2).size hidden excitatory
neurons, the inhibitory feedback is relatively weak. The optimizer may find
it easier to simply work around the inhibition (via W\_ee) than to shape it
(via W\_ei/W\_ie). A larger I population or stronger initial E↔I coupling
could make the inhibitory pathway more influential and give the optimizer
more to work with.


== Next steps


- *Scale up I population*: More I neurons would create stronger PING
  dynamics and give W\_ei/W\_ie more impact on the hidden representations.
- *Separate learning rates*: The E↔I weights may need a different
  learning rate than W\_ee — the gradient magnitudes through the inhibitory
  loop are likely much smaller.
- *Burn-in period*: Let PING rhythms stabilize before counting output
  spikes for classification.
- *More epochs*: 5 epochs may not be enough for the E↔I weights to
  converge, given their indirect influence on the loss.
