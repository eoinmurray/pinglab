// title: study.11-ping-snn-ee-training
// date: 2026-03-02
// description: Training a PING-SNN on MNIST with surrogate gradient BPTT — only W_ee is trained while E↔I weights remain fixed.

#let config = json("_artifacts/study.11-ping-snn-ee-training/config.json")
#let results = json("_artifacts/study.11-ping-snn-ee-training/results.json")






= Summary


This study trains a spiking neural network with an active PING
(Pyramidal-Interneuron Network Gamma) loop on MNIST digit classification.
Unlike study.8, which trains a purely excitatory feedforward SNN, and
study.10, which transplants study.8's trained weights into a PING network
for inference, this study trains W\_ee from scratch with the inhibitory
population active during training.

The core question: can the optimizer learn to classify through the PING
dynamics, or does the E↔I feedback loop disrupt gradient-based learning?

Only W\_ee is trained — the E→I and I→E weights remain at their random
initial values throughout.


= Architecture


#figure(
  image("_artifacts/study.11-ping-snn-ee-training/graph_dark.png"),
  caption: [Network topology with initialization parameters.],
)



The network extends study.8's three-layer excitatory chain with a global
inhibitory population forming a PING loop with E\_hid:

#table(
  columns: 4,
  [Layer], [Population], [Size], [Role],
  [Input], [`E_in`], [784], [One neuron per pixel, driven by rate-coded image],
  [Hidden], [`E_hid`], [#config.nodes.at(2).size], [Learned feature extraction],
  [Output], [`E_out`], [10], [One neuron per digit class],
  [Inhibitory], [`I_global`], [#config.nodes.at(4).size], [PING feedback inhibition on E\_hid],
)


The PING loop: E\_hid excites I\_global (E→I, mean=0.3, std=0.1), and
I\_global inhibits E\_hid (I→E, mean=0.15, std=0.05). Both connections
have 1 ms delays.


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



== Training setup


The training procedure matches study.8 exactly — Adam optimizer, cross-entropy
loss on spike-count logits, #config.meta.epochs epochs, #config.meta.subset_size training samples — with
one critical difference: the inhibitory population is active during every
forward pass. During each 100 ms simulation, E\_hid drives I\_global, which
fires back inhibition onto E\_hid, suppressing some excitatory activity before
it can propagate to E\_out.


== What is trained


Only `W_ee` (the E→E weight matrix) receives gradient updates. This single
matrix encodes both the E\_in→E\_hid and E\_hid→E\_out connections via a
structural mask. The E→I and I→E weights are randomly initialized and frozen
— the optimizer must learn to classify *through* the fixed inhibitory
dynamics rather than adapting them.


== Surrogate gradient through PING


The surrogate gradient (fast-sigmoid) flows through both E and I neuron
dynamics. When an E\_hid spike causes an I\_global spike that suppresses a
subsequent E\_hid spike, the gradient traces this full causal chain:
W\_ee → E\_hid spike → g\_e on I → I spike → g\_i on E\_hid → suppressed
E\_hid → reduced E\_out count → loss. This means W\_ee gradients implicitly
account for the inhibitory feedback.


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



== Comparison with studies 8 and 10


#table(
  columns: 4,
  [Study], [Architecture], [Training], [Best Accuracy],
  [study.8 (standard SNN)], [E-only], [Train W\_ee], [64.2%],
  [study.10 (PING inference)], [E + I\_global], [Load study.8 W\_ee], [55.3%],
  [*study.11 (PING training)*], [*E + I\_global*], [*Train W\_ee with PING*], [*#results.best_test_accuracy%*],
)


Training through the PING dynamics recovers most of the accuracy lost by
transplanting E-only weights into a PING network. The optimizer learns
W\_ee values that are compatible with the inhibitory feedback, rather than
being disrupted by it.


== Training dynamics


#figure(
  image("_artifacts/study.11-ping-snn-ee-training/loss_train_dark.png"),
  caption: [Per-iteration training loss over 5 epochs.],
)



#figure(
  image("_artifacts/study.11-ping-snn-ee-training/loss_test_dark.png"),
  caption: [Test loss evaluated at the end of each epoch on 1000 held-out samples.],
)



The test loss decreases monotonically from #results.test_losses_per_epoch.at(0) to #results.final_test_loss across all
#results.epochs epochs — no overfitting uptick as seen in study.8. The inhibitory feedback
may act as an implicit regularizer, preventing the network from memorizing
the small training set.

#figure(
  grid(
    columns: 2,
    gutter: 4pt,
    image("_artifacts/study.11-ping-snn-ee-training/accuracy_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/accuracy_light.png"),
  ),
  caption: [Per-iteration training accuracy.],
)




== Per-class accuracy


#figure(
  image("_artifacts/study.11-ping-snn-ee-training/accuracy_per_class_dark.png"),
  caption: [Per-class test accuracy with PING loop active during training.],
)




== Confusion matrix


#figure(
  image("_artifacts/study.11-ping-snn-ee-training/confusion_dark.png"),
  caption: [Confusion matrix on 1000 test samples.],
)




== Output layer rasters


#figure(
  image("_artifacts/study.11-ping-snn-ee-training/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class.],
)




== Hidden, inhibitory, and output layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.11-ping-snn-ee-training/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit (0–9). Top: I_global. Middle: E_hid. Bottom: E_out. Dashed lines separate layers.],
)



The raster plots show PING dynamics during trained inference. Unlike
study.10 where the transplanted weights sometimes produced chaotic or
suppressed activity, the PING-trained network should show more structured
E↔I alternation — the optimizer has shaped W\_ee to produce spike patterns
that work *with* the inhibitory feedback rather than against it.


= Discussion



== Training through inhibition works


The key result is that surrogate gradient BPTT successfully flows through
the full E↔I loop. The optimizer finds W\_ee values that produce correct
classifications despite the fixed inhibitory interference. At
#results.best_test_accuracy%, the PING-trained SNN recovers most of the gap between the
E-only SNN (64.2%) and the weight-transplant PING SNN (55.3%).


== Inhibition as regularization


The monotonically decreasing test loss is notable. Study.8 shows clear
overfitting (test loss rises at epoch 5), while study.11 does not. The PING
loop may function as a stochastic regularizer — the random E↔I weights
inject structured noise into the hidden representations, preventing the
network from relying on fragile spike-count patterns that don't generalize.


== Next steps


- *Train E↔I weights*: Currently the E→I and I→E connections are random
  and frozen. Making them trainable could allow the network to learn optimal
  inhibitory dynamics for classification.
- *Burn-in period*: Adding a burn-in window would let PING rhythms
  stabilize before counting output spikes, potentially improving accuracy.
- *Larger I population*: The current #config.nodes.at(4).size I neurons may be too few to
  establish strong gamma oscillations. Scaling up could reveal clearer
  rhythmic structure.
