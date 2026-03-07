// title: study.8-standard-snn-ff-training
// date: 2026-02-27
// description: Trainable feedforward SNN (784-64-10) on MNIST via surrogate gradient BPTT.

#let config = json("_artifacts/study.8-standard-snn-ff-training/config.json")
#let results = json("_artifacts/study.8-standard-snn-ff-training/results.json")






= Summary


This study trains a biophysical spiking neural network to classify MNIST digits
using surrogate gradient descent. Unlike the ANN baseline in study.7, every
neuron here is a conductance-based leaky integrate-and-fire (LIF) cell
simulated at millisecond resolution. The network learns by backpropagating
through time (BPTT) using a fast-sigmoid surrogate for the non-differentiable
Heaviside spike function.

The core question: can a biophysically constrained SNN, running in pinglab's
simulator with real membrane dynamics, learn to classify handwritten digits
end-to-end?


= Architecture


#figure(
  image("_artifacts/study.8-standard-snn-ff-training/graph_dark.png"),
  caption: [Network topology with initialization parameters.],
)




The network is a three-layer feedforward excitatory chain with no inhibitory neurons:

#table(
  columns: 4,
  [Layer], [Population], [Size], [Role],
  [Input], [`E_in`], [784], [One neuron per pixel, driven by rate-coded image],
  [Hidden], [`E_hid`], [64], [Learned feature extraction],
  [Output], [`E_out`], [10], [One neuron per digit class],
)


Edges connect `E_in → E_hid` and `E_hid → E_out` only. A *structural weight mask*
enforces this topology during training — gradients are zeroed outside the
defined edge blocks so the optimizer cannot create shortcut connections (e.g.
`E_in → E_out` directly), which would bypass the hidden layer entirely.


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



== Encoding


Each 28x28 MNIST image is flattened to 784 pixel values and injected as a
constant excitatory conductance into the `E_in` population for the full
simulation window ($T = 100$ ms). This is a *rate code* — brighter pixels
produce stronger sustained drive, causing input neurons to fire at higher
rates. The pixel values are scaled by `input_scale` = #config.meta.input_scale before
injection.


== Surrogate gradient


The LIF spike mechanism is a Heaviside step function, which has zero gradient
almost everywhere. We replace it during the backward pass with a fast-sigmoid
surrogate:

$
  tilde(sigma)(x) = frac(x, 2(1 + |x|)^2)
$


This provides a smooth gradient signal that allows BPTT to flow through the
spike times back to the synaptic weights.


== Decoding


Classification uses *spike-count decoding*: the total number of spikes
emitted by each of the 10 output neurons over the full 100 ms window is taken
as the logit for that class. The predicted digit is whichever output neuron
fires the most. Cross-entropy loss is applied to these spike-count logits.


== Weight constraints


All excitatory-to-excitatory weights are clamped to be non-negative
(`nonnegative_weights: true`), consistent with Dale's law — excitatory
synapses can only excite. Initial weights are drawn from
$cal(N)(mu=0.01, sigma=0.1)$, then clamped and scaled by $1/sqrt(N_"src")$ to
normalize total synaptic drive.


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
  [Training time], [#results.elapsed_seconds],
)



== Training dynamics


#figure(
  image("_artifacts/study.8-standard-snn-ff-training/loss_train_dark.png"),
  caption: [Per-iteration training loss over 5 epochs (~1560 batches of 16). The loss drops from ~2.3 (chance) to ~1.0 with high per-batch variance typical of small-batch SNN training.],
)



Figure 1.1 shows the training loss across all iterations. The initial
loss is approximately 2.3, consistent with random 10-class prediction
($-ln(0.1) approx 2.3$). Over #results.epochs epochs the loss drops to a running average
around #results.final_train_loss, though individual batches still show large swings — a consequence
of the small batch size (#config.meta.batch_size) and the discrete, noisy nature of spike-count
logits.

#figure(
  image("_artifacts/study.8-standard-snn-ff-training/loss_test_dark.png"),
  caption: [Test loss evaluated at the end of each epoch on 1000 held-out samples.],
)



Figure 1.2 shows the test loss per epoch. It drops to a
minimum of #results.best_test_loss at epoch #results.best_test_loss_epoch, ending at #results.final_test_loss at epoch #results.epochs — an
early sign of overfitting on the #str(results.train_samples)-sample training subset.

#figure(
  image("_artifacts/study.8-standard-snn-ff-training/accuracy_dark.png"),
  caption: [Per-iteration training accuracy. Batch-level accuracy is noisy (batch size 16) but the trend clearly rises from ~10% to a median around 60–70%.],
)



Figure 1.3 plots per-batch training accuracy. The trend rises from chance
(~10%) to a median in the 60–70% range by epoch #results.epochs, with individual batches
occasionally hitting 90–100%. The best epoch-level test accuracy is #results.best_test_accuracy% at
epoch #results.best_test_accuracy_epoch. For context, a simple ANN achieves ~97% on full MNIST — the SNN is
working with #results.train_samples training samples, only #config.nodes.at(2).size ?? 64 hidden neurons, and the hard
constraint of spike-count decoding.


== Output layer rasters


#figure(
  image("_artifacts/study.8-standard-snn-ff-training/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class. Each panel shows all 10 output neurons over 100 ms. The y-axis is the output neuron index (0–9).],
)



Figure 2.1 shows the output layer response to one canonical test image per
digit class. Several patterns are visible:

- *Digit 0*: Neurons 0, 5, and 6 fire regularly. Neuron 0 and 5 are active
  across the full window with neuron 0 accumulating the most spikes — a
  correct classification.
- *Digit 1*: Very sparse output — only neurons 1 and 7 fire, each with 1–2
  spikes. This extreme sparsity means classification is fragile; a single
  extra spike from a competing neuron would flip the prediction.
- *Digit 7*: Neuron 7 dominates with dense periodic firing, while neurons 5
  and 9 contribute scattered spikes. A clear and correct winner.
- *Digits 5, 8, 9*: Multiple output neurons fire with similar spike counts,
  illustrating the tie-breaking limitation of spike-count decoding. When two
  neurons have equal counts, `argmax` breaks ties in favor of the lower
  index — an arbitrary choice that hurts accuracy.


== Hidden and output layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.8-standard-snn-ff-training/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit (0–9). Top region: E_hid (64 neurons, subsampled to 64 for display). Bottom region: E_out (10 neurons, labeled 0–9 on the right margin). Dashed line separates layers.],
)



Figure 2.2 shows the complete hidden + output layer activity for each
canonical digit. The input layer (784 neurons) is omitted since it directly
mirrors the pixel intensities. Key observations:

*Signal propagation is working.* The hidden layer shows clear spiking driven
by the input, and the output layer fires in response to hidden layer activity.
Spikes in `E_out` are delayed relative to `E_hid` by roughly 10–20 ms,
consistent with the 1 ms axonal delay plus the time needed for synaptic
integration to reach threshold.

*Hidden layer patterns differ across digits.* Digit 0 produces dense,
broad activation across most hidden neurons — consistent with the large
spatial extent of a zero's stroke. Digit 1, by contrast, activates very few
hidden neurons and produces only a handful of output spikes. The "1" digit has
the smallest number of active pixels, so fewer input neurons fire, producing
weaker drive through the network.

*Temporal structure emerges.* The hidden layer shows quasi-periodic bursting
at roughly 8–12 ms intervals across many digit classes. This rhythm is not
imposed by the input (which is constant-rate) — it arises from the interplay
of membrane time constants, refractory periods ($t_("ref") = 3$ ms for E
cells), and the recurrent feedback through the synaptic conductance dynamics
($tau_("ampa") = 2$ ms).

*Output sparsity varies dramatically.* Digit 0 drives ~15–20 output spikes
across multiple neurons, while digit 1 produces only 2–3. This large dynamic
range means the cross-entropy loss gradient is dominated by high-activity
digits, potentially starving low-activity classes of learning signal.


= Discussion



== What works


This study demonstrates that pinglab's simulator can be used for end-to-end
gradient-based learning. The surrogate gradient flows through the full
biophysical simulation — membrane dynamics, synaptic conductances, delays,
refractory periods — and produces meaningful weight updates. The structural
mask correctly prevents gradient leaking through undefined connections.

The #results.best_test_accuracy% test accuracy on a #str(results.train_samples)-sample subset confirms the network is
learning real digit features, not memorizing. The hidden layer develops
digit-specific activation patterns visible in the rasters.


== Limitations and next steps


*Spike-count decoding is coarse.* With only 100 ms and 10 output neurons,
the maximum possible spike count per neuron is ~30 (limited by the 3 ms
refractory period). Ties are common (Figure 2.1) and resolved arbitrarily.
Switching to membrane-potential readout would provide continuous-valued logits
and eliminate ties entirely.

*No inhibition.* The current architecture uses only excitatory populations.
Adding inhibitory interneurons could sharpen representations by enforcing
competition between hidden features (lateral inhibition) or between output
classes (winner-take-all dynamics).

*Small scale.* #config.nodes.at(2).size ?? 64 hidden neurons and #results.train_samples training samples are deliberately
small to keep iteration fast. Scaling to 256+ hidden neurons and the full 60K
training set would likely improve accuracy substantially.

*Overfitting signal.* The test loss uptick at epoch #results.epochs (Figure 1.2) suggests
regularization or early stopping would help, especially at this small data
scale.
