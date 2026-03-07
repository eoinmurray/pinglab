// title: study.14-ping-snn-poisson-input
// date: 2026-03-05
// description: Training a PING-SNN on MNIST with Poisson input encoding instead of tonic current injection.

#let config = json("_artifacts/study.14-ping-snn-poisson-input/config.json")
#let results = json("_artifacts/study.14-ping-snn-poisson-input/results.json")






= Summary


This study replaces the tonic (constant current) input encoding used in
studies 8–13 with Poisson spike trains. Everything else is identical to
study.13: all weights trained, voltage readout.

In tonic encoding, each input neuron receives a constant current
proportional to its pixel intensity at every timestep — there is no
temporal structure in the input. In Poisson encoding, each input neuron
fires a stochastic spike at each timestep with probability equal to the
pixel intensity, injecting genuine temporal variability.

The question: does giving the network temporal input structure improve
classification?


= Architecture


#figure(
  image("_artifacts/study.14-ping-snn-poisson-input/graph_dark.png"),
  caption: [Network topology with initialization parameters.],
)



Identical to study.13:

#table(
  columns: 4,
  [Layer], [Population], [Size], [Role],
  [Input], [`E_in`], [784], [One neuron per pixel, driven by Poisson spike trains],
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
  [`encoding`], [#results.encoding],
  [`readout`], [#results.readout],
)



= Method



== Encoding change


In studies 8–13, each input neuron received a constant (tonic) current at
every timestep:

$
  I_n(t) = "pixel"_n times "scale"   forall t
$


This is deterministic — the same image always produces the same external
drive. All temporal dynamics emerge purely from recurrent network activity.

Study.14 replaces this with Poisson encoding:

$
  I_n(t) = "scale" times "Bernoulli"("pixel"_n)
$


At each timestep, each input neuron independently fires a current pulse
of amplitude `scale` with probability equal to the pixel intensity
(clamped to [0, 1]). A bright pixel (intensity 0.9) produces a spike
90% of the time; a dark pixel (0.1) spikes 10% of the time. The expected
total current over $T$ steps matches the tonic case, but the input now
carries stochastic temporal structure.

This matters because:

- The same image produces a **different spike train on every forward
  pass**, acting as implicit data augmentation during training.
- The network receives *temporally varying input* that it can integrate
  over time, rather than a static DC drive.
- Each forward pass during training sees a different realization, which
  regularizes the learned weights and may improve generalization.


== What is trained


Same as studies 12–13 — all three weight matrices:

#table(
  columns: 3,
  [Matrix], [Params], [Connection],
  [`W_ee`], [#results.trainable_params_breakdown.W_ee], [E\_in→E\_hid and E\_hid→E\_out (structural mask)],
  [`W_ei`], [#results.trainable_params_breakdown.W_ei], [E\_hid→I\_global],
  [`W_ie`], [#results.trainable_params_breakdown.W_ie], [I\_global→E\_hid],
  [*Total*], [*#results.trainable_params*], [],
)



= Results



== Results Snapshot


#table(
  columns: 2,
  [Metric], [Value],
  [Best test accuracy], [#results.best_test_accuracy% (epoch #results.best_test_accuracy_epoch)],
  [Final test accuracy], [#results.final_test_accuracy%],
  [Best test loss], [#results.best_test_loss (epoch #results.best_test_loss_epoch)],
  [Final test loss], [#results.final_test_loss],
  [Encoding], [#results.encoding],
  [Readout], [#results.readout],
  [Trainable params], [#results.trainable_params],
  [Training time], [#results.elapsed_seconds],
)



== Comparison across studies


#table(
  columns: 5,
  [Study], [Encoding], [Readout], [Trained weights], [Best Accuracy],
  [study.8 (standard SNN)], [tonic], [spike count], [W\_ee], [64.2%],
  [study.11 (PING, W\_ee only)], [tonic], [spike count], [W\_ee], [61.9%],
  [study.12 (PING, all weights)], [tonic], [spike count], [W\_ee + W\_ei + W\_ie], [62.9%],
  [study.13 (voltage readout)], [tonic], [voltage], [W\_ee + W\_ei + W\_ie], [62.0%],
  [*study.14 (Poisson input)*], [*Poisson*], [*voltage*], [*W\_ee + W\_ei + W\_ie*], [*#results.best_test_accuracy%*],
)



== Training dynamics


#figure(
  image("_artifacts/study.14-ping-snn-poisson-input/loss_train_dark.png"),
  caption: [Per-iteration training loss over 5 epochs.],
)



Training loss is noticeably noisier than study.13 — expected because each
forward pass draws a different Poisson realization of the same image. The
optimizer never sees the same input twice, which acts as stochastic
regularization.

#figure(
  image("_artifacts/study.14-ping-snn-poisson-input/loss_test_dark.png"),
  caption: [Test loss evaluated at the end of each epoch on 1000 held-out samples.],
)



Test loss reaches its minimum of #results.best_test_loss at
epoch #results.best_test_loss_epoch, then oscillates in later
epochs. The non-monotonic pattern reflects the stochastic nature of both
training and evaluation under Poisson encoding.

#figure(
  grid(
    columns: 2,
    gutter: 4pt,
    image("_artifacts/study.14-ping-snn-poisson-input/accuracy_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/accuracy_light.png"),
  ),
  caption: [Per-iteration training accuracy.],
)




== Per-class accuracy


#figure(
  image("_artifacts/study.14-ping-snn-poisson-input/accuracy_per_class_dark.png"),
  caption: [Per-class test accuracy with Poisson input encoding.],
)



Per-class accuracy remains balanced (no argmax bias from voltage readout),
with the overall distribution shifted upward compared to study.13.


== Confusion matrix


#figure(
  image("_artifacts/study.14-ping-snn-poisson-input/confusion_dark.png"),
  caption: [Confusion matrix on 1000 test samples.],
)




== Input rasters (Poisson spike trains)


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_00_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_01_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_02_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_03_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_04_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_05_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_06_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_07_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_08_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_input_digit_09_dark.png"),
  ),
  caption: [Poisson input spike trains for each digit (0–9). Each dot is a current pulse injected into an input neuron. Bright pixels produce dense spike trains; dark pixels are sparse.],
)



The input rasters show the stochastic nature of Poisson encoding. Unlike
tonic encoding (where every timestep is identical), each timestep has a
different pattern of active input neurons. Digit 1 produces the sparsest
input (~12K spikes over 100 ms) because it has the fewest bright pixels,
while digit 0 produces the densest (~44K spikes).


== Output neuron voltage traces


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_00_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_01_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_02_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_03_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_04_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_05_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_06_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_07_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_08_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/voltage_output_digit_09_dark.png"),
  ),
  caption: [Membrane voltage traces for the 10 output neurons over 100 ms. Dashed grey line is the spike threshold (V_th = -50 mV). The network classifies via sub-threshold voltage separation.],
)



The voltage traces reveal how the network actually classifies: **output
neurons never reach spike threshold** ($V_("th")$ = -50 mV). Instead, the
network has learned to produce different mean voltages across the 10
output neurons. The class with the highest mean voltage wins. This is a
direct consequence of voltage readout — the cross-entropy loss on mean
voltage has no incentive to push neurons above threshold.


== Output layer rasters


#figure(
  image("_artifacts/study.14-ping-snn-poisson-input/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class. Mostly empty — the network classifies via voltage, not spikes.],
)



The output rasters are nearly empty, confirming the voltage trace
analysis. Across all 10 digit classes, the output layer produces
essentially zero spikes. This is expected and correct — the trained
network has no reason to spike when the loss function only cares about
mean membrane voltage.


== Hidden, inhibitory, and output layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.14-ping-snn-poisson-input/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit (0–9). Top: I_global. Middle: E_hid. Bottom: E_out. Dashed lines separate layers.],
)



The hidden layer (E\_hid) and inhibitory layer (I\_global) are both
active — spikes propagate through the network and the PING loop
operates. Only the output layer is quiescent, having learned to encode
class information in voltage rather than spike rate.


= Discussion



== Poisson encoding improves accuracy


At #results.best_test_accuracy%, study.14 is the best-performing
PING-SNN to date, surpassing study.13's 62.0% and approaching study.8's
64.2% E-only baseline.

The improvement likely comes from implicit data augmentation: each
forward pass generates a unique stochastic spike train from the same
image, effectively multiplying the training set size. The network must
learn features that are robust to input noise.


== Voltage readout produces a non-spiking output layer


The most striking finding is that the output neurons never spike. The
voltage readout objective has learned a sub-threshold classification
strategy: different digit classes produce different voltage profiles in
the output neurons, but none cross the spike threshold. This is
biologically unusual — real neurons communicate via spikes — but
mathematically valid for the training objective.

This raises a design question: should the loss function encourage
spiking in the output layer? A hybrid objective (voltage readout +
spike rate penalty) could push the network toward biologically
plausible solutions while preserving the continuous-logit advantages.


== Training is noisier but converges


The per-iteration loss curve is substantially noisier than study.13,
but the epoch-level test accuracy still converges. The stochastic input
acts as a regularizer — the network cannot memorize exact input patterns
and must instead learn generalizable features.


== Next steps


- *Spike-count readout comparison*: Re-run with spike-count readout
  to see if Poisson encoding also helps when the network must produce
  output spikes.
- *More epochs*: With stochastic input, the network may benefit from
  longer training — the implicit augmentation means each epoch is
  effectively novel.
- *Full MNIST*: Scale to 60K training samples to see how Poisson
  encoding interacts with larger datasets.
- *Burn-in period*: PING rhythms may need time to stabilize before the
  voltage readout window begins.
