// title: study.13-ping-snn-training-voltage-readout
// date: 2026-03-04
// description: Training a PING-SNN on MNIST with voltage readout instead of spike counts, eliminating argmax tie-breaking bias.

#let config = json("_artifacts/study.13-ping-snn-training-voltage-readout/config.json")
#let results = json("_artifacts/study.13-ping-snn-training-voltage-readout/results.json")






= Summary


This study uses the same architecture and training setup as study.12 (all
weights trained via surrogate gradient BPTT) but changes the readout
mechanism. Instead of counting output spikes and taking argmax, we use the
mean membrane voltage of the output neurons as logits.

The motivation: in spike-count decoding, when two output neurons fire the
same number of spikes, `argmax` returns the lowest index — systematically
disadvantaging higher-numbered digit classes. Voltage readout produces
continuous-valued logits, eliminating ties entirely.


= Architecture


#figure(
  image("_artifacts/study.13-ping-snn-training-voltage-readout/graph_dark.png"),
  caption: [Network topology with initialization parameters.],
)



The architecture is identical to study.12:

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
  [`readout`], [#results.readout],
)



= Method



== Readout change


In studies 8–12, the forward pass decoded output neuron activity via spike
counts:

$
  "logits"_c = sum_(t=t_"burn-in")^(T) s_(c)(t)
$


where $s_c(t) in \{0, 1\}$ is the spike indicator for output neuron $c$ at
timestep $t$. This produces integer-valued logits. When two classes have
equal spike counts, `torch.argmax` returns the lowest index — biasing
classification toward digit 0 and against digit 9.

Study.13 replaces this with voltage readout:

$
  "logits"_c = \frac{1}{T - t_"burn-in"} sum_(t=t_"burn-in")^(T) V_(c)(t)
$


where $V_c(t)$ is the membrane voltage of output neuron $c$. Because
voltage is continuous, ties are vanishingly unlikely and all classes
compete on equal footing.


== What is trained


Same as study.12 — all three weight matrices receive gradient updates:

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
  [Readout], [#results.readout],
  [Trainable params], [#results.trainable_params],
  [Training time], [#results.elapsed_secondss],
)



== Comparison across studies


#table(
  columns: 4,
  [Study], [Readout], [Trained weights], [Best Accuracy],
  [study.8 (standard SNN)], [spike count], [W\_ee], [64.2%],
  [study.10 (PING inference)], [spike count], [None (loaded)], [55.3%],
  [study.11 (PING, W\_ee only)], [spike count], [W\_ee], [61.9%],
  [study.12 (PING, all weights)], [spike count], [W\_ee + W\_ei + W\_ie], [62.9%],
  [*study.13 (voltage readout)*], [*voltage*], [*W\_ee + W\_ei + W\_ie*], [*#results.best_test_accuracy%*],
)



== Training dynamics


#figure(
  image("_artifacts/study.13-ping-snn-training-voltage-readout/loss_train_dark.png"),
  caption: [Per-iteration training loss over 5 epochs.],
)



#figure(
  image("_artifacts/study.13-ping-snn-training-voltage-readout/loss_test_dark.png"),
  caption: [Test loss evaluated at the end of each epoch on 1000 held-out samples.],
)



Test loss decreases from
#results.test_losses_per_epoch.at(0) to a minimum of #results.best_test_loss at
epoch #results.best_test_loss_epoch, then rises slightly — suggesting mild overfitting
in later epochs.

#figure(
  grid(
    columns: 2,
    gutter: 4pt,
    image("_artifacts/study.13-ping-snn-training-voltage-readout/accuracy_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/accuracy_light.png"),
  ),
  caption: [Per-iteration training accuracy.],
)




== Per-class accuracy


#figure(
  image("_artifacts/study.13-ping-snn-training-voltage-readout/accuracy_per_class_dark.png"),
  caption: [Per-class test accuracy with voltage readout.],
)



This is the key result. With spike-count readout (study.12), per-class
accuracy showed a strong bias toward low-index digits: digit 0 reached
~80% while digit 9 scored just 25.5%. With voltage readout, the
distribution is dramatically more balanced — all classes compete on
continuous-valued logits with no tie-breaking advantage.


== Confusion matrix


#figure(
  image("_artifacts/study.13-ping-snn-training-voltage-readout/confusion_dark.png"),
  caption: [Confusion matrix on 1000 test samples.],
)




== Output layer rasters


#figure(
  image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical example per digit class.],
)




== Hidden, inhibitory, and output layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.13-ping-snn-training-voltage-readout/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit (0–9). Top: I_global. Middle: E_hid. Bottom: E_out. Dashed lines separate layers.],
)




= Discussion



== Voltage readout fixes per-class bias


The headline result is not overall accuracy (62.0% vs 62.9%) but the
per-class distribution. Spike-count decoding with argmax creates a
systematic bias: when output neurons fire at similar rates, the lowest
index wins. This makes digit 0 easy and digit 9 hard. Voltage readout
eliminates this artifact by providing continuous logits — the network
must actually learn to differentiate classes rather than relying on
tie-breaking luck.


== Overall accuracy is comparable


At #results.best_test_accuracy%, study.13 performs similarly to study.12 (62.9%).
The slight decrease is expected: spike-count argmax artificially inflated
accuracy for low-index digits. With fairer decoding, the network's true
discriminative ability is revealed — and it is consistent across all 10
classes rather than concentrated in a few.


== Voltage readout as default


Going forward, voltage readout should be the default for PING-SNN
classification experiments. It provides a more honest evaluation of
network performance and gives the optimizer a smoother loss landscape
(continuous logits vs integer spike counts).


== Next steps


- *Voltage readout + more data*: Run with full 60K MNIST to see if
  the fairer decoding translates to higher absolute accuracy at scale.
- *Burn-in period*: The first few milliseconds of voltage may be
  dominated by transient dynamics. A burn-in window could improve
  signal quality.
- *Scale up*: Larger hidden layer and I population to see if PING
  dynamics contribute to classification when properly decoded.
