// title: study.19-ping-regulariser-on-mnist
// date: 2026-03-06T09:51:06Z
// description: study.19-ping-regulariser-on-mnist

#let config = json("_artifacts/study.19-ping-regulariser-on-mnist/config.json")
#let results = json("_artifacts/study.19-ping-regulariser-on-mnist/results.json")






= Summary


This study solves the gradient flow bottleneck that limited all previous
PING training runs. Conductance-based LIF models amplify gradients by
~65x per timestep through the $(E_e - V)$ driving force, forcing
previous studies to detach the spike buffer and restrict training to
2,560 of 1.2M parameters.

We introduce a *surrogate Jacobian* on $C_m$: the forward pass uses
the real $C_m = 1$, but the backward pass pretends
$C_m$ = #results.cm_backward_scale, shrinking the
explosive Jacobian entry from 65 to
n/a.
Combined with *per-parameter gradient clipping*, this enables stable
full BPTT — all #results.trainable_params parameters
across W\_ee, W\_ei, and W\_ie receive gradient.

*Result:* #results.best_test_accuracy% test accuracy
on #results.test_samples samples — a
20-point improvement over the 75.6% detached-spike baseline. Test loss
is still decreasing at epoch #results.epochs,
suggesting further gains with more training.


= Architecture


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/graph_dark.png"),
  caption: [Network topology with initialization parameters.],
)



Identical to studies 13–14:

#table(
  columns: 4,
  [Layer], [Population], [Size], [Role],
  [Input], [`E_in`], [784], [One neuron per pixel, driven by Poisson spike trains],
  [Hidden], [`E_hid`], [#config.nodes.at(2).size], [Feature extraction — trained via surrogate Jacobian BPTT],
  [Output], [`E_out`], [10], [One neuron per digit class],
  [Inhibitory], [`I_global`], [#config.nodes.at(4).size], [PING feedback inhibition on E\_hid — also trained],
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
  [`meta.readout`], [#config.meta.readout],
  [`meta.readout_alpha`], [#config.meta.readout_alpha],
  [`meta.cm_backward_scale`], [#config.meta.cm_backward_scale],
  [`meta.max_grad_norm`], [#config.meta.max_grad_norm (per-parameter)],
  [`meta.subset_size`], [#config.meta.subset_size],
  [`encoding`], [#results.encoding],
)



= Method



== Encoding


Poisson encoding (introduced in study.14):

$
  I_n(t) = "scale" times "Bernoulli"("pixel"_n)
$


Each input neuron fires a current pulse with probability equal to its
pixel intensity at every timestep. The same image produces a different
spike train on every forward pass, acting as implicit data augmentation.


== What is trained


With `cm_backward_scale` = #results.cm_backward_scale
and `detach_spikes` = false, gradient flows through the full temporal
spike chain via the surrogate Jacobian. All weight matrices are in the
optimizer with *per-parameter gradient clipping* at
#config.meta.max_grad_norm.

#table(
  columns: 4,
  [Matrix], [Params], [Gradient?], [Connection],
  [`W_ee`], [#results.trainable_params_breakdown.W_ee], [yes], [E\_in→E\_hid and E\_hid→E\_out],
  [`W_ei`], [#results.trainable_params_breakdown.W_ei], [yes], [E\_hid→I\_global (PING loop)],
  [`W_ie`], [#results.trainable_params_breakdown.W_ie], [yes], [I\_global→E\_hid (PING loop)],
)



== Surrogate Jacobian on $C_m$



=== The problem


The conductance-based LIF voltage update couples voltage and
conductance through a multiplicative driving force:

$
  frac(partial V_(t+1), partial g_e) = frac(d t dot (E_e - V_t), C_m) = frac(1.0 times (0 - (-65)), 1.0) = 65
$


Each timestep multiplies the gradient by $approx 65$. Over 200
timesteps of BPTT, this is catastrophic — even 5 steps gives
$65^5 approx 1.2 times 10^9$. Previous studies detached the spike
buffer to prevent explosion, but this limited gradient to only the
2,560 parameters in the final E\_hid→E\_out block of W\_ee.


=== The solution


Pretend $C_m$ is $kappa$ times larger in the backward pass only. A
one-line identity trick on `dVdt`:

$
  tilde(f) = f / kappa + "detach"(f) dot (1 - 1/kappa)
$


Forward: $tilde(f) = f$ (the two terms sum to $f$). Backward:
$partial tilde(f)/partial f = 1/kappa$ (only the first term carries
gradient). The forward dynamics are completely unchanged.

With $kappa$ = #results.cm_backward_scale:

#table(
  columns: 3,
  [Jacobian entry], [True value], [Surrogate value],
  [$partial V'/partial g_e$], [65], [n/a],
  [$partial V'/partial g_i$], [−15], [n/a],
  [$partial V'/partial V$], [0.8], [n/a],
)



=== Per-parameter gradient clipping


The surrogate Jacobian alone is not enough. Even with $kappa = 100$,
the raw gradient norms differ by orders of magnitude across weight
matrices (W\_ee $gt.double$ W\_ie $gt.double$ W\_ei). Standard gradient clipping
computes the *total* norm across all parameters, so the largest
matrix dominates the clip direction — W\_ei and W\_ie effectively get
zero update.

Per-parameter clipping applies `clip_grad_norm_` to each weight matrix
independently, ensuring each gets a fair update regardless of scale.


=== What didn't work: $kappa = 10$


The first attempt used $kappa = 10$ with total (not per-parameter)
gradient clipping. This produced 25.6% accuracy — far worse than the
75.6% detached baseline. Analysis of the gradient norms revealed the
failure mode:

- W\_ee gradient norms hit *2 × 10¹³* at iteration ~30
- This single spike corrupted Adam's momentum estimates
- After the spike, W\_ee and W\_ei effectively stopped receiving
  meaningful updates
- Only W\_ie continued learning (shortest gradient path)
- Firing rates ran away to 200+ Hz — the network saturated

The fix was two-fold: increase $kappa$ from 10 to 100, and switch to
per-parameter gradient clipping.


= Results



== Results Snapshot


#table(
  columns: 2,
  [Metric], [Value],
  [Best test accuracy], [#results.best_test_accuracy% (epoch #results.best_test_accuracy_epoch)],
  [Final test loss], [#results.final_test_loss],
  [Encoding], [#results.encoding],
  [Readout], [#results.readout ($alpha$ = #results.readout_alpha)],
  [Trainable params], [#results.trainable_params],
  [Surrogate C\_m scale ($kappa$)], [#results.cm_backward_scale],
  [Spike detachment], [no],
  [Grad clip], [#config.meta.max_grad_norm per-parameter],
  [Training time], [n/a min],
  [Device], [#results.device (#results.runtime)],
)



== Comparison across studies


#table(
  columns: 5,
  [Study], [Encoding], [Readout], [Trained weights], [Best Accuracy],
  [study.8 (standard SNN)], [tonic], [spike count], [W\_ee (2,560)], [64.2%],
  [study.11 (PING, W\_ee)], [tonic], [spike count], [W\_ee (2,560)], [61.9%],
  [study.12 (PING, all\*)], [tonic], [spike count], [W\_ee (2,560)\*], [62.9%],
  [study.13 (voltage)], [tonic], [voltage], [W\_ee (2,560)\*], [62.0%],
  [study.14 (Poisson)], [Poisson], [voltage], [W\_ee (2,560)\*], [84.8%],
  [*study.15*], [*Poisson*], [*hybrid*], [*all (#results.trainable_params)*], [*#results.best_test_accuracy%*],
)


Studies 12–14 included W\_ei and W\_ie in the optimizer, but spike-buffer
detachment meant these weights received zero gradient. Only 2,560
params in W\_ee (E\_hid→E\_out) were actually trained.


== Hyperparameter sweep


#table(
  columns: 6,
  [$kappa$], [LR], [Grad clip], [Data], [Epochs], [Accuracy],
  [— (detach)], [3e-4], [total 1.0], [5k], [5], [75.6%],
  [10], [3e-4], [total 1.0], [5k], [5], [25.6%],
  [100], [1e-4], [per-param 1.0], [5k], [5], [90.5%],
  [*100*], [*1e-4*], [*per-param 1.0*], [*20k*], [*10*], [*#results.best_test_accuracy%*],
)



== Training dynamics


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/loss_train_dark.png"),
  caption: [Per-iteration training loss. Loss drops sharply after epoch 1, reaching near-zero by epoch 10.],
)



#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/loss_test_dark.png"),
  caption: [Test loss per epoch. Monotonically decreasing — still improving at epoch 10.],
)



#figure(
  grid(
    columns: 2,
    gutter: 4pt,
    image("_artifacts/study.19-ping-regulariser-on-mnist/accuracy_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/accuracy_light.png"),
  ),
  caption: [Per-iteration training accuracy. Rises from ~15% to 95%+ over the first two epochs, then stabilizes.],
)




== Gradient norms


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/grad_norms_dark.png"),
  caption: [L2 gradient norm per weight matrix. Per-parameter clipping keeps each matrix in a healthy range: W_ee (2–10), W_ei (0.01–0.07), W_ie (0.1–0.5).],
)



With per-parameter clipping and $kappa = 100$, all three weight
matrices receive stable, well-scaled gradients throughout training.
Compare this to the $kappa = 10$ run where W\_ee hit 10¹³ at iteration
30 and corrupted the optimizer.


== Firing rates


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/firing_rates_dark.png"),
  caption: [Mean firing rate (Hz) per excitatory layer. E_hid and E_out settle to ~6 Hz — healthy, sparse activity regulated by trained inhibition.],
)



E\_hid and E\_out firing rates start elevated (~25 Hz) and settle to
~6 Hz as training progresses. This is the opposite of the $kappa = 10$
run where rates ran away to 200+ Hz. The trained W\_ei and W\_ie
weights successfully regulate network activity through the PING loop.


== Per-class accuracy


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/accuracy_per_class_dark.png"),
  caption: [Per-class test accuracy. All digits between 93–98%.],
)



All 10 digit classes are well-classified with balanced accuracy.
Digit 1 leads at 98.2% (sparse, easy to distinguish), and digit 9
trails at 94.2%.


== Confusion matrix


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/confusion_dark.png"),
  caption: [Confusion matrix on 5,000 test samples. Strong diagonal with few off-diagonal errors.],
)




== Output layer rasters


#figure(
  image("_artifacts/study.19-ping-regulariser-on-mnist/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical sample per digit. Each digit activates its corresponding output neuron with sparse, discriminative spiking.],
)



The output rasters show clean, class-selective spiking. For digit 0,
neuron 0 fires; for digit 7, neuron 7 fires. This is genuine
spike-based classification — not voltage-only discrimination.


== Output neuron voltage traces


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_00_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_01_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_02_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_03_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_04_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_05_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_06_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_07_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_08_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_09_dark.png"),
  ),
  caption: [Membrane voltage traces for the 10 output neurons over 200 ms. The correct class neuron is pushed toward threshold while others remain subthreshold.],
)




== Input rasters (Poisson spike trains)


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_00_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_01_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_02_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_03_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_04_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_05_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_06_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_07_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_08_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_input_digit_09_dark.png"),
  ),
  caption: [Poisson input spike trains for each digit (0–9). Bright pixels produce dense spike trains; dark pixels are sparse.],
)




== Full layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit. Top: E_in (blue), middle: E_hid (white), bottom: E_out and I_global (red). PING oscillations visible in the I_global bursts.],
)




= Discussion



== Why this works


Two insights were critical:

*1. Per-parameter gradient clipping.* Standard clipping computes a
single norm across all parameters, so the largest matrix dominates the
direction. With conductance-based gradients, W\_ee's raw norm can be
10⁵× larger than W\_ie's. Per-parameter clipping ensures each weight
matrix gets an independent, bounded update step.

*2. Sufficient $kappa$.* With $kappa = 10$, the per-step gain
$65/10 = 6.5$ still compounds to $6.5^(200) approx 10^(163)$ over the
full simulation — far too large. With $kappa = 100$, the per-step gain
is $65/100 = 0.65 < 1$, which naturally decays over time. The gradient
can flow backward through many timesteps without exploding.


== Hybrid readout


The hybrid readout formula:

$
  "logits"_c = sum_t s_c(t) + alpha dot frac(1, T) sum_t V_c(t)
$


Spike counts provide the primary classification signal. The voltage
component ($alpha$ = #results.readout_alpha)
provides smooth gradient early in training when output neurons rarely
spike. As the network learns discriminative spiking, spike counts
dominate.


== What the PING loop contributes


Unlike detached-spike training where W\_ei and W\_ie received zero
gradient, here the inhibitory loop is actively shaped by learning.
The firing rate plot shows E\_hid rates decreasing from ~25 Hz to ~6 Hz
over training — the network learns to use inhibition to keep activity
sparse and discriminative, rather than saturating.


== Limitations


- The surrogate Jacobian distorts gradient direction. The true and
  surrogate Jacobians differ by a factor of $kappa$ on the
  conductance-voltage coupling. This is a biased estimator of the true
  gradient.
- Still using only 20k of 60k MNIST samples. More data + epochs would
  likely push accuracy higher — the test loss is still decreasing.
- $kappa$ was not extensively tuned. A per-matrix $kappa$ (different
  scaling for E and I populations) could improve results further.


== Next steps


- *Full MNIST*: train on all 60k samples for 20+ epochs.
- *$kappa$ sweep*: test 50, 100, 200 to find the optimum.
- *Per-matrix $kappa$*: the E and I populations have different
  membrane properties — they may benefit from different surrogate
  scaling.
- *Larger networks*: scale E\_hid beyond 256 neurons now that all
  weights can be trained.
- *e-prop comparison*: compare surrogate Jacobian against eligibility
  trace methods that avoid the need for temporal BPTT entirely.
