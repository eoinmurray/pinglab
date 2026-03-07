// title: study.16-e-prop
// date: 2026-03-05T10:07:20Z
// description: E-prop with firing rate regularization reaches 88.7% on MNIST with biologically plausible ~10 Hz firing rates — 7 points below BPTT but no gradient explosion.

#let config = json("_artifacts/study.16-e-prop/config.json")
#let results = json("_artifacts/study.16-e-prop/results.json")






= Summary


Study.15 showed that BPTT with a surrogate Jacobian ($kappa = 100$)
reaches 95.6% on MNIST, but requires careful hyperparameter tuning
(per-parameter clipping, large $kappa$) to prevent gradient explosion
from the conductance-based driving force $(E_e - V) approx 65"mV"$.

This study replaces BPTT entirely with *e-prop* (eligibility
propagation), a biologically plausible learning rule that maintains
forward-running eligibility traces. The $(E_e - V)$ term appears
additively in each trace update rather than multiplicatively through
time, sidestepping the gradient explosion problem by construction.

*Result:* #results.best_test_accuracy% test
accuracy on #results.test_samples
samples with firing rates of ~10 Hz — biologically plausible and
energy-efficient. A firing rate regularizer ($lambda = 0.1$,
$r_"target" = 10$ Hz) trades 1.3 points of accuracy (vs 90.0%
without) for dramatically lower rates (down from 65–220 Hz).


= Architecture


#figure(
  image("_artifacts/study.16-e-prop/graph_dark.png"),
  caption: [Network topology — identical to study.15.],
)



Same PING network as studies 13–15:

#table(
  columns: 4,
  [Layer], [Population], [Size], [Role],
  [Input], [`E_in`], [784], [One neuron per pixel, driven by Poisson spike trains],
  [Hidden], [`E_hid`], [#config.nodes.at(2).size], [Feature extraction — trained via e-prop traces],
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
  [`meta.readout`], [#config.meta.readout],
  [`meta.readout_alpha`], [#config.meta.readout_alpha],
  [`meta.training_method`], [#config.meta.training_method],
  [`meta.max_grad_norm`], [#config.meta.max_grad_norm],
  [`meta.rate_reg`], [#config.meta.rate_reg],
  [`meta.rate_target_hz`], [#config.meta.rate_target_hz],
  [`meta.subset_size`], [#config.meta.subset_size],
  [`encoding`], [#results.encoding],
)



= Method



== E-prop overview


E-prop (Bellec et al. 2020) replaces backpropagation through time with
three forward-running quantities per synapse $W_("ij")$:

*1. Eligibility vector* $epsilon_("ij")(t)$ — tracks how postsynaptic
voltage $V_j$ depends on $W_("ij")$:

$
  epsilon_("ij")(t+1) = alpha dot epsilon_("ij")(t) + frac(d t, C_m) (E_"syn" - V_j(t)) dot s_i(t-d)
$


where $alpha = 1 - d t dot g_L / C_m$ is the membrane leak factor and
$s_i(t-d)$ is the delayed presynaptic spike.

*2. Eligibility trace* $e_("ij")(t) = psi_j(t) dot epsilon_("ij")(t)$,
gated by the surrogate derivative:

$
  psi_j(t) = frac(1, (1 + |V_j(t) - V_"th"|)^2)
$


*3. Learning signal* $L_j$ — computed once at trial end from the
cross-entropy gradient:

$
  L_k^"out" = "softmax"("logits")_k - bold(1)_(k="label")
$


Hidden neurons receive the signal via symmetric feedback (transpose of
the feedforward weights):

$
  L_j^"hid" = sum_k W_("jk")^("hid" \to "out") dot L_k^"out"
$


The final weight gradient is:

$
  nabla W_("ij") = frac(1, B) sum_b L_j^((b)) sum_t e_("ij")^((b))(t)
$


No temporal backpropagation is needed. The $(E_e - V)$ driving force
enters additively each timestep rather than compounding
multiplicatively.


== What is trained


All weight matrices receive gradient via e-prop traces. No surrogate
Jacobian or spike detachment is needed.

#table(
  columns: 4,
  [Matrix], [Params], [Gradient source], [Connection],
  [`W_ee`], [#results.trainable_params_breakdown.W_ee], [direct e-prop traces], [E\_in→E\_hid and E\_hid→E\_out],
  [`W_ei`], [#results.trainable_params_breakdown.W_ei], [symmetric feedback via I], [E\_hid→I\_global],
  [`W_ie`], [#results.trainable_params_breakdown.W_ie], [symmetric feedback], [I\_global→E\_hid],
)



== Encoding


Poisson encoding (same as studies 14–15):

$
  I_n(t) = "scale" times "Bernoulli"("pixel"_n)
$



== Memory footprint


The main cost is the eligibility vectors. Per sample:

#table(
  columns: 3,
  [Trace], [Shape], [Size],
  [$epsilon_"in→hid"$], [[256, 784]], [800 KB],
  [$epsilon_"hid→out"$], [[10, 256]], [10 KB],
  [$epsilon_"E→I"$], [[64, 256]], [64 KB],
  [$epsilon_"I→E"$], [[256, 64]], [64 KB],
)


With $B = 64$: ~56 MB total. Feasible on a T4.


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
  [Training method], [#results.training_method],
  [Grad clip], [#config.meta.max_grad_norm (total norm)],
  [Training time], [n/a min],
  [Device], [#results.device (#results.runtime)],
)



== Comparison across studies


#table(
  columns: 5,
  [Study], [Method], [Trained weights], [Best Accuracy], [E\_hid rate],
  [study.8 (standard SNN)], [BPTT, detached], [W\_ee (2,560)], [64.2%], [—],
  [study.14 (Poisson)], [BPTT, detached], [W\_ee (2,560)], [84.8%], [—],
  [study.15 (surrogate Jacobian)], [BPTT, $kappa = 100$], [all (1.2M)], [*95.6%*], [~6 Hz],
  [study.16 (e-prop, no reg)], [e-prop], [all (1.2M)], [90.0%], [~65 Hz],
  [*study.16 (e-prop + rate reg)*], [*e-prop*], [*all (1.2M)*], [*#results.best_test_accuracy%*], [*~10 Hz*],
)


Without rate regularization, e-prop reaches 90.0% but with high firing
rates (E\_hid ~65 Hz, E\_out ~220 Hz). Adding a rate penalty
($lambda = 0.1$, target 10 Hz) brings rates down to ~10 Hz — matching
biologically plausible levels — at a cost of 1.3 accuracy points.


== Accuracy over epochs


#table(
  columns: 2,
  [Epoch], [Test Accuracy],
  [1], [28.6%],
  [2], [84.6%],
  [3], [86.6%],
  [4], [86.3%],
  [5], [87.2%],
  [6], [88.0%],
  [7], [88.3%],
  [8], [*88.7%*],
  [9], [88.3%],
  [10], [88.5%],
)


The large jump from epoch 1 (29%) to epoch 2 (85%) corresponds to the
network transitioning from diffuse spiking to class-selective output.
The rate regularizer slows early learning slightly (epoch 1 is lower
than the unregularized run) but the network converges to a similar
accuracy plateau by epoch 8.


== Training dynamics


#figure(
  image("_artifacts/study.16-e-prop/loss_train_dark.png"),
  caption: [Per-iteration training loss. Noisy early on (epochs 1–2), then settles to 0.2–0.8 range. The high variance is characteristic of e-prop — each batch gets a single trial-end learning signal rather than per-timestep gradient.],
)



#figure(
  image("_artifacts/study.16-e-prop/loss_test_dark.png"),
  caption: [Test loss per epoch. Decreasing from 2.30 to 0.57 — still improving at epoch 10. The rate regularization term contributes to the higher absolute loss values.],
)



#figure(
  grid(
    columns: 2,
    gutter: 4pt,
    image("_artifacts/study.16-e-prop/accuracy_dark.png"),
    image("_artifacts/study.16-e-prop/accuracy_light.png"),
  ),
  caption: [Per-iteration training accuracy. Rises from ~10% to ~85% over the first 3 epochs, then oscillates in the 80–95% range.],
)




== Gradient norms


#figure(
  image("_artifacts/study.16-e-prop/grad_norms_dark.png"),
  caption: [L2 gradient norm per weight matrix over training. W_ee dominates (500–2000), W_ie receives moderate gradient (50–100), and W_ei receives small but non-zero gradient (5–15) — confirming the buffer bug fix restored gradient flow to the E→I pathway.],
)



W\_ee receives strong gradient throughout training. W\_ie (I→E\_hid) and
W\_ei (E\_hid→I) receive moderate gradient via the symmetric feedback
path, allowing the PING inhibitory loop to be shaped by learning.

An earlier version of this implementation had a bug where `buffer_e_to_i`
was read after `apply_delayed_events()` cleared it, resulting in zero
W\_ei gradient. Fixing this — by cloning the buffer before the integrate
step — restored proper gradient flow and improved accuracy from 86.6%
to 90.0%.


== Firing rates


#figure(
  image("_artifacts/study.16-e-prop/firing_rates_dark.png"),
  caption: [Mean firing rate (Hz) per excitatory layer over training. E_hid and E_out both converge to ~10 Hz — right at the regularization target.],
)



The firing rate regularizer adds a homeostatic learning signal
$L_j^"rate" = 2lambda(r_j - r_"target")$ to each neuron's
e-prop update. This penalizes deviation from the 10 Hz target without
leaving the e-prop framework.

E\_hid starts at ~30 Hz, spikes briefly during early learning, then
settles to ~8–10 Hz by epoch 3. E\_out follows a similar trajectory.
Compare this to the unregularized run (E\_hid ~65 Hz, E\_out ~220 Hz)
and study.15's BPTT-trained rates (~6 Hz). The regularizer achieves
biologically plausible rates that the symmetric feedback signal alone
could not.


== Per-class accuracy


#figure(
  image("_artifacts/study.16-e-prop/accuracy_per_class_dark.png"),
  caption: [Per-class test accuracy. Digits 0 and 1 lead at ~96%, while 8 is the hardest at ~81%.],
)



#table(
  columns: 2,
  [Digit], [Accuracy],
  [0], [96.0%],
  [1], [96.5%],
  [2], [87.5%],
  [3], [88.0%],
  [4], [90.0%],
  [5], [84.2%],
  [6], [91.1%],
  [7], [86.1%],
  [8], [81.0%],
  [9], [85.7%],
)


The per-class spread (81–97%) is wider than study.15 (93–98%). Easy
digits (0, 1) are well above 95%; harder digits (5, 8) show the
limitation of the trial-end learning signal combined with the rate
constraint forcing sparser representations.


== Confusion matrix


#figure(
  image("_artifacts/study.16-e-prop/confusion_dark.png"),
  caption: [Confusion matrix on 5,000 test samples. Strong diagonal with residual confusion on visually similar digit pairs.],
)



The remaining confusion involves visually similar digits — the same
pairs that are typically hardest for spike-based classifiers.


== Output layer rasters


#figure(
  image("_artifacts/study.16-e-prop/raster_output_all_all_dark.png"),
  caption: [Output layer (10 neurons) spike rasters for one canonical sample per digit. Multiple neurons fire densely — less selective than study.15's sparse, class-specific spiking.],
)



The output rasters show a key difference from study.15: multiple output
neurons fire for each input, rather than a single class-selective
neuron. The network relies on differential spike counts (the correct
class fires more) rather than study.15's winner-take-all pattern. This
is characteristic of the trial-end learning signal — e-prop shapes
total spike counts but not temporal selectivity.


== Output neuron voltage traces


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.16-e-prop/voltage_output_digit_00_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_01_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_02_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_03_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_04_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_05_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_06_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_07_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_08_dark.png"),
    image("_artifacts/study.16-e-prop/voltage_output_digit_09_dark.png"),
  ),
  caption: [Membrane voltage traces for the 10 output neurons over 200 ms.],
)




== Input rasters (Poisson spike trains)


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.16-e-prop/raster_input_digit_00_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_01_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_02_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_03_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_04_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_05_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_06_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_07_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_08_dark.png"),
    image("_artifacts/study.16-e-prop/raster_input_digit_09_dark.png"),
  ),
  caption: [Poisson input spike trains for each digit (0–9).],
)




== Full layer rasters


#figure(
  grid(
    columns: 4,
    gutter: 4pt,
    image("_artifacts/study.16-e-prop/raster_layers_digit_00_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_01_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_02_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_03_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_04_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_05_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_06_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_07_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_08_dark.png"),
    image("_artifacts/study.16-e-prop/raster_layers_digit_09_dark.png"),
  ),
  caption: [Full signal path rasters for each digit.],
)




= Discussion



== E-prop vs BPTT: the accuracy gap


E-prop with rate regularization reaches 88.7% vs BPTT's 95.6% — a
~7-point gap. Three factors contribute:

*1. Trial-end learning signal.* E-prop computes the learning signal
once at trial end, while BPTT provides per-timestep gradient. This is
like the difference between reinforcement learning (sparse reward) and
supervised learning (dense supervision). The batch-level noise in the
training loss plot reflects this.

*2. Symmetric feedback approximation.* Hidden neuron learning signals
use the transpose of the feedforward weights ($W^\top$), not the true
error gradient. This is the "weight transport problem" — a known
limitation of biologically plausible learning rules.

*3. Rate regularization.* The firing rate penalty ($lambda = 0.1$)
competes with the classification objective. Without it, e-prop reaches
90.0% but with biologically unrealistic rates (65–220 Hz). The 1.3-point
cost buys dramatically more plausible dynamics (~10 Hz).


== What works well


Despite these limitations, e-prop has important advantages:

- *No hyperparameter sensitivity.* No $kappa$, no per-parameter
  clipping, no careful tuning of the surrogate Jacobian. The same
  learning rate (1e-4) and gradient clip (1.0) work out of the box.
- *Forward-only computation.* The eligibility traces run in lockstep
  with the simulation. No computational graph needs to be stored, and
  memory scales as $O(|"synapses"|)$ rather than
  $O(|"synapses"| times T)$.
- *Biological plausibility.* Each synapse only needs local
  information (presynaptic spike, postsynaptic voltage) plus a global
  learning signal. This is compatible with three-factor Hebbian
  learning in biological neural circuits.
- *Composable with rate control.* The firing rate regularizer slots
  into the learning signal without modifying the e-prop machinery,
  achieving ~10 Hz rates at minimal accuracy cost.


== Firing rate regularization


Without rate control, e-prop produces high firing rates because the
classification objective alone has no incentive to minimize spiking.
We add a homeostatic penalty directly to the e-prop learning signal:

$
  L_j^"total" = L_j^"class" + 2lambda(r_j - r_"target")
$


where $r_j$ is neuron $j$'s mean firing rate over the trial and
$r_"target" = 10$ Hz. This stays within canonical e-prop — the
rate error acts as an additional learning signal that modulates the
same eligibility traces. No autograd is needed.

#table(
  columns: 4,
  [Setting], [Best Acc], [E\_hid rate], [E\_out rate],
  [No reg ($lambda = 0$)], [90.0%], [~65 Hz], [~220 Hz],
  [$lambda = 0.1$], [88.7%], [~10 Hz], [~10 Hz],
)


The trade-off is modest: 1.3 accuracy points for a 6–22x reduction
in firing rates.


== The W\_ei buffer bug


An initial run produced 86.6% accuracy with W\_ei gradient identically
zero. Investigation revealed a bug: the E→I delayed spike buffer
(`buffer_e_to_i`) was read *after* `apply_delayed_events()` cleared
it inside the integrate step. The presynaptic signal for the E→I
eligibility trace was always zero.

The fix was simple — use the already-cloned `delayed_E` (from
`buffer_e_to_e`, cloned before the integrate step) instead. Both
buffers carry the same spike data when delays are equal.

This restored proper gradient flow to W\_ei, improved accuracy from
86.6% → 90.0%, and confirmed that canonical e-prop can train all three
weight matrices in the PING network without modification.


== Limitations


- Only 20k of 60k MNIST samples used. More data and epochs would
  likely push accuracy higher — test loss is still decreasing.
- The symmetric feedback assumption ($W^\top approx B$) is a known
  limitation of biologically plausible learning.
- Trial-end learning signal prevents learning temporal structure
  within the spike train.


== Next steps


- *Longer training.* Run on full 60k MNIST for 20+ epochs to see
  where e-prop saturates.
- *Random feedback alignment.* Replace $W^\top$ with a fixed random
  matrix $B$ to eliminate weight transport.
- *Tune rate\_reg.* Try $lambda = 0.01$–$0.05$ to find a better
  accuracy/rate trade-off.
- *Hybrid e-prop + BPTT.* Use e-prop for the long E\_in→E\_hid path
  (where BPTT explodes) and surrogate Jacobian BPTT for the short
  inhibitory loop (where BPTT is stable).
