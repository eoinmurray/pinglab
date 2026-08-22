#let meta = (
  title: "Variable-rate streaming with a spike-count readout",
  date: "2026-08-10",
  description: "Successor to exp048 using the variable-rate PING training bank and a readout matched to each presentation window.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp082/numbers.json")
#let provisional = (
  r.profile != "production"
  or r.config.digits_per_seed_cell < 200
  or r.config.psychometric_rates_hz.len() < r.config.training_rates_hz.len()
)

#let body = [
  == Abstract

  This experiment tests whether PING networks trained across input rates can classify continuous MNIST streams with the same spiking readout used during training. Excitatory spikes drive ten output LIF neurons, and each class logit is the corresponding neuron's total spike count within the current digit's presentation window. A variable-rate, variable-duration stream shows the online network response, while a 200 ms rate psychometric and a duration-by-rate map measure performance across the trained input distribution.

  == Methods

  === 1. Training source

  The experiment loads `ping__variable_rate__seed42`, `ping__variable_rate__seed43`, and `ping__variable_rate__seed44` from exp022. Each PING network is trained by sampling one maximum-pixel Poisson rate independently per image presentation from 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, and 25 Hz. The recurrent weights and classifier are frozen throughout this experiment.

  *Note.* The exp022 cells are registered and `tools/snn` implements the required output-LIF `spike-count` training, restore, and output-raster path. Presentation boundaries reset only the output LIF and its counter; the hidden PING state remains continuous. This entry still fails loudly if the variable-rate checkpoint bank is absent or uses another readout.

  Streaming inference uses each seed's checkpoint selected by validation accuracy, because this is a deployment evaluation of the selected classifier rather than an analysis of final-epoch drift.

  === 2. Spike-count readout

  At each timestep, the 1024-element excitatory spike vector $bold(s)^E(t)$ drives a trained 1024-by-10 projection $W_"out"$ into ten output LIF neurons. For a presentation beginning at timestep $a$ and ending before timestep $b$, the cumulative spike-count evidence for class $c$ at timestep $u$, where $a <= u < b$, is

  $ z_c(u) = sum_(t=a)^u s_c^"out"(t), quad c in {0, dots, 9}. quad "(1)" $

  Here $s_c^"out"(t)$ is the spike emitted by output LIF neuron $c$ at timestep $t$, and $bold(z)(u)$ contains the ten dimensionless cumulative output-spike counts. The final spike-count logit is $z_c(b - 1)$, and the predicted digit is $arg max_c z_c(b - 1)$. Dividing final counts by the common presentation duration gives firing rates for reporting and leaves this argmax unchanged, but rates are not passed to cross-entropy as logits. The output neurons therefore have membrane state, leak, threshold, spikes, and reset; unlike `mem-mean`, their membrane voltages are not the logits.

  === 3. Single-trial evidence visualization

  Before the continuous stream, one matched-condition trial is shown in isolation. The first correctly classified presentation in the pre-existing matched stream is selected for this explanatory figure: digit 4, presented for 200 ms at a maximum-pixel rate of 5 Hz. The selection is deliberately outcome-based so the figure can explain a successful readout trajectory; it is not used to estimate performance. The output-LIF state and spike counter reset at the presentation boundary, while the hidden PING state continues from the preceding presentation. The displayed class trace is

  $ p_c(u) = "softmax"_c(bold(z)(u)). quad "(2)" $

  Here $p_c(u)$ is the softmax visualization of cumulative integer output-spike counts for class $c$. It is not a calibrated posterior probability. The trial is used to explain how discrete output spikes produce the class-evidence traces; aggregate performance is reported separately.

  A second view enlarges 91.5–94.5 ms of the same stored trial. This interval is selected post hoc to explain a conspicuous transition in the full-trial trace, not to estimate performance. It aligns individual output-LIF spikes with the corresponding cumulative class counts and their softmax transformation.

  === 4. Variable-rate and variable-duration inference

  Every digit is read from only the spikes generated during its own presentation:

  $ T_"readout" = T_"presentation". quad "(3)" $

  Here $T_"readout"$ is the spike-summation window and $T_"presentation"$ is the duration of the corresponding digit. The five-digit headline stream changes both presentation duration and maximum-pixel Poisson rate at digit boundaries. A factorial evaluation then crosses presentation durations 25, 50, 100, and 200 ms with the eleven training rates. Every cell is evaluated on 200 classified digits independently for seeds 42, 43, and 44. Independent five-digit streams are evaluated in simulator batches; each batch member has separate neuronal state, while hidden state remains continuous only between the five digits belonging to that stream. Statistical grid cells retain compact per-presentation spike counts and population totals. Full rasters are retained for the illustrative stream and an internal matched-condition validation stream.

  === 5. Psychometric protocol

  The rate psychometric fixes presentation and readout duration at 200 ms, varies the maximum-pixel rate across the eleven training values, and reports held-out classification accuracy across the three trained seeds. This isolates rate robustness from shortened evidence windows.

  == Results

  #if provisional [
    #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
      *Provisional validation output.* These figures verify the complete inference, measurement, and rendering path. Each seed-level grid cell contains only #r.config.digits_per_seed_cell classified digits, and the checkpoints come from reduced training runs, so the numerical values are not estimates of publication performance.
    ]
  ]

  === 1. Single-trial spike-count evidence

  The isolated trial shows the complete path from one input digit through E- and I-population activity to the ten output-LIF class traces. Each step in a trace is caused by an output spike; confidence changes only when one of the cumulative class counts changes.

  #figure(
    image("/artifacts/data/exp082/single_trial.png", width: 100%, alt: "One MNIST digit above excitatory and inhibitory spike rasters and ten cumulative spike-count class-evidence traces."),
    caption: [A correctly classified 200 ms presentation at a maximum-pixel input rate of 5 Hz, selected as the first success in the pre-existing matched stream. The output readout resets at the boundary while hidden PING state remains continuous. The top badge gives the true label and final prediction. The middle panels show E- and I-cell spike rasters. The lower panel shows $p_c(u)$ from Equation 2; the true and winning class is red, and the annotation gives its final count, the runner-up count, and the winning margin. Equation 2 visualizes the cumulative integer counts $z_c(u)$ from Equation 1 and is not a calibrated posterior probability.],
  )

  The transition near 93 ms is not a continuous change in confidence. It is the softmax response to discrete output spikes, including spikes from the true and winning class.

  #figure(
    image("/artifacts/data/exp082/single_trial_transition.png", width: 100%, alt: "A zoom from 91.5 to 94.5 milliseconds showing output spikes, cumulative class counts, and class evidence."),
    caption: [The 91.5–94.5 ms transition from Figure 1. Top: output-LIF spikes by class. Middle: cumulative counts $z_c(u)$ from Equation 1. Bottom: $p_c(u)$ from Equation 2. The true and winning class 4 is red; other classes are grey. The apparent jump in class evidence is produced by discrete count increments rather than by a continuous-valued readout.],
  )

  Figure 2 resolves the transformation from output spikes to the plotted class evidence. Each mark in the top panel increments exactly one cumulative count $z_c(u)$ in the middle panel; between output spikes every count remains constant. Equation 2 exponentiates and normalizes all ten counts, so, if the other counts are fixed, one additional spike multiplies that class's unnormalized softmax weight by $e approx 2.72$. At 92.6 ms, an output spike from class 4 raises $p_4(u)$ from 0.47 to 0.71. At 93.9 ms, class 4 spikes alongside classes 0, 3, and 6, moving $p_4(u)$ from 0.53 to 0.73. Conversely, $p_4(u)$ can fall while $z_4(u)$ remains constant when competing classes spike and enlarge Equation 2's denominator.

  The horizontal 0.5 line therefore marks where class 4's exponential weight equals the combined weight of the other nine classes; it does not represent half of the output spikes. By the end of the presentation, class 4 has 31 spikes and runner-up class 9 has 28. Softmax maps this three-spike margin to $p_4 = 0.95$. The near-saturated trace is thus a normalized display of relative integer-count evidence, not a calibrated probability that the classification is correct.

  === 2. Streaming classification and temporal evidence

  The variable stream changes both input rate and presentation duration at digit boundaries. Each decision still uses exactly the spikes emitted during that digit, so evidence never leaks across labels.

  #figure(
    image("/artifacts/data/exp082/variable_stream.png", width: 100%, alt: "Excitatory and inhibitory rasters with online spike-count class evidence as input rate and digit duration vary."),
    caption: [Successor to exp048 Figure 1, using the variable-rate-trained PING network and its native output-LIF spike-count readout. Presentation duration and maximum-pixel input rate vary between segments. Thumbnail opacity increases with encoding rate; badges show true label to prediction, with errors in red. The middle panels plot E- and I-cell spike rasters against time (ms). The lower panel plots $p_c(u)$ from Equation 2, with the true-class trace emphasized in red. The cumulative counts $z_c(u)$ from Equation 1 reset at each boundary while hidden PING state remains continuous.],
  )

  The factorial summary separates the two manipulations. The left panel crosses presentation duration with maximum-pixel input rate. The right panel holds presentation and readout at 200 ms, giving the rate psychometric without a simultaneous change in integration time.

  #figure(
    image("/artifacts/data/exp082/duration_rate_summary.png", width: 100%, alt: "Accuracy map over presentation duration and input rate beside the 200 millisecond input-rate psychometric curve."),
    caption: [Accuracy across matched presentation/readout durations and input rates, with the 200 ms rate psychometric shown separately.],
  )

  #figure(
    image("/artifacts/data/exp082/psychometric_200ms.svg", width: 85%, alt: "Classification accuracy versus maximum-pixel input rate at 200 milliseconds."),
    caption: [Input-rate psychometric at a fixed 200 ms presentation and spike-count window.],
  )

  === 3. Comparison with exp048

  Exp048 used a fixed-25-Hz training distribution and reconstructed a sliding `mem-mean` output. This experiment instead trains across the evaluated rates and uses the trained output-LIF spike counts directly. A numerical difference between the experiments therefore combines a change in training distribution with a change in readout; it is not an isolated comparison of either intervention.
]
