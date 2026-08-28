#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp049")

#let meta = (
  status: "[▦ DATA]",
  title: "Training Recurrent Weights Weakens PING Rhythmicity",
  date: "2026-06-09",
  updated_at: "2026-08-28",
  description: "Trainable recurrent conductances produced lower reference-image rhythmicity and higher excitatory firing than the frozen PING control; outcomes depended on initialization.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp049",)
#let preview-figures = (
  (path: "exp049/attractor_ei.svg", label: "Final population rates"),
  (path: "exp049/training_curves.svg", label: "Learning and rhythmicity"),
  (path: "exp049/weights__trainable_ping_init.svg", label: "Recurrent conductances"),
  (path: "exp049/phase_portrait.svg", label: "Rate and rhythmicity trajectories"),
  (path: "exp049/acc_rate_trajectory.svg", label: "Accuracy and firing trajectories"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r049 = data-json(data-file("exp049/numbers.json"))
#let mean(values) = values.sum() / values.len()
#let rounded(value, digits: 1) = calc.round(value, digits: digits)
#let condition(name) = r049.summary.filter(r => r.condition == name)
#let average(name, key) = rounded(mean(condition(name).map(r => r.at(key))))
#let trainable = r049.summary.filter(r => r.condition != "frozen_ping")
#let canonical42 = condition("trainable_ping_init").filter(r => r.seed == 42).first()
#let wei_zero42 = rounded(100 * canonical42.w_ei.trained_zero_fraction)
#let wie_zero42 = rounded(100 * canonical42.w_ie.trained_zero_fraction)
#let wei_mean42 = str(rounded(canonical42.w_ei.trained_mean, digits: 6))
#let wie_mean42 = str(rounded(canonical42.w_ie.trained_mean, digits: 6))
#let frozen_e = average("frozen_ping", "e_rate_hz")
#let frozen_i = average("frozen_ping", "i_rate_hz")
#let frozen_acc = average("frozen_ping", "acc")
#let acc_low = rounded(calc.min(..trainable.map(r => r.acc)))
#let acc_high = rounded(calc.max(..trainable.map(r => r.acc)))
#let contrast_low = rounded(r049.rhythmicity.final_contrast_trainable_min, digits: 3)
#let contrast_high = rounded(r049.rhythmicity.final_contrast_trainable_max, digits: 3)
#let contrast_frozen = rounded(r049.rhythmicity.canonical_contrast, digits: 3)
#let contrast_first = rounded(r049.rhythmicity.epoch1_contrast_trainable, digits: 3)
#let eval_n = r049.config.evaluation_samples

  == Abstract

  Training the recurrent E→I and I→E conductances weakened reference-image
  rhythmicity relative to a frozen PING control. I reanalysed retained results
  from four conditions and three seeds per condition after 50 epochs of MNIST
  training. Final contrast was #contrast_low–#contrast_high across trainable
  networks, versus a frozen-control mean of #contrast_frozen. Excitatory firing
  increased, while inhibitory firing depended on initialization. Trainable
  networks scored #acc_low–#acc_high% on #eval_n test images, compared with a
  #frozen_acc% control mean. These conditions did not recover the control's
  strongly rhythmic regime; they do not establish a universal failure of
  learning or an accuracy-equivalence result.

  #run-view("exp049", inputs)

  == Results

  === Recurrent training changes population activity

  #figure(
    data-image(data-file("exp049/attractor_ei.svg"), width: 100%,
      alt: "Final test-set E/I firing rates: the frozen control has low E and high I activity; trainable conditions have higher E rates and differing residual I activity."),
    caption: [
      Each point is one final-epoch network, evaluated on #eval_n official-test
      images; legend accuracies are condition means over three seeds. Frozen
      E/I means were #frozen_e/#frozen_i Hz. Canonical, zero and small trainable
      initializations gave E means of #average("trainable_ping_init", "e_rate_hz"),
      #average("trainable_zero_init", "e_rate_hz") and
      #average("trainable_small_init", "e_rate_hz") Hz; corresponding I means
      were #average("trainable_ping_init", "i_rate_hz"),
      #average("trainable_zero_init", "i_rate_hz") and
      #average("trainable_small_init", "i_rate_hz") Hz. Only zero initialization
      left I completely silent. Firing rates alone do not identify PING.
    ],
  )

  === Rhythmicity is already low at the first logged epoch

  #figure(
    data-image(data-file("exp049/training_curves.svg"), width: 100%,
      alt: "Validation accuracy and E/I rates over 50 epochs, alongside reference-image rhythmicity; frozen recurrence retains high contrast while trainable recurrence has low contrast."),
    caption: [
      Retained histories: validation accuracy and population rates, plus
      contrast from a fixed reference-image diagnostic. Lines show three-seed
      means; shading spans seed minima and maxima. Each series is smoothed
      with a five-epoch edge-padded moving average, not a confidence interval.
      The unsmoothed trainable contrast averaged #contrast_first after epoch 1;
      final values were #contrast_low–#contrast_high. This figure contains no
      epoch-0 observation and cannot resolve the intervening transition.
    ],
  )

  === The two recurrent matrices change differently

  #figure(
    data-image(data-file("exp049/weights__trainable_ping_init.svg"), width: 100%,
      alt: "Positive initial and final recurrent conductances for canonical trainable initialization, with separate E-to-I and I-to-E means and zero fractions."),
    caption: [
      Initial and final conductance distributions, pooled across three seeds
      with canonical trainable initialization. Histograms contain positive
      entries; mean and zero-fraction annotations include all entries. For
      seed 42, #wei_zero42% of E→I entries and #wie_zero42% of I→E entries were
      zero; final means were #wei_mean42 and #wie_mean42 in model conductance
      units. Lower E→I recruitment is consistent with reduced I activity, but
      sparsification is not symmetric and these observations do not isolate
      its causal contribution.
    ],
  )

  === Rate–rhythmicity trajectories do not establish attractors

  #figure(
    data-image(data-file("exp049/phase_portrait.svg"), width: 100%,
      alt: "Unsmoothed mean validation E rate versus reference-image contrast: trainable trajectories remain at low contrast; frozen endpoints remain near contrast one."),
    caption: [
      Trainable curves are unsmoothed three-seed means; fading indicates epoch
      order, open markers epoch 1 and filled markers epoch 50. Frozen points
      show the three final endpoints and their mean, not a full trajectory.
      Rates and contrast come from different evaluation samples. The contrast
      gap describes these observations; it identifies neither a separatrix
      nor a basin boundary, and is not evidence that only one attractor exists.
    ],
  )

  === Similar validation accuracy can accompany different firing rates

  #figure(
    data-image(data-file("exp049/acc_rate_trajectory.svg"), width: 100%,
      alt: "Validation accuracy versus E firing rate through training, with colour showing reference-image contrast and markers indicating the first and final epochs."),
    caption: [
      Unsmoothed three-seed mean validation trajectories; each segment's colour
      averages the reference-image contrast at its endpoints. Open and filled
      markers denote epochs 1 and 50. This complements the
      #link("/exp025/")[accuracy–rate comparison], without establishing equal
      accuracy or measured energy savings. Final official-test means were
      #average("trainable_ping_init", "acc")%,
      #average("trainable_zero_init", "acc")% and
      #average("trainable_small_init", "acc")% for canonical, zero and small
      trainable initializations, versus #frozen_acc% for the frozen control.
      Initialization therefore matters within the tested conditions.
    ],
  )

  == Methods

  I reused networks from the #link("/exp022/")[shared training study] and
  reanalysed retained observations. No new training or simulation was performed.

  + *Compare recurrent trainability.* Twelve conductance-based leaky-integrate-and-fire
    classifiers had 784 Poisson input channels, 1,024 excitatory (E), 256
    inhibitory (I) and 10 output cells. Three seeds per condition compared
    frozen canonical recurrence with trainable canonical, zero and 10%-canonical
    E→I/I→E conductances; E→E and I→I coupling stayed zero. Canonical initializer
    means were $1/1024$ and $2/256$, respectively, with standard deviations one
    tenth of each mean and negative draws clamped to zero.

  + *Train on a held-out split.* The 7,000-image subset contained 6,300 optimizer-training
    and 700 validation images from the official MNIST training partition.
    Input and readout weights trained for 50 epochs with AdamW, learning rate
    $4 times 10^(-4)$, zero weight decay, batch size 256, surrogate slope 1,
    voltage-gradient damping 1,000, gradient-norm clipping at 1 and no firing-rate
    penalty. Class scores were mean pre-reset output voltages; each 200 ms
    presentation used 0.1 ms steps and pixel-dependent input rates up to 25 Hz.

  + *Constrain conductance signs.* Trainable recurrent magnitudes were projected
    onto the non-negative cone after each optimizer step. Inhibition arose
    through $g_I (E_I - V)$, where $g_I$ is inhibitory conductance, $E_I = -80$
    mV its reversal potential, and $V$ membrane voltage: a positive I→E weight
    need not become negative to inhibit. Input zeros remained trainable and
    could regrow; initialization details are listed below.

  + *Evaluate final networks.* All endpoint tests and weight comparisons used
    epoch 50, not validation-selected weights. Accuracy and whole-population
    mean E/I rates used the same #eval_n official-test images per network;
    per-epoch validation metrics averaged three fixed Poisson encoding draws.
    For endpoint spectra, demeaned nonconstant E-population traces received
    full-trial Welch density estimation; the mean spectrum's largest bin
    within 5–150 Hz defined the retained peak, without interpolation.

  + *Measure temporal contrast.* After each epoch, the same fixed reference
    digit's Poisson spike realization elicited a diagnostic response.
    E-population counts were binned at 1 ms; their autocorrelation was normalized
    by lag overlap and squared mean count, over 0–100 ms, then smoothed with
    weights $(1/4, 1/2, 1/4)$ after replacing the zero-lag entry by its neighbour:
    #math.equation(block: true, numbering: "(1)", $R = (L - Q) / (L + Q)$)
    Here $R$ is dimensionless contrast, $Q$ the first local trough from lag
    2 ms onward, and $L$ the preceding positive-lag maximum of the smoothed
    autocorrelogram. I reused the retained scalar; it is neither a
    test-population rhythm estimate nor a calibrated probability of PING.

  == Appendix: retained parameters and interpretation limits

  Input weights used lower-clamped normal draws with parent mean 0.9 and
  standard deviation 0.09, followed by 95% initial zeros; retained values were
  divided by $0.05 times 784$ to preserve expected summed input coupling.
  Readout initialization
  used parent mean 1.12060546875 and standard deviation 0.8349609375.
  Excitatory and inhibitory synaptic decays were 2 and 6 ms. Membrane time
  constants were not trained; adaptive thresholds were disabled. The frozen
  control used the same canonical recurrence as the
  #link("/exp025/")[fixed-loop training comparison].

  Recurrent-weight summaries distinguished the two directions: zeros counted
  non-positive entries, positive means excluded zeros, and distribution plots
  pooled seeds before binning. Illustrative diagnostic cards used a separate
  test-image snapshot, seed 42 and image index 0, with 200 E and 50 I cells
  sampled for display. Their accuracy trajectories are validation measurements;
  their E/I rate trajectories are reference-image diagnostics, while header
  statistics and spectra describe final official-test evaluations.

  The PING interpretation concerns excitatory recruitment of inhibition and
  rhythmic feedback, not the sign of a stored conductance magnitude. Reduced
  contrast, higher E firing and weaker I activity support loss or weakening of
  the frozen control's regime in these conditions. Residual contrast and a
  spectral maximum do not establish a surviving gamma rhythm: the
  #link("/exp054/")[rhythmicity diagnostic study] addresses metric specificity.
  In particular, a low contrast value alone does not identify its source as
  low-rate inflation or shared input. The three initializations and three seeds
  do not establish impossibility of learning PING, accuracy equivalence,
  attractor stability or a continuous transition between epochs.
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How does training recurrent conductances change population activity and rhythmicity relative to a frozen PING control?],
    preview-figures, json-inputs: ("exp049",),
  )
}

#let meta = meta + (assets: input-assets("exp049", inputs))
#let body = with-datasets("exp049", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
