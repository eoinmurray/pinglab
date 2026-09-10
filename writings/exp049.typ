#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-figure-ref, result-card, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp049")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Training Recurrent Weights Weakens PING Rhythmicity",
  created_at: "2026-06-09T00:00:00Z",
  updated_at: "2026-09-10",
  description: "Trainable recurrent conductances produced lower reference-image rhythmicity and higher excitatory firing than the frozen PING control; outcomes depended on initialization.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp049",)
#let preview-figures = (
  (path: "exp049/training_summary.svg", label: "Endpoints and recurrent weights"),
  (path: "exp049/training_curves.svg", label: "Learning and rhythmicity"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r049 = data-json(data-file("exp049/numbers.json"))
#let rounded(value, digits: 1) = calc.round(value, digits: digits)
#let contrast_low = rounded(r049.rhythmicity.final_contrast_trainable_min, digits: 3)
#let contrast_high = rounded(r049.rhythmicity.final_contrast_trainable_max, digits: 3)
#let contrast_first = rounded(r049.rhythmicity.epoch1_contrast_trainable, digits: 3)
#let eval_n = r049.config.evaluation_samples

  #journal-abstract(body: [
  We asked whether surrogate-gradient training preserves PING rhythmicity when
  recurrent excitatory–inhibitory conductances can change. We compared trainable
  recurrent loops across initializations with a control whose recurrent PING
  weights remained frozen.

  Recurrent training weakened rhythmicity early and did not recover the rhythmic
  control, while activity and accuracy changed by condition. This constrains these
  learning recipes, but does not show that recurrent learning must always
  destroy gamma or that the compared conditions are accuracy-equivalent.
  ])

  == Results

  #with-result-sections[

  #result-card[
  === E→I pruning accompanies low contrast

  In all three networks in each condition, training from standard and
  10%-standard recurrence drove most E→I weights to zero, while most I→E
  weights remained positive and their mean across all entries grew.
  Both conditions ended with lower reference-image contrast and higher E rates
  than frozen recurrence. Zero-initialized recurrence remained zero
  (#result-figure-ref(<fig:exp049-result-1>)). This pattern across three training
  replicates per condition remains descriptive: it does not isolate pruning
  as the cause of reduced contrast or establish equivalent accuracy.

  #figure(
    data-image(data-file("exp049/training_summary.svg"), width: 100%,
      alt: "Final accuracy, E/I rates and reference-image contrast across four conditions, with initial and final recurrent nonzero fractions and mean weights."),
    caption: [
      Final checkpoints: *(A)* official-test accuracy, *(B)* per-neuron E/I
      firing rates over the same #eval_n test images and *(C)* unsmoothed
      reference-image contrast $R = R_"contrast"$. Bars and error bars in A–C
      show means ±1 standard error of the mean (SEM) across three independently
      trained networks (sample standard deviation divided by $sqrt(3)$).
      In B, black denotes E and red I.
      Frozen denotes fixed standard recurrence; Std., 10% and Zero denote
      trainable recurrence initialized at standard, 10%-standard and zero weights.
      *(D, E)* Positive-weight fractions for E→I and I→E; *(F, G)* corresponding
      per-edge means including zeros, in model conductance units scaled by
      $10^(-3)$. For each direction, condition and time point, weight statistics
      pool 786,432 entries from three equal-sized matrices, giving each network
      equal weight. Wide grey bars show initialization and narrow red bars show
      epoch 50. Error bars show ±1 SEM across the three per-network statistics,
      calculated separately before and after training; they do not describe
      uncertainty in the paired change. Individual weights are not independent
      training replicates. Black arrows connect pooled values before and after
      when the relative change is at least 5%; this is a display threshold,
      not a statistical-significance test.
    ],
  ) <fig:exp049-result-1>

  ]

  #result-card[
  === Rhythmicity starts low

  The unsmoothed trainable contrast averaged #contrast_first after epoch 1 and
  ended at #contrast_low–#contrast_high. Because there is no epoch-0 observation,
  these histories cannot resolve the intervening transition (#result-figure-ref(<fig:exp049-result-2>)).

  #figure(
    data-image(data-file("exp049/training_curves.svg"), width: 100%,
      alt: "Validation accuracy and E/I rates over 50 epochs, alongside reference-image rhythmicity; frozen recurrence retains high contrast while trainable recurrence has low contrast."),
    caption: [
      Recorded histories: *(A)* validation accuracy, *(B)* E rate, *(C)* I rate
      and *(D)* contrast from a fixed reference-image diagnostic. Lines show three-seed
      means; shading spans seed minima and maxima. Each series is smoothed
      with a five-epoch edge-padded moving average, not a confidence interval.
    ],
  ) <fig:exp049-result-2>

  ]

  ]

  #journal-methods(
    orientation: [
  We reused networks from the #link("/exp022/")[exp022] — #link("/exp022/")[_Training Runs_] and
  reanalysed recorded observations. No new training or simulation was performed.
    ],
    compute: [
  + *Compare recurrent trainability.* Twelve conductance-based leaky-integrate-and-fire
    classifiers had 784 Poisson input channels, 1,024 excitatory (E), 256
    inhibitory (I) and 10 output neurons. With three training replicates per
    condition, we compared frozen canonical recurrence with trainable canonical,
    zero and 10%-canonical
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
    through $g_I (E_I - V_m)$, where $g_I$ is inhibitory conductance, $E_I = -80$
    mV its reversal potential, and $V_m$ membrane voltage: a positive I→E weight
    need not become negative to inhibit. Input zeros remained trainable and
    could regrow.
    ],
    analyse: [
  #set enum(start: 4)

  + *Evaluate final networks.* All endpoint tests and weight comparisons used
    epoch 50, not validation-selected weights. Accuracy and whole-population
    mean E/I rates used the same #eval_n official-test images per network;
    per-epoch validation metrics averaged three fixed Poisson encoding draws.
    For endpoint spectra, demeaned nonconstant E-population traces received
    full-trial Welch density estimation; the mean spectrum's largest bin
    within 5–150 Hz defined the selected peak, without interpolation.

  + *Measure temporal contrast.* After each epoch, the same fixed reference
    digit's Poisson spike realization elicited a diagnostic response.
    E-population counts were binned at 1 ms; their autocorrelation was normalized
    by lag overlap and squared mean count, over 0–100 ms, then smoothed with
    weights $(1/4, 1/2, 1/4)$ after replacing the zero-lag entry by its neighbour:
    #math.equation(block: true, $R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$)
    Here $R_"contrast"$ is dimensionless contrast, $A_"trough"$ the first local trough from lag
    2 ms onward, and $A_"lobe"$ the preceding positive-lag maximum of the smoothed
    autocorrelogram. We reused the recorded scalar; it is neither a
    test-population rhythm estimate nor a calibrated probability of PING.

  + *Summarize recurrent weights.* We reanalysed the initial and epoch-50
    E→I and I→E matrices. For each direction and condition, we calculated the
    strictly positive fraction and arithmetic mean including zeros, both per
    network and over the three pooled matrices. Equal matrix sizes gave each
    network equal weight. Separately at each time point, we calculated SEM as
    the sample standard deviation of the three network summaries divided by
    $sqrt(3)$. The independently trained network was the unit of replication;
    jointly trained weights were not additional replicates. These descriptive
    summaries did not test the significance of training-induced changes.
    ],
    present: [
  #set enum(start: 7)

  + *Expose retained training evidence.* We displayed retained validation,
    activity and temporal-contrast measurements with their
    distinct training-replicate and illustrative-probe roles.
    ],
  )
  #run-view("exp049", inputs)

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
#let body = journal-article("exp049", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
