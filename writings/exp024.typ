#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp024")

#let meta = (
  title: "Accuracy Plateaus While Firing Rate Rises",
  date: "2026-06-02",
  description: "Audits validation accuracy and firing-rate stability in unregularised PING and COBA training histories.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp024",)
#let preview-figures = (
  (path: "exp024/coba_curves.svg", label: "coba curves"),
  (path: "exp024/ping_curves.svg", label: "ping curves"),
  (path: "exp024/confidence_inflation.svg", label: "confidence inflation"),
)

#let render-report(data-file) = [
#let r = data-json(data-file("exp024/numbers.json"))
#let c = r.config
#let n = c.seeds.len()
#let value(model, field, digits: 2) = calc.round(r.models.at(model).at(field).mean, digits: digits)
#let count(model, field) = r.models.at(model).at(field)
#let body = [
  == Abstract

  We audited whether firing rate settled once classification accuracy plateaued,
  using unregularised PING and COBA networks from the shared training study.
  The comparison reused #n seeds per architecture and #c.epochs epochs of MNIST
  training. Final mean validation accuracy was #value("coba", "final_acc")% for
  COBA and #value("ping", "final_acc")% for PING; excitatory rates were
  #value("coba", "final_e_rate_hz") and #value("ping", "final_e_rate_hz") Hz.
  Under the stated final-window criterion, #count("coba", "e_rate_converged_count")/#n
  COBA seeds and #count("ping", "e_rate_converged_count")/#n PING seeds met the
  rate-stability threshold. Accuracy and firing rate therefore require separate
  convergence checks; a low rate alone does not establish a fixed-rate attractor.

  == Inputs

  Uses the unregularised baseline learning histories from
  #link("exp022.html")[Training Runs]: COBA and PING, seeds #c.seeds.map(str).join([, ]).
  The complete trajectories support comparisons throughout learning, rather than
  only at a selected checkpoint.

  == Results

  === 1. COBA training trajectories

  #figure(
    data-image(data-file("exp024/coba_curves.svg"), width: 100%,
      alt: "COBA training and validation loss, validation accuracy, and excitatory firing rate over epochs."),
    caption: [COBA, #n seeds shown separately. Training loss is solid and validation
      loss dashed. Final mean validation accuracy is #value("coba", "final_acc")%
      and E rate #value("coba", "final_e_rate_hz") Hz. The mean final-window E-rate
      slope is #value("coba", "e_rate_slope_last10_hz_per_ep", digits: 3) Hz/epoch;
      #count("coba", "accuracy_converged_count")/#n seeds meet the accuracy criterion
      and #count("coba", "e_rate_converged_count")/#n meet the rate criterion.],
  )

  === 2. PING training trajectories

  #figure(
    data-image(data-file("exp024/ping_curves.svg"), width: 100%,
      alt: "PING training and validation loss, validation accuracy, and excitatory and inhibitory firing rates over epochs."),
    caption: [PING, #n seeds shown separately; E rate solid, I rate dashed.
      Final mean E and I rates are #value("ping", "final_e_rate_hz") and
      #value("ping", "final_i_rate_hz") Hz. The mean final-window E-rate slope is
      #value("ping", "e_rate_slope_last10_hz_per_ep", digits: 3) Hz/epoch;
      #count("ping", "accuracy_converged_count")/#n seeds meet the accuracy criterion
      and #count("ping", "e_rate_converged_count")/#n meet the rate criterion.],
  )

  === 3. Accuracy, cross-entropy and activity

  #figure(
    data-image(data-file("exp024/confidence_inflation.svg"), width: 100%,
      alt: "Validation accuracy, validation cross-entropy, and excitatory firing rate for COBA and PING."),
    caption: [Validation accuracy, cross-entropy on a log axis, and E rate;
      COBA red, PING black, #n seeds each. Dotted lines show each architecture's
      mean first epoch reaching 99% of its final accuracy. These markers do not
      establish sustained convergence. Curves show reused training observations,
      not a direct measurement of confidence or a causal rate–margin relation.],
  )

  == Methods

  We assessed finite changes in accuracy, activity and weights using retained
  learning histories from unregularised classifiers.

  + *Select the baseline histories.* We reused all #n seeds per architecture from
    the unregularised activity comparison. Each history contains #c.epochs
    consecutive completed epochs; final values refer to the last epoch, not the
    checkpoint selected by minimum validation loss. The audit performs no new
    training or inference.

  + *Identify the evaluation split.* The training pool contained #c.max_samples
    images from MNIST's official training partition, split into
    #c.dataset_split.optimizer_train_samples optimisation samples and
    #c.dataset_split.validation_samples validation samples. The official
    test partition of #c.dataset_split.official_test_samples images was not used
    during training. Per-epoch evaluation averaged #c.validation_encoder_draws.count
    fixed encoder draws per validation sample; those draws are not independent
    training seeds.

  + *Recover the training conditions.* Images drove #c.n_in Poisson input channels
    at a maximum pixel rate of #c.input_rate Hz for #c.t_ms ms, with a #c.dt ms
    timestep. The networks used #c.n_hidden excitatory and #c.n_inh inhibitory
    hidden neurons and #c.n_out output neurons. Mean output membrane voltage
    supplied the class logits. Training used surrogate gradients#cite(1), learning
    rate #c.lr and batches of #c.batch_size. Voltage-gradient damping was
    #c.voltage_gradient_damping.coba for COBA and #c.voltage_gradient_damping.ping
    for PING; no activity regulariser was applied.

  + *Measure final-window drift.* For each seed, we retained validation accuracy,
    training and validation cross-entropy, and population-mean E and I rates.
    The final #c.window_epochs epochs define the endpoint slope
    #math.equation(block: true,
      $s_x = (x_E - x_(E - w + 1)) / (w - 1) quad "(1)"$)
    Here $x_e$ is a measurement at epoch $e$, $E$ is the final epoch, $w$ is the
    window length, and $s_x$ is change per epoch. Absolute slopes below
    #r.measurement.accuracy_threshold_pp_per_epoch percentage points/epoch for
    accuracy or #r.measurement.rate_threshold_hz_per_epoch Hz/epoch for E rate
    meet the audit's operational stability criterion. This endpoint diagnostic
    does not exclude fluctuations within the window or prove asymptotic convergence.

    We retained per-seed slopes, first-to-final-epoch
    weight-norm ratios, and final-window weight-norm slopes, and computed means
    and sample standard deviations across seeds. Curves show individual seeds.
    The first epoch reaching 99% of final accuracy supplies a separate descriptive
    marker, averaged across seeds; it does not require subsequent accuracy to
    stay above the threshold.

  == Discussion

  Cross-entropy can keep rewarding larger decision gaps after the predicted class
  becomes correct:
  #math.equation(block: true,
    $"CE" = -log p_y = log(1 + sum_(k != y) e^(z_k - z_y)) quad "(2)"$)
  Here $"CE"$ is cross-entropy for one example, $y$ its true class,
  $k$ an alternative class, $z_y$ and $z_k$ their logits, and $p_y$ the
  softmax probability of the true class. The decision margin
  $m = z_y - max_(k != y) z_k$ determines correctness by its sign, whereas
  cross-entropy depends on all the logit gaps. Aggregate accuracy can remain
  steady while individual predictions change.

  The continued activity drift is consistent with ongoing optimisation, but
  these curves do not establish that confidence growth causes the rate increase.
  The mean-membrane readout depends on synaptic drive and membrane dynamics;
  it is not simply a linear function of the mean E rate. PING's lower activity
  and slower drift in the retained comparison do not demonstrate a fixed-rate
  attractor or isolate a causal benefit of gamma timing.

  #reference-list((
    (text: [E. O. Neftci, H. Mostafa, and F. Zenke.
      “Surrogate Gradient Learning in Spiking Neural Networks.”
      _IEEE Signal Processing Magazine_ 36(6), 51–63 (2019).],
      doi: "10.1109/MSP.2019.2931595"),
  ))
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs,
    [Does firing rate settle when classification accuracy plateaus? Compare the
      retained validation trajectories of COBA and PING.], preview-figures)
}
