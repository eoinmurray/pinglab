#import "contents.typ": with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp080")

#let meta = (
  status: "[▦ DATA | v31.2.0]",
  title: "Decoder Accuracy Improves with Input Rate",
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "Direct-simulation decoder calibration of the input-rate range for later variable-rate PING training.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp080",)
#let preview-figures = (
  (path: "exp080/training_history.svg", label: "training history"),
  (path: "exp080/feature_images.png", label: "feature images"),
  (path: "exp080/psychometric.svg", label: "psychometric"),
)

#let result-card-style = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-result-card { margin: 1.25rem 0; padding: 1.2rem 1.35rem 1.3rem; border: 1px solid var(--rule-strong); border-radius: 3px; background: var(--paper); } "
      + ".pinglab-result-card > h4:first-child { margin-top: 0; } "
      + ".pinglab-result-card > :last-child { margin-bottom: 0; } "
      + ".pinglab-result-card-notes { margin-top: 1rem; padding-top: .75rem; border-top: 1px solid var(--rule); font-size: var(--fs-small); line-height: 1.5; color: var(--muted); } "
      + ".pinglab-result-card-notes > p:first-child { margin: 0 0 .25rem; } "
      + ".pinglab-result-card-notes ul { margin: 0; padding-left: 1.2rem; } "
      + ".pinglab-result-card-notes li { margin: .2rem 0; } "
      + "@media (max-width: 520px) { .pinglab-result-card { margin: 1rem 0; padding: .95rem 1rem 1.05rem; } }",
    )
  }
}

#let result-card(body, notes: none) = context {
  let notes-body = if notes == none { none } else if target() == "html" {
    html.elem("aside", attrs: (class: "pinglab-result-card-notes", "aria-label": "Notes"), [
      *Notes.*
      #notes
    ])
  } else { [
    *Notes.*
    #notes
  ] }
  let card-body = [#body #notes-body]
  if target() == "html" {
    html.elem("article", attrs: (class: "pinglab-result-card"), card-body)
  } else { card-body }
}

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp080/numbers.json"))
#let p = r.parameters
#let d = r.decision
#let criterion-crossed = d.criterion_crossed
#let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
#let body = [
  == Abstract


  This experiment asked which input rates preserve enough information in
  synaptically and membrane-filtered MNIST images for classification. In the
  retained calibration, nonlinear decoders were trained across the tested rates
  and evaluated on held-out digits.

  Decoder accuracy improved with input rate, yielding a practical tested range
  for later variable-rate PING experiments. The result calibrates the filtering
  and decoding pipeline; it does not measure PING-network accuracy or predict
  performance between tested rates.

  == Results

  #with-result-sections[

  #result-card-style

  #result-card(notes: [
    - The first occurrence of the maximum validation accuracy selected the
      checkpoint for each training replicate; later ties did not replace it.
  ])[
  === Mixed-rate decoder validation accuracy across training

  The retained calibration contains three independently initialized nonlinear
  decoders trained on fresh encoding draws sampled across the complete
  input-rate grid.

  If mixed-rate training learned digit structure that survived stochastic
  filtering, validation accuracy should improve across all three training
  replicates rather than only one trajectory.

  Each replicate reached its maximum late in training. The selected epochs were
  #r.training.map(record => str(record.selected_epoch)).join(", "), with
  validation accuracies
  #r.training.map(record => pct(record.selected_validation_accuracy)).join(", ").

  #figure(
    data-image(data-file("exp080/training_history.svg"), width: 85%,
      alt: "Mixed-rate validation accuracy over training epochs, with one curve per decoder."),
    caption: [Each curve shows validation accuracy by epoch for one of
      #p.seeds.len() training replicates. Accuracy is the fraction correct across
      #p.validation_count validation presentations, with input rate sampled
      uniformly and a fresh encoding draw used for every presentation and epoch.],
  )
  ]

  #result-card[
  === Filtered digit features at 0.5, 5 and 25 Hz

  A reused illustration shows one MNIST digit after the same finite-window
  synaptic and membrane filtering at three input rates.

  Because sparse Bernoulli input produces a small, random number of events,
  reducing the rate should remove responses from different pixels rather than
  attenuate the whole digit uniformly.

  At 0.5 Hz only isolated fragments remained. The digit became progressively
  more complete at 5 and 25 Hz, consistent with stochastic event loss rather
  than uniform contrast scaling.

  #figure(
    data-image(data-file("exp080/feature_images.png"), width: 100%,
      alt: "One MNIST input digit and filtered feature images at 0.5, 5 and 25 Hz."),
    caption: [Reused illustrative input digit and its originally simulated
      features at #p.rates_hz.at(2), #p.rates_hz.at(5) and
      #p.rates_hz.last() Hz. Simulations used #p.probe_uS μS conductance
      increments, #p.presentation_ms ms presentations and independent encoding
      draws; feature panels share a 0–65 mV scale.
      #if r.illustration.kind == "historical-image" [The illustration was carried forward unchanged, not
      regenerated.]],
  )
  ]

  #result-card[
  === Held-out decoder accuracy across tested input rates

  The three selected decoders received the same held-out feature vector and
  encoding draw for each image and rate. The practical floor required every
  decoder to reach
  #pct(p.useful_accuracy) accuracy at a tested rate.

  If increasing input rate preserved more digit structure through the filter,
  held-out accuracy should rise from the sparse-drive conditions before
  approaching a high-rate plateau.

  Mean accuracy across training replicates increased monotonically from
  #pct(d.rows.first().accuracy) at #d.rows.first().rate_hz Hz to
  #pct(d.rows.last().accuracy) at #d.rows.last().rate_hz Hz.
  #if criterion-crossed [The selected floor was #d.r_train_hz Hz: all three
  decoders first crossed the criterion there, and mean accuracy was
  #pct(d.rows.filter(row => row.rate_hz == d.r_train_hz).first().accuracy).] else [No
  rate met the criterion for every decoder, so the floor was
  right-censored at #d.recommendation.ceiling_hz Hz.] The resulting interval is
  a decoder-relative calibration without interpolation; it does not establish
  PING-network performance.

  #figure(
    data-image(data-file("exp080/psychometric.svg"), width: 85%,
      alt: "Held-out accuracy across tested input rates, showing the decoder mean and minimum-to-maximum range."),
    caption: [Points show mean accuracy across #p.seeds.len() training
      replicates at each maximum-pixel encoding rate; every replicate received
      the same #p.test_count held-out images and encoding draws. Shading spans
      the minimum–maximum replicate accuracy, not a confidence interval. Rules
      mark 10% chance and the #pct(p.useful_accuracy) criterion.],
  )
  ]

  ]

  == Methods

  === Compute

  + *Retained computation.* We reused a completed calibration containing three
    trained decoders, their validation histories and their held-out correctness
    records. We did not rerun feature simulation or decoder training for this
    article.

  + *MNIST partitions.* Of the #r.training_dataset.image_shape.first() official
    training images, the first #p.train_count trained the decoders and the next
    #p.validation_count selected checkpoints; the remaining
    #(r.training_dataset.image_shape.first() - p.train_count - p.validation_count)
    were unused. Evaluation used the first #p.test_count images from the
    separate official test partition, with no overlap.

  + *Generate input events.* Each normalized pixel intensity $x_i in [0,1]$
    generated an independent binary event $s_i[k]$ at integration timestep
    $Delta t_"sim"=#p.dt_ms$ ms:

    $ p_("event",i) = (r_"input,max" x_i Delta t_"sim") / 1000,
      quad s_i[k] tilde "Bernoulli"(p_("event",i)). $ <exp080-events>

    Here $i$ indexes pixels, $k$ is the simulation-step index,
    $p_("event",i)$ is event probability, and $r_"input,max"$ is maximum-pixel
    encoding rate in spikes/s; 1000 converts milliseconds to seconds.

  + *Filter synaptic conductance.* Excitatory conductance $g_i[k]$, in μS,
    decayed each step by $exp(-Delta t_"sim"/tau_"AMPA")$ before an event added
    $w_"event"$. The AMPA time constant was $tau_"AMPA"=2$ ms and event
    strength was $w_"event"=#p.probe_uS$ μS.

  + *Integrate membrane voltage.* During simulation step $k$, the updated
    conductance $g_i[k]$ was held fixed while each non-spiking membrane voltage
    $V_(m,i)(t)$ obeyed

    $ C_m (d V_(m,i))/(d t) = g_L (E_L-V_(m,i)) + g_i[k](E_e-V_(m,i)). $ <exp080-membrane>

    Capacitance was $C_m=1$ nF, leak conductance $g_L=0.05$ μS,
    leak reversal $E_L=-65$ mV and excitatory reversal $E_e=0$ mV. Starting at
    zero conductance and $E_L$, voltage advanced by the exact exponential
    solution for that step. Simulation and decoder arithmetic used single
    precision.

  + *Form pixel features.* The feature $z_("feature",i)$, in mV, averaged
    post-update voltages above rest:

    $ z_("feature",i) = 1/N_t sum_(k=1)^(N_t) (V_(m,i)(k Delta t_"sim")-E_L)
      approx 1/T_"present" integral_0^(T_"present") (V_(m,i)(t)-E_L) dif t. $ <exp080-feature>

    Here $T_"present"=#p.presentation_ms$ ms,
    $N_t=T_"present"/Delta t_"sim"$ is the timestep count, and physical time
    at step $k$ is $t_k=k Delta t_"sim"$. Fresh encoding draws retained
    finite-window shot-noise effects without a stationary Gaussian
    approximation#cite(1).

  + *Train mixed-rate decoders.* At every epoch, we sampled the input rate for
    each training and validation presentation uniformly from
    #p.rates_hz.map(str).join(", ") Hz and generated a fresh encoding draw. Each
    784–1024–10 ReLU decoder was trained for #p.epochs epochs using
    cross-entropy, Adam with learning rate 0.001, no weight decay and batch size
    256.

  + *Independent training replicates.* Stochastic-stream identifiers
    #p.seeds.map(str).join(", ") defined independent model initializations,
    rate assignments and encoding draws.

  + *Select checkpoints.* At each of the #p.epochs eligible epochs, validation
    accuracy was the fraction correct across #p.validation_count validation
    presentations. The earliest epoch attaining the maximum validation accuracy
    supplied the selected checkpoint for each training replicate.

  + *Evaluate shared test features.* Every held-out image was simulated once
    at each tested rate, and all selected decoders received the same feature
    vector for that image and rate; the feed-forward decoder had no state across
    presentations. The predicted class $hat(y)$ was the class $c$ with the
    largest output logit $z_c$, and accuracy was measured per training replicate
    and rate.

  === Analyse

  #set enum(start: 11)

  + *Aggregate held-out accuracy.* For each rate and training replicate, we
    averaged correctness across #p.test_count held-out images. We then recorded
    the mean, minimum and maximum accuracy across the #p.seeds.len() training
    replicates at each rate.

  + *Select the tested interval.* The practical floor was the lowest tested
    rate where every decoder reached #pct(p.useful_accuracy) accuracy, and the
    ceiling was the highest tested rate. No interpolation was used; an empty
    qualifying set was reported as right-censored.

  === Present

  #set enum(start: 13)

  + *Present retained evidence.* We redrew the validation trajectories and
    rate-accuracy summary from recorded measurements, using the aggregation
    defined above without interpolation. We reused the original finite-window
    feature illustration unchanged rather than implying a new simulation.

  #run-view("exp080", inputs)

  == Appendix: Finite-window filtering and interpretation

  The Bernoulli input events form _shot noise_: each produces a discrete conductance
  jump followed by exponential AMPA decay. These pulses change both the voltage
  toward which the membrane moves and how quickly it moves there. A spike's
  An event's effect therefore depends on the voltage and conductance left by earlier
  events, rather than adding a fixed voltage increment.

  At low input rates, a finite presentation may contain no events, one event,
  or a few arriving at different times. Response statistics can change during
  the presentation (_nonstationary_), and their distribution can be asymmetric
  or concentrated around a few outcomes (_non-Gaussian_)#cite(1). Direct
  simulation retained these count and timing effects rather than replacing
  them with a steady, bell-shaped approximation.

  Neural decoding measures information accessible to a specified readout,
  not absolute information content or the mechanism by which a biological
  population uses it#cite(2). The criterion therefore selects a practical
  interval for this representation and decoder. Transfer to a PING network
  requires a separate evaluation; decoding accuracy here does not establish
  gamma timing benefits.

  #reference-list((
    (
      text: [Brigham & Destexhe: _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
    (
      text: [Quian Quiroga & Panzeri: _Extracting Information from Neuronal Populations: Information Theory and Decoding Approaches_. Nature Reviews Neuroscience, 2009.],
      doi: "10.1038/nrn2578",
    ),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [This experiment asks how maximum-pixel input rate affects decoder accuracy after synaptic and membrane filtering.],
    preview-figures, json-inputs: ("exp080",),
  )
}

#let meta = meta + (assets: input-assets("exp080", inputs))
#let body = with-datasets("exp080", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
