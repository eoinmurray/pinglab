#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp080")

#let meta = (
  status: "[▦ DATA | v28.0.0]",
  title: "Calibrating Accuracy Across Input Rates",
  date: "2026-08-10",
  updated_at: "2026-08-31",
  description: "Direct-simulation decoder calibration of the input-rate range for later variable-rate PING training.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp080",)
#let preview-figures = (
  (path: "exp080/training_history.svg", label: "training history"),
  (path: "exp080/feature_images.png", label: "feature images"),
  (path: "exp080/psychometric.svg", label: "psychometric"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp080/numbers.json"))
#let p = r.parameters
#let d = r.decision
#let criterion-crossed = d.criterion_crossed
#let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
#set math.equation(numbering: "(1)")
#counter(math.equation).update(0)
#show math.equation.where(block: true): equation => context {
  if target() == "html" {
    html.elem("div", attrs: (class: "exp080-equation", style: "display:flex;align-items:center;gap:1em"), {
      html.elem("div", attrs: (style: "flex:1;min-width:0"), equation)
      html.elem("span", numbering("(1)", ..counter(math.equation).at(equation.location())))
    })
  } else { equation }
}
#let body = [
  == Abstract


  Asked which input-rate regime makes filtered pixel features sufficiently
  informative for a downstream MNIST decoder. Simulated the synaptic and
  membrane feature pipeline, trained nonlinear decoders across input rates and
  evaluated frozen checkpoints on held-out digits.

  The decoders established a tested interval where the feature representation
  supported useful classification, with weak-drive conditions remaining
  inadequate. Calibrates the feature and decoder pipeline, not PING-network
  accuracy or performance between the explicitly tested rates.

  == Results

  #with-result-sections[

  === Mixed-rate decoder validation accuracy across training

  The first validation maximum selected each decoder; selected epochs were
  #r.training.map(record => str(record.selected_epoch)).join(", ").

  #figure(
    data-image(data-file("exp080/training_history.svg"), width: 85%,
      alt: "Mixed-rate validation accuracy over training epochs, with one curve per decoder."),
    caption: [Mixed-rate validation accuracy for #p.seeds.len() independently
      trained decoders, using fresh feature simulations each epoch.],
  )

  === Filtered digit features at 0.5, 5 and 25 Hz

  Sparse input left fragments of the digit rather than a uniformly attenuated
  image.

  #figure(
    data-image(data-file("exp080/feature_images.png"), width: 100%,
      alt: "One MNIST input digit and filtered feature images at 0.5, 5 and 25 Hz."),
    caption: [Illustrative input digit and directly simulated features at
      #p.rates_hz.at(2), #p.rates_hz.at(5) and #p.rates_hz.last() Hz, using
      #p.probe_uS μS conductance increments and #p.presentation_ms ms
      presentations. Feature panels share a 0–65 mV scale and independent
      spike realizations. #if r.illustration.kind == "historical-image" [This
      illustration is reused from the original calibration; it is not a new
      simulation.]],
  )

  === Held-out decoder accuracy across tested input rates

  #if criterion-crossed [The selected floor was #d.r_train_hz Hz, with mean
  accuracy
  #pct(d.rows.filter(row => row.rate_hz == d.r_train_hz).first().accuracy).] else [No
  rate met the criterion for every decoder; the floor was right-censored at
  #d.recommendation.ceiling_hz Hz.] This interval is a decoder-relative
  calibration, not evidence of PING-network performance.

  #figure(
    data-image(data-file("exp080/psychometric.svg"), width: 85%,
      alt: "Held-out accuracy across tested input rates, showing the decoder mean and minimum-to-maximum range."),
    caption: [Mean held-out accuracy across #p.test_count images and
      #p.seeds.len() decoders; shading spans the lowest and highest decoder
      accuracy, not a confidence interval. Rules mark 10% chance and the
      #pct(p.useful_accuracy) criterion.],
  )

  ]

  == Methods

  We calibrated which input rates preserved usable digit information after
  independent synaptic and membrane filtering of each pixel.

  #enum(
    [*Partition MNIST.* Of the #r.training_dataset.image_shape.first() official
      training images, the first #p.train_count trained the decoders and the
      next #p.validation_count selected checkpoints; the remaining
      #(r.training_dataset.image_shape.first() - p.train_count - p.validation_count)
      were unused. Evaluation used the first #p.test_count images from the
      separate official test partition, with no overlap.],

    [*Simulate filtered features.* Each normalized pixel intensity $x_i in [0,1]$
      generated an independent binary event $s_i[k]$ at integration timestep
      $Delta t_"sim"=#p.dt_ms$ ms:

      $ s_i[k] tilde "Bernoulli"((r_"input,max" x_i Delta t_"sim") / 1000). $ <exp080-events>

      Here $i$ indexes pixels, $k$ is the update index, and $r_"input,max"$ is maximum-pixel
      encoding rate in spikes/s; 1000 converts milliseconds to seconds.
      Excitatory conductance $g_i (t)$, in μS, decayed before each event increment:

      $ g_i[k] = exp(-(Delta t_"sim") / tau_"AMPA") g_i[k-1] + w_"event" s_i[k]. $ <exp080-conductance>

      The AMPA decay time was $tau_"AMPA"=2$ ms and event strength
      $w_"event"=#p.probe_uS$ μS. Each non-spiking membrane voltage $V_(m,i) (t)$ obeyed

      $ C_m^E (d V_(m,i))/(d t) = g_L^E (E_L-V_(m,i)) + g_i (t)(E_e-V_(m,i)). $ <exp080-membrane>

      Capacitance was $C_m^E=1$ nF, leak conductance $g_L^E=0.05$ μS,
      leak reversal $E_L=-65$ mV and excitatory reversal $E_e=0$ mV.
      Starting at zero conductance and $E_L$, voltage advanced by the exact
      exponential solution with each updated conductance held fixed for one
      timestep. The feature $z_("feature",i)$, in mV, averaged post-update voltages above rest:

      $ z_("feature",i) = 1/N_t sum_(k=1)^(N_t) (V_(m,i) (k Delta t_"sim")-E_L)
        approx 1/T_"present" integral_0^(T_"present") (V_(m,i) (t)-E_L) dif t. $ <exp080-feature>

      Here $T_"present"=#p.presentation_ms$ ms, $N_t=T_"present"/(Delta t_"sim")$ is the timestep count, and
      $k$ indexes updates. Fresh event trains retained finite-window shot-noise
      effects without a stationary Gaussian approximation#cite(1).],

    [*Train mixed-rate decoders.* Each training and validation presentation
      sampled uniformly from #p.rates_hz.map(str).join(", ") Hz. A 784–1024–10
      ReLU decoder used cross-entropy and Adam with learning rate 0.001,
      batch size 256 and #p.epochs epochs. Seeds #p.seeds.map(str).join(", ")
      defined independent initializations, rate assignments and spike trains;
      the first maximum validation accuracy selected one checkpoint per seed.],

    [*Evaluate shared test features.* Every held-out image was simulated once
      at each tested rate. All frozen decoders received the same feature for
      that image and rate, so their differences did not arise from different
      test spike realizations. Accuracy was the fraction of correctly
      classified images for each decoder and rate.],

    [*Select the tested interval.* The practical floor was the lowest tested
      rate where every decoder reached #pct(p.useful_accuracy) accuracy:

      $ r_"train" = min {r in cal(R): min_(s in cal(S)) A_s (r) >= 0.5}. $ <exp080-floor>

      Here $cal(R)$ is the tested rate set, $cal(S)$ the decoder seed set,
      $A_s (r)$ the accuracy of decoder $s$ at rate $r$, and $r_"train"$ the
      selected floor. The ceiling was the highest tested rate; no interpolation
      was used. An empty qualifying set was reported as right-censored.],
  )

  #run-view("exp080", inputs)

  == Appendix: Finite-window filtering and interpretation

  The random spikes form _shot noise_: each produces a discrete conductance
  jump followed by exponential AMPA decay. These pulses change both the voltage
  toward which the membrane moves and how quickly it moves there. A spike's
  effect therefore depends on the voltage and conductance left by earlier
  spikes, rather than adding a fixed voltage increment.

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
    [How does input rate affect classification accuracy? Calibrate the rate-response curve using controlled visual features and a recorded training history.],
    preview-figures, json-inputs: ("exp080",),
  )
}

#let meta = meta + (assets: input-assets("exp080", inputs))
#let body = with-datasets("exp080", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
