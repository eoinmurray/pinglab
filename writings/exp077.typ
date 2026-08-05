#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Filter-matched rate calibration for variable-rate PING training",
  date: "2026-08-05",
  description: "A staged empirical and linear calibration maps Poisson-encoded MNIST pixels into subthreshold voltage features, measures their decodability with a variable-rate ANN, and selects a lower rate for subsequent PING training.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let presentation-ms = 200
#let dt-ms = 0.1
#let training-rates-hz = (0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 25)
#let probe-us = (0.6, 1.2, 2.4)
#let seeds = (42, 43, 44)
#let r = json("/artifacts/data/exp077/numbers.json")
#let rounded(x, digits: 3) = str(calc.round(x, digits: digits))
#let training-rate-text = training-rates-hz.map(str).join(", ")
#let probe-text = probe-us.map(str).join(", ")
#let seed-text = seeds.map(str).join(", ")
#let staged-pending(caption: none, note: [figure pending], ratio: 16 / 9) = block(
  width: 100%,
  breakable: false,
  fill: luma(249),
  radius: 5pt,
  stroke: (thickness: 0.75pt, paint: luma(203), dash: "dashed"),
  inset: 10pt,
)[
  *Pending plot.* #note
  #linebreak()
  _Planned caption._ #caption
]

#let body = [
  == Abstract

  This experiment will find the lowest Poisson rates that preserve MNIST class
  evidence. An artificial neural network (ANN) will classify static features
  formed by passing each pixel's spike train through an AMPA synapse and
  non-spiking membrane, then averaging its voltage over the presentation.

  ANN accuracy against rate is the primary result; a linear transfer-function
  calculation checks the empirical feature variance. The output is a justified
  lower rate for later variable-rate PING training.

  Step 1 is now complete. The local generator passed all
  #r.validations.len() focused checks, including agreement with the shared
  `tools/snn` cell below the registered numerical tolerance. The remaining
  response-library, decoder, and threshold stages have not been run.

  == Purpose and scope

  The question is:

  #quote(block: true)[At a given encoding rate, does the temporally filtered
    image still contain enough information for an ANN to recognize the digit?]

  The target PING architecture has 784 pixel channels feeding conductance-based
  excitatory cells. Its fixed synaptic and membrane equations define the
  features, without trained input, recurrent, or output weights. The ANN thus
  measures filtered-input decodability independently of a trained PING network.

  The experimental logic is:

  #enum(
    [Convert 0.25--25 Hz Poisson inputs into static, filter-matched features.],
    [Measure their distributions and test a linear variance prediction.],
    [Train the ANN across the rate grid with the held-out test set sealed.],
    [Measure ANN accuracy and locate its lowest decodable rates.],
  )

  #list(
    [Nonlinear ANN above chance: this decoder can access class evidence.],
    [Nonlinear ANN at chance: this representation and decoder cannot extract
      reliable evidence.],
    [Nonlinear ANN succeeds but linear decoder fails: the evidence is not
      linearly accessible.],
  )

  The ANN threshold is a decoder-relative decodability edge, not an absolute
  information boundary or a prediction of PING accuracy. The practical training
  floor instead applies a predeclared useful-accuracy criterion. A later
  experiment must test whether PING can train across the selected range.

  == Methods

  1. *Generated and validated filter-matched pixel features.* Every image used
    a #presentation-ms ms presentation and #dt-ms ms timestep. The tested rates
    are #training-rate-text Hz; dense sampling below 5 Hz resolves the lower
    edge, while 25 Hz is the proposed later training ceiling. Training samples
    uniformly from these points. Seeds #seed-text define independent feature,
    response-library, and ANN runs.

    `protocol.json` records the grid and deterministic MNIST train,
    validation, and held-out indices. Pixel-intensities x remain in [0, 1]. If r
    is the encoding rate at pixel-intensity one, the expected pixel rate is r x.

    Each pixel drives an independent AMPA synapse and uncoupled, non-spiking
    excitatory membrane, capturing AMPA decay, conductance-dependent integration,
    and within-window timing that raw counts omit. This probe has no threshold,
    reset, recurrence, or trained weights. It uses the target PING cell's
    decay-then-add synapse and exponential-Euler membrane update, implemented
    locally without changing the shared simulator.

    The complete feature path is

    $ S_i (t) -> g_i^"pix" (t) -> v_i (t) -> z_i -> "ANN". quad "(1)" $

    Here S#sub[i], g#super[pix]#sub[i], v#sub[i], and z#sub[i] are pixel i's spike
    train, AMPA conductance, subthreshold voltage, and time-averaged feature;
    t is time and ANN is the classifier.

    At each timestep, we drew the encoded input according to

    $ S_i (t) tilde "Bernoulli"(r Delta t x_i). quad "(2)" $

    Here x#sub[i] is grayscale pixel-intensity, r is the encoding rate in spikes
    per second at pixel-intensity one, Δt is the timestep in seconds, and
    Bernoulli is an independent binary draw. Configurations with probability
    above one are invalid.

    We updated the AMPA conductance using

    $ g_i^"pix" (t) = beta_"AMPA" g_i^"pix" (t-1) + w_"probe" S_i (t). quad "(3)" $

    $ beta_"AMPA" = exp(-(Delta t) / tau_"AMPA"). quad "(4)" $

    Here β#sub[AMPA] is one-timestep conductance retention, τ#sub[AMPA] is AMPA
    decay time, w#sub[probe] is conductance added per spike, and exp is the
    exponential function. Other symbols follow Equation 2. The primary 1.2 μS
    probe is the target architecture's nominal mean initial pixel-to-excitatory
    conductance; 0.6 and 2.4 μS test half and double scale.

    We integrated the membrane without thresholding or resetting:

    $ C_E (d v_i) / (d t) = g_"L,E" (E_L - v_i) + g_i^"pix" (t) (E_e - v_i). quad "(5)" $

    $ tau_"eff",i (t) = C_E / (g_"L,E" + g_i^"pix" (t)). quad "(6)" $

    Here C#sub[E] is excitatory capacitance; g#sub[L,E] is leak conductance;
    E#sub[L] and E#sub[e] are leak and AMPA reversal potentials; and
    τ#sub[eff,i] is the instantaneous conductance-dependent time constant. Other
    symbols follow Equation 1.

    We initialized voltage at E#sub[L] and conductance at zero, simulated one
    complete presentation, and calculated

    $ z_i = 1 / T integral_0^T (v_i (t) - E_L) dif t. quad "(7)" $

    Here T is presentation duration and z#sub[i] is mean baseline-subtracted
    voltage. We will not divide by rate or disclose it to either decoder, so
    lower rates retain their weaker, noisier signal.

    Focused tests covered encoder probability, count moments, pixel independence,
    AMPA decay, exponential-Euler voltage updates, deterministic replay,
    agreement with one uncoupled target-PING excitatory cell, and spike-timing
    sensitivity. All #r.validations.len() checks passed. At the seeded count
    validation point, the expected and observed mean counts were
    #rounded(r.validations.spike_count_moments.expected_mean) and
    #rounded(r.validations.spike_count_moments.empirical_mean); the largest
    absolute inter-channel count correlation was
    #rounded(r.validations.pixel_independence.maximum_absolute_count_correlation).

    The corresponding plot is defined in Results, Step 1.

  2. *Measure and retain the empirical pixel-feature distributions.* We will run
    K independent Step 1 draws for all 256 grayscale levels,
    #training-rates-hz.len() rates, and three probe conductances. A predeclared
    pilot will choose K from convergence of the mean and variance, subject to a
    maximum K.

    We will estimate the conditional mean

    $ hat(mu)_z (x, r, w_"probe") = 1 / K sum_(k=1)^K z^(k). quad "(8)" $

    and conditional variance

    $ hat(sigma)_z^2 (x, r, w_"probe") = 1 / (K - 1) sum_(k=1)^K (z^(k) - hat(mu)_z)^2. quad "(9)" $

    Here x, r, and w#sub[probe] are pixel-intensity, rate, and probe conductance;
    z#super[(k)] is draw k; K is the draw count; and Equations 8 and 9 estimate
    the conditional mean and variance.

    We will retain all K simulated values as the primary empirical response
    library. During ANN feature generation, we will draw one value according to

    $ J tilde "DiscreteUniform"(1, dots.c, K), quad z_"sample" = z^(J). quad "(10)" $

    Here J is uniform on the K draws and z#sub[sample] is the sampled feature;
    other symbols follow Equations 8 and 9. Mean and variance summarize noise,
    but the ANN samples the retained empirical values because low-rate responses
    may be zero-heavy, skewed, and non-Gaussian.

    Committed artifacts will contain plot-ready moments and a manifest for the
    scratch response library, including checksum, dimensions, seed recipe, and
    regeneration command. The calibration has at most
    #(256 * training-rates-hz.len() * probe-us.len()) conditions.

    The corresponding plot is defined in Results, Step 2.

  3. *Compare the empirical variance with a linear-filter prediction.* This
    supplementary check asks whether a local linear approximation explains the
    Step 2 feature variance. It will not set the training range.

    For mean spike rate λ, stationary mean conductance is

    $ bar(g)_lambda = lambda w_"probe" tau_"AMPA". quad "(11)" $

    and stationary mean voltage is

    $ bar(v)_lambda = (g_"L,E" E_L + bar(g)_lambda E_e) /
      (g_"L,E" + bar(g)_lambda). quad "(12)" $

    Here λ is encoding rate times pixel-intensity; bar(g)#sub[λ] and
    bar(v)#sub[λ] are stationary mean conductance and voltage. Other symbols
    follow Step 1.

    Linearizing the synapse and membrane around that operating point gives

    $
      G_lambda(omega) = w_"probe" / (i omega + 1 / tau_"AMPA") dot
      (E_e - bar(v)_lambda) /
      (i omega C_E + g_"L,E" + bar(g)_lambda). quad "(13)"
    $

    Here G#sub[λ] (ω) is the local synapse-plus-membrane transfer function; ω is
    angular frequency; and i is the imaginary unit. Because mean conductance
    changes gain and effective membrane time constant, Equation 13 defines a
    family of responses.

    The finite averaging window contributes

    $ A_T(omega) = (1 - exp(-i omega T)) / (i omega T). quad "(14)" $

    $ H_lambda(omega) = A_T(omega) G_lambda(omega). quad "(15)" $

    Here A#sub[T] (ω) is the duration-T averaging response; H#sub[λ] (ω) is the
    complete response from input fluctuation to averaged feature; T is
    presentation duration; and exp is the exponential function.

    For centred ideal Poisson input,

    $ S_"in"(omega) = lambda. quad "(16)" $

    $ S_z(omega) = abs(H_lambda(omega))^2 S_"in"(omega). quad "(17)" $

    $
      "Var"_"linear"(z) = 1 / (2 pi) integral_(-oo)^oo
      abs(H_lambda(omega))^2 S_"in"(omega) dif omega. quad "(18)"
    $

    Here S#sub[in] (ω) and S#sub[z] (ω) are input and predicted output power
    spectral densities; |H#sub[λ] (ω)|#super[2] is transmitted noise power;
    Var#sub[linear] (z) is predicted feature variance; and π is the circle
    constant. Appendix A derives Equations 11--18.

    We will evaluate every distinct calibration point, compare predicted with
    empirical Step 2 variance, and validate low-, middle-, and high-drive gains
    by sinusoidally modulating the numerical probe. This stationary,
    continuous-time, linearized Poisson model approximates the finite, discrete
    Bernoulli simulation and remains diagnostic only.

    The corresponding plot is defined in Results, Step 3.

  4. *Construct and validate complete feature images.* For each MNIST image, we
    will select one of the #training-rates-hz.len() rates uniformly and
    sample each pixel from Equation 10, resampling every training epoch. Sampling
    is independent across pixels because each channel and probe is independent.

    Before fitting, the recorded MNIST training partition will be split into
    decoder-training and validation subsets; the test set remains sealed. On
    predeclared images at low, transitional, and high rates, the empirical
    sampler must match direct Step 1 simulation in pixel and image moments, zero
    fractions, and representative images.

    This validation must pass before ANN training. The corresponding plot is
    defined in Results, Step 4.

  5. *Train the mixed-rate decoders.* We will train a primary ANN with 784
    inputs, one 1,024-unit rectified-linear hidden layer, and ten outputs. The
    ANN learns both weight matrices from voltage features. A regularized linear
    softmax decoder on the same images tests linear accessibility. Neither model
    receives the rate or pretrained weights.

    Seeds #seed-text use Adam, learning rate 0.001, batch size 256, and initially
    at most 15 epochs. Every presentation samples a rate uniformly and
    regenerates features. Validation alone controls model selection,
    regularization, early stopping, and any epoch extension; test data remain
    sealed until Step 6. We will record each configuration, selected epoch, and
    training history.

    The primary ensemble uses the 1.2 μS probe; separate 0.6 and 2.4 μS runs test
    sensitivity. Conductance conditions are neither pooled nor disclosed.

    The corresponding plot is defined in Results, Step 5.

  6. *Generate inference-only psychometric curves and choose two rate
    thresholds.* We will freeze all decoders before testing. At every tested
    rate, seeds use the same held-out images and fixed feature draws; additional
    draws measure encoding variability. The primary curve is

    $ A_r (r) = P("correct" | r, "mixed-rate nonlinear decoder"). quad "(19)" $

    Here A#sub[r] is held-out nonlinear-ANN accuracy, P is correct-classification
    probability, and r is rate. The linear decoder is diagnostic only.

    Bootstrap resampling of held-out images, probe draws, response-library
    simulations, and decoder seeds gives lower confidence bound L#sub[r]. The
    decoder-relative edge is

    $ r_"decode" = "lowest " r in cal(R) " satisfying " L_r (r) > 1 / N_"class". quad "(20)" $

    The practical training floor is

    $ r_"train" = "lowest " r in cal(R) " satisfying " L_r (r) >= a_"use". quad "(21)" $

    Here $cal(R)$ is the tested grid; N#sub[class] is the number of classes;
    r#sub[decode] is the lowest rate reliably above chance; a#sub[use] is the
    predeclared 50% useful-accuracy target; and r#sub[train] is the lowest rate
    whose lower bound reaches it. The target is locked before testing.

    Primary thresholds use the 1.2 μS nonlinear ANN. The linear decoder and 0.6
    and 2.4 μS runs show sensitivity to decoder capacity and conductance. We will
    not treat interpolated rates as observations.

    The corresponding plot is defined in Results, Step 6.

  7. *Report the training-range decision and stop.* We will write
    `decision.json` with r#sub[decode], r#sub[train], their uncertainty, the
    probe-conductance sensitivity, decoder and artifact hashes, and all rule
    outcomes. The recommended later PING range is r#sub[train]--25 Hz. If the
    floor shifts by more than one adjacent grid point across conductances, we
    will report plausible floors rather than one value. This experiment stops at
    the recommendation; a separate experiment must train and test PING across it.

  #block(breakable: false)[
    == Results

    Step 1 is complete; pending panels in later steps imply no result.

    === Step 1: filter-matched feature generation

    #image("/artifacts/data/exp077/probe_dynamics.svg", width: 100%)

    _Finite-window probe dynamics._ Panels A--C show input spikes, AMPA
    conductance, and subthreshold voltage for no-spike, early-spike, and
    late-spike inputs. Panel D places the single spike across the presentation
    and reports the resulting mean voltage feature. With the same one-spike
    count, a spike at
    #rounded(r.validations.spike_timing_sensitivity.early_spike_ms) ms produced
    z = #rounded(r.validations.spike_timing_sensitivity.early_z_mV) mV, whereas
    one at #rounded(r.validations.spike_timing_sensitivity.late_spike_ms) ms
    produced #rounded(r.validations.spike_timing_sensitivity.late_z_mV) mV.
    Earlier input therefore contributed
    #rounded(r.validations.spike_timing_sensitivity.early_minus_late_z_mV) mV
    more to the finite-window average.
  ]

  The decay-then-add check reproduced the first two conductances with a maximum
  absolute error of #rounded(r.validations.ampa_decay_then_add.maximum_absolute_error_uS)
  μS. The independently calculated exponential-Euler step and the complete local
  trajectory both matched the existing `tools/snn` uncoupled cell within their
  registered machine-precision tolerances. This validates the Step 1 feature
  generator only; it does not establish MNIST decodability or select a PING
  input-rate range.

  === Step 2: empirical response library

  #staged-pending(
    caption: [Empirical feature distributions. Panels show the mean and standard
      deviation of z#sub[i] (mV) across grayscale pixel-intensity x (0--1),
      encoding rate r (Hz) at pixel-intensity one, and probe conductance w#sub[probe]
      (μS); representative low-, transitional-, and high-rate distributions;
      and convergence with Monte Carlo draw count K. Probe-conductance colours
      and markers are consistent across panels.],
    note: [Pending Step 2. Planned file: `response_library.png`.],
    ratio: 16 / 9,
  )

  === Step 3: linear-filter prediction

  #staged-pending(
    caption: [Linear response and variance prediction. A log-frequency (Hz) Bode
      panel shows the normalized complete-response magnitude (dB), including
      200 ms averaging, at five operating points (A). Panel B compares predicted
      and empirical variance with an identity line; C plots their ratio against
      expected count (shape: probe conductance; colour: count).],
    note: [Pending Step 3. Planned file: `linear_filter.svg`.],
    ratio: 16 / 9,
  )

  === Step 4: complete feature images

  #staged-pending(
    caption: [Feature images at representative rates. Matched originals, library
      samples, and direct simulations are compared by pixel and image moments,
      zero fractions, and image agreement.],
    note: [Pending Step 4. Planned file: `feature_images.png`.],
    ratio: 16 / 9,
  )

  === Step 5: mixed-rate decoder training

  #staged-pending(
    caption: [Mixed-rate training. Training and validation loss and accuracy are
      shown by epoch and seed for both decoders; a rate histogram checks uniform
      sampling. Held-out accuracy is excluded.],
    note: [Pending Step 5. Planned file: `decoder_training.svg`.],
    ratio: 16 / 9,
  )

  #block(breakable: false)[
    === Step 6: ANN psychometric curve and thresholds

    #staged-pending(
      caption: [Decoder-relative psychometrics at 200 ms, spanning 0.25--25 Hz
        with a 0.25--5 Hz inset. Panel A compares frozen nonlinear and linear
        decoders at 1.2 μS, with chance and both thresholds; B compares nonlinear
        accuracy at 0.6, 1.2, and 2.4 μS. Uncertainty covers encoding and library
        draws and ANN seeds.],
      note: [Pending Step 6. Planned file: `ann_psychometric.svg`.],
      ratio: 16 / 9,
    )
  ]

  === Step 7: handoff

  *Pending output.* `decision.json` will report the decoder-relative edge,
  practical training floor, uncertainty, probe-conductance sensitivity, and the
  recommended rate range for a separate variable-rate PING training experiment.

  == Relation to prior work

  Filtered conductance shot noise can violate fixed-time-constant models#cite(1),
  and Gaussian approximations can miss skewed responses#cite(2), supporting the
  empirical response library. Decoder performance is not absolute neural
  information#cite(3), and linear and nonlinear decoders can differ#cite(4),
  motivating a decoder-relative edge and both decoder diagnostics.

  #reference-list((
    (
      text: [Wolff & Lindner: _Mean, Variance, and Autocorrelation of Subthreshold Potential Fluctuations Driven by Filtered Conductance Shot Noise_. Neural Computation, 2010.],
      doi: "10.1162/neco.2009.02-09-958",
    ),
    (
      text: [Brigham & Destexhe: _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
    (
      text: [Quian Quiroga & Panzeri: _Extracting Information from Neuronal Populations: Information Theory and Decoding Approaches_. Nature Reviews Neuroscience, 2009.],
      doi: "10.1038/nrn2578",
    ),
    (
      text: [Warland, Reinagel & Meister: _Decoding Visual Information From a Population of Retinal Ganglion Cells_. Journal of Neurophysiology, 1997.],
      doi: "10.1152/jn.1997.78.5.2336",
    ),
  ))

  == Appendix A: linear-filter derivation

  This appendix derives Equations 11--18 from the same AMPA synapse and
  subthreshold membrane used by the empirical probe. It replaces the finite
  Bernoulli drive with a continuous-time Poisson point process and linearizes
  the membrane around the mean drive at each calibration condition.

  === A.1 Stationary operating point

  Represent pixel i's idealized spike train and synaptic conductance by

  $
    s_i(t) = sum_k delta(t - t_k), quad
    (d g_i(t)) / (d t) = -g_i(t) / tau_"AMPA" + w_"probe" s_i(t). quad "(A1)"
  $

  Here s#sub[i] (t) is a sum of Dirac impulses; t#sub[k] is spike k's time; δ is
  the Dirac delta; g#sub[i] (t) is AMPA conductance; τ#sub[AMPA] is its decay time;
  w#sub[probe] is conductance added per spike; and t is time. Equation A1 is the
  continuous-time counterpart of the decay-then-add update in Equation 3.

  For stationary Poisson rate λ, the mean conductance satisfies

  $
    0 = -bar(g)_lambda / tau_"AMPA" + w_"probe" lambda, quad
    bar(g)_lambda = lambda w_"probe" tau_"AMPA". quad "(A2)"
  $

  Here λ is encoding rate times pixel-intensity, and
  bar(g)#sub[λ] is stationary mean conductance. The zero on the left states that
  the mean no longer changes with time.

  Setting the mean membrane derivative in Equation 5 to zero gives

  $
    0 = g_"L,E" (E_L - bar(v)_lambda) + bar(g)_lambda (E_e - bar(v)_lambda), quad
    bar(v)_lambda = (g_"L,E" E_L + bar(g)_lambda E_e) / (g_"L,E" + bar(g)_lambda). quad "(A3)"
  $

  Here bar(v)#sub[λ] is stationary mean voltage; g#sub[L,E] is leak
  conductance; and E#sub[L] and E#sub[e] are leak and AMPA reversal potentials.
  Other symbols follow Equation A2.

  === A.2 Local synapse-plus-membrane response

  Write each signal as its operating point plus a small fluctuation:

  $
    g_i = bar(g)_lambda + delta g_i, quad
    v_i = bar(v)_lambda + delta v_i, quad
    s_i = lambda + delta s_i. quad "(A4)"
  $

  Here δg#sub[i], δv#sub[i], and δs#sub[i] are complete conductance, voltage, and
  input perturbation variables around the means defined in Equations A2 and A3.

  To derive Equation A5, substitute the conductance and input decompositions
  from Equation A4 into the synapse equation, Equation A1:

  $
    (d (bar(g)_lambda + delta g_i)) / (d t)
      = -(bar(g)_lambda + delta g_i) / tau_"AMPA"
      + w_"probe" (lambda + delta s_i).
  $

  The operating point is stationary, so its time derivative is zero. Expanding
  the right-hand side then gives

  $
    (d delta g_i) / (d t)
      = (-bar(g)_lambda / tau_"AMPA" + w_"probe" lambda)
      - frac(delta g_i, tau_"AMPA") + w_"probe" delta s_i.
  $

  The parenthesized stationary terms cancel by Equation A2. The remaining
  perturbation dynamics are

  $
    (d delta g_i) / (d t) = -frac(delta g_i, tau_"AMPA")
      + w_"probe" delta s_i. quad "(A5)"
  $

  To derive Equation A6, substitute the conductance and voltage decompositions
  from Equation A4 into the membrane equation, Equation 5:

  $
    C_E (d (bar(v)_lambda + delta v_i)) / (d t)
      = g_"L,E" (E_L - bar(v)_lambda - delta v_i)
      + (bar(g)_lambda + delta g_i)
        (E_e - bar(v)_lambda - delta v_i).
  $

  As above, the stationary mean has zero time derivative. Expanding and grouping
  the right-hand side gives

  $
    C_E (d delta v_i) / (d t)
      = (g_"L,E" (E_L - bar(v)_lambda)
        + bar(g)_lambda (E_e - bar(v)_lambda))
      - (g_"L,E" + bar(g)_lambda) delta v_i
      + (E_e - bar(v)_lambda) delta g_i
      - delta g_i delta v_i.
  $

  The first parenthesized group is zero by Equation A3. The final product is
  second order in the small perturbations and is omitted by the local linear
  approximation. This leaves

  $
    C_E (d delta v_i) / (d t) = -(g_"L,E" + bar(g)_lambda) delta v_i
      + (E_e - bar(v)_lambda) delta g_i. quad "(A6)"
  $

  Here C#sub[E] is excitatory-cell capacitance. All other symbols follow
  Equations A1--A4.

  To derive Equation A7, take the Fourier transform of Equation A5. The
  derivative becomes iω times the transformed signal:

  $
    i omega delta g_i(omega)
      = -frac(delta g_i(omega), tau_"AMPA")
      + w_"probe" delta s_i(omega).
  $

  Move both conductance terms to the left, then divide by their common factor:

  $
    (i omega + 1 / tau_"AMPA") delta g_i(omega)
      = w_"probe" delta s_i(omega),
  $

  $
    delta g_i(omega) = w_"probe" / (i omega + 1 / tau_"AMPA") delta s_i(omega). quad "(A7)"
  $

  Equation A8 follows in the same way from Equation A6. Fourier transformation
  first gives

  $
    i omega C_E delta v_i(omega)
      = -(g_"L,E" + bar(g)_lambda) delta v_i(omega)
      + (E_e - bar(v)_lambda) delta g_i(omega).
  $

  Collect the voltage terms on the left:

  $
    (i omega C_E + g_"L,E" + bar(g)_lambda) delta v_i(omega)
      = (E_e - bar(v)_lambda) delta g_i(omega).
  $

  Dividing by the coefficient of δv#sub[i] (ω) gives

  $
    delta v_i(omega) = (E_e - bar(v)_lambda) /
      (i omega C_E + g_"L,E" + bar(g)_lambda) delta g_i(omega). quad "(A8)"
  $

  Here ω is angular frequency, i is the imaginary unit, and each argument ω
  denotes a Fourier-domain signal.

  Equations A5--A8 form a *linear time-invariant (LTI)* system after
  linearization about the operating point λ. Because the system has been
  linearized about the operating point, it is locally linear and
  time-invariant, so the standard transfer-function formalism applies. The
  canonical frequency-domain input--output relationship is

  $
    delta v_i(omega) = G_lambda(omega) delta s_i(omega).
  $

  Here G#sub[λ] (ω) is the local transfer function relating input spike-train
  perturbations to membrane-voltage perturbations. For non-zero input, the
  input--output relationship can be rearranged as

  $
    G_lambda(omega) = (delta v_i(omega)) / (delta s_i(omega)).
  $

  Thus the ratio follows from the standard input--output relationship rather
  than serving as an independent definition. To derive G#sub[λ] (ω), start from
  Equation A7:

  $
    delta g_i(omega)
      = w_"probe" / (i omega + 1 / tau_"AMPA") delta s_i(omega).
  $

  Substitute this expression for δg#sub[i] (ω) into Equation A8:

  $
    delta v_i(omega)
      = (E_e - bar(v)_lambda) /
        (i omega C_E + g_"L,E" + bar(g)_lambda)
        (w_"probe" / (i omega + 1 / tau_"AMPA") delta s_i(omega)).
  $

  Reorder the scalar factors and factor out δs#sub[i] (ω):

  $
    delta v_i(omega)
      = (w_"probe" / (i omega + 1 / tau_"AMPA") dot
        (E_e - bar(v)_lambda) /
        (i omega C_E + g_"L,E" + bar(g)_lambda))
        delta s_i(omega).
  $

  Comparing this result with the canonical LTI relationship identifies the
  coefficient multiplying δs#sub[i] (ω) as the transfer function:

  $
    G_lambda(omega)
      = w_"probe" / (i omega + 1 / tau_"AMPA") dot
        (E_e - bar(v)_lambda) /
        (i omega C_E + g_"L,E" + bar(g)_lambda). quad "(A9)"
  $

  Equation A9 is therefore the synaptic filter multiplied by the membrane
  filter, not an arbitrarily chosen ratio. Its λ-dependence captures the change
  in membrane gain and effective time constant with mean conductance.

  === A.3 Finite-window averaging

  Start from the baseline-subtracted Step 1 feature in Equation 7:

  $
    z_i = 1 / T integral_0^T (v_i(t) - E_L) dif t.
  $

  Within the stationary linearized model used in this appendix, substitute the
  voltage decomposition from Equation A4:

  $
    z_i = 1 / T integral_0^T
      (bar(v)_lambda + delta v_i(t) - E_L) dif t.
  $

  The operating-point voltage and leak reversal potential are constant in time,
  so their integral is their difference multiplied by T. Dividing by T gives

  $
    z_i = bar(v)_lambda - E_L
      + 1 / T integral_0^T delta v_i(t) dif t.
  $

  Define the operating-point feature and its perturbation by

  $
    bar(z)_lambda = bar(v)_lambda - E_L, quad
    delta z_i = z_i - bar(z)_lambda.
  $

  Subtracting the operating-point feature from the expanded expression for
  z#sub[i] cancels the constant term and leaves

  $
    delta z_i = 1 / T integral_0^T delta v_i(t) dif t. quad "(A10)"
  $

  Thus δz#sub[i] is the fluctuation around the operating-point feature, obtained
  by averaging the voltage perturbation over the 200 ms presentation duration
  T. To derive the frequency response, represent the averaging operation as a
  causal rectangular kernel a#sub[T] (u), where

  $
    a_T(u) = 1 / T, quad 0 <= u <= T,
  $

  and a#sub[T] (u) is zero outside that interval. Applying this kernel as a
  moving-average filter gives

  $
    delta z_i(t) = integral_(-infinity)^infinity
      a_T(u) delta v_i(t-u) dif u
      = 1 / T integral_0^T delta v_i(t-u) dif u.
  $

  At the end of the presentation, set t equal to T. To put the resulting
  integral into the same form as Equation A10, introduce the new variable

  $
    t' = T - u, quad dif t' = -dif u.
  $

  When u is zero, t' is T; when u is T, t' is zero. The substitution therefore
  reverses the integration limits. Its minus sign reverses them back:

  $
    delta z_i(T) = 1 / T integral_0^T delta v_i(T-u) dif u
      = -1 / T integral_T^0 delta v_i(t') dif t'
      = 1 / T integral_0^T delta v_i(t') dif t'.
  $

  The name of an integration variable has no effect on the value, so replacing
  t' with t recovers Equation A10 exactly.

  For any time-domain function f(u), use the Fourier-transform convention

  $
    hat(f)(omega) = integral_(-infinity)^infinity
      f(u) exp(-i omega u) dif u.
  $

  Here hat(f) (ω) is the Fourier transform of f(u). Applying this definition
  with f(u) equal to the rectangular kernel a#sub[T] (u), and naming the result
  A#sub[T] (ω), gives

  $
    A_T(omega) = integral_(-infinity)^infinity
      a_T(u) exp(-i omega u) dif u
      = 1 / T integral_0^T exp(-i omega u) dif u.
  $

  Evaluating the integral and rearranging gives

  $
    A_T(omega) = (exp(-i omega T) - 1) / (-i omega T)
      = (1 - exp(-i omega T)) / (i omega T).
  $

  At zero frequency, the continuous limit of A#sub[T] (ω) is one, as expected:
  averaging does not change a constant signal. By the convolution theorem, the
  averaging filter multiplies the voltage spectrum:

  $
    delta z_i(omega) = A_T(omega) delta v_i(omega).
  $

  Substitute the input--voltage relation from Equation A9:

  $
    delta z_i(omega) = A_T(omega) G_lambda(omega)
      delta s_i(omega).
  $

  Define the complete local input--feature response by the canonical LTI
  relationship

  $
    delta z_i(omega) = H_lambda(omega) delta s_i(omega).
  $

  Comparing the coefficients multiplying δs#sub[i] (ω) identifies
  H#sub[λ] (ω). The two components of Equation A11 are therefore

  $
    A_T(omega) = (1 - exp(-i omega T)) / (i omega T), quad
    H_lambda(omega) = A_T(omega) G_lambda(omega). quad "(A11)"
  $

  Here A#sub[T] (ω) is the averaging-window response, exp is the exponential
  function, and H#sub[λ] (ω) is the complete response from input spike-train
  perturbation to averaged-voltage perturbation.

  For the Bode comparison, the zero-frequency-normalized magnitude of the
  complete input--feature response is

  $
    B_H(omega) = 20 log_10 (abs(H_lambda(omega)) / abs(H_lambda(0))). quad "(A12)"
  $

  Here B#sub[H] is magnitude in decibels; |·| is complex magnitude; and
  log#sub[10] is the base-ten logarithm. Only H#sub[λ] is plotted because it
  includes the averaging window and is the response used to predict the ANN
  feature variance.

  === A.4 Poisson-noise variance

  A centred, unit-amplitude Poisson point process has a flat two-sided input
  power spectral density:

  $
    S_"in"(omega) = lambda. quad "(A13)"
  $

  Here S#sub[in] (ω) is input power per unit angular frequency and λ is mean spike
  rate. Passing this noise through Equation A11 gives

  $
    S_z(omega) = abs(H_lambda(omega))^2 S_"in"(omega). quad "(A14)"
  $

  Here S#sub[z] (ω) is predicted feature-noise power and
  |H#sub[λ] (ω)|#super[2] is the transmitted fraction of input-noise power. Under
  the two-sided angular-frequency convention, total predicted variance is

  $
    "Var"_"linear"(z) = 1 / (2 pi) integral_(-oo)^oo
      abs(H_lambda(omega))^2 S_"in"(omega) dif omega. quad "(A15)"
  $

  Here Var#sub[linear] (z) is predicted feature variance and π is the circle
  constant; the remaining symbols follow Equations A11--A14. This result is
  exact for the stated linear stationary model, but only approximate for the
  finite, discrete, conductance-dependent probe. Step 3 therefore compares it
  with empirical variance rather than assuming agreement.
]
