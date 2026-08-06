#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Filter-matched rate calibration for variable-rate PING training",
  date: "2026-08-05",
  description: "Empirical and analytical calibration of Poisson-driven, filter-matched MNIST features for conductance-based PING inputs.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let presentation-ms = 200
#let dt-ms = 0.1
#let training-rates-hz = (0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 25)
#let probe-us = (0.6, 1.2, 2.4)
#let seeds = (42, 43, 44)
#let r = json("/artifacts/data/exp077/numbers.json")
#let m = json("/artifacts/data/exp077/step2_manifest.json")
#let diagnostic-k = m.diagnostic_draws_per_condition_per_seed
#let s3 = r.step3
#let s3-comparison = json("/artifacts/data/exp077/step3_empirical_comparison.json")
#let s4 = r.step4
#let s5 = json("/artifacts/data/exp077/step5_outcome.json")
#let s6 = json("/artifacts/data/exp077/step6_outcome.json")
#let decision = json("/artifacts/data/exp077/decision.json")
#let s3-nominal-low = s3.agreement_summaries.at(3)
#let s3-nominal-transition = s3.agreement_summaries.at(4)
#let s3-nominal-high = s3.agreement_summaries.at(5)
#let s4-nominal-low = s4.condition_records.at(3).comparison
#let s4-nominal-transition = s4.condition_records.at(4).comparison
#let s4-nominal-high = s4.condition_records.at(5).comparison
#let nominal-quarter = s6.nonlinear.at(12)
#let nominal-half = s6.nonlinear.at(13)
#let rounded(x, digits: 3) = str(calc.round(x, digits: digits))
#let pct(x) = rounded(100 * x, digits: 2) + "%"
#let training-rate-text = training-rates-hz.map(str).join(", ")
#let probe-text = probe-us.map(str).join(", ")
#let seed-text = seeds.map(str).join(", ")
#let body = [
  == Abstract

  Selecting a Poisson input rate for conductance-based spiking networks requires
  accounting for the filtering and variability introduced by synapses,
  membrane integration, and finite observation windows. We propagated MNIST
  pixel spike trains through the target AMPA and subthreshold membrane dynamics,
  characterized their empirical responses, and derived a corresponding linear
  filter model. Nonlinear and linear decoders were then trained on fresh direct
  simulations across the registered rate grid. Held-out nonlinear-decoder
  accuracy was reliably above chance at 0.25 Hz and reliably exceeded 50% from
  0.5 Hz. This practical floor was unchanged across 0.6, 1.2, and 2.4 μS probes,
  supporting 0.5--25 Hz for subsequent variable-rate PING training.

  == Purpose and scope

  The question is:

  #quote(block: true)[At a given encoding rate, does the temporally filtered
    image still contain enough information for an ANN to recognize the digit?]

  The target PING architecture has 784 pixel channels feeding conductance-based
  excitatory cells. Its fixed synaptic and membrane equations define the
  features, without trained input, recurrent, or output weights. The ANN thus
  measures filtered-input decodability independently of a trained PING network.

  We characterized these features empirically and analytically, trained matched
  nonlinear and linear decoders on fresh simulations, and evaluated frozen
  models on held-out images. The resulting thresholds measure decoder-relative
  accessibility, not an absolute information boundary or PING accuracy.

  == Methods

  1. *Generated and validated filter-matched pixel features.* Every image used
    a #presentation-ms ms presentation and #dt-ms ms timestep. The tested rates
    were #training-rate-text Hz, sampled densely below 5 Hz and capped at 25 Hz,
    a range previously effective in trained PING networks. Seeds #seed-text
    defined independent feature and empirical-response runs.

    The rate grid and deterministic MNIST training and validation indices were
    recorded before analysis; the official MNIST test set was held out and not
    loaded. Pixel intensities x lay in [0, 1]. If r was
    the encoding rate at pixel intensity one, the expected pixel rate was r x.

    Each pixel drove an independent AMPA synapse and uncoupled non-spiking
    excitatory membrane. This captured synaptic decay, conductance-dependent
    integration, and within-window timing, but included no threshold, reset,
    recurrence, or trained weights.

    The complete feature path is

    $ S_i (t) -> g_i^"pix" (t) -> v_i (t) -> z_i -> "ANN". quad "(1)" $

    Here S#sub[i], g#super[pix]#sub[i], v#sub[i], and z#sub[i] are pixel i's spike
    train, AMPA conductance, subthreshold voltage, and time-averaged feature;
    t is time and ANN is the classifier.

    At each timestep, we drew the encoded input according to

    $ S_i (t) tilde "Bernoulli"(r Delta t x_i). quad "(2)" $

    Here x#sub[i] is grayscale pixel-intensity, r is the encoding rate in spikes
    per second at pixel-intensity one, Δt is the timestep in seconds, and
    Bernoulli is an independent binary draw.

    We updated the AMPA conductance using

    $ g_i^"pix" (t) = beta_"AMPA" g_i^"pix" (t-1) + w_"probe" S_i (t). quad "(3)" $

    $ beta_"AMPA" = exp(-(Delta t) / tau_"AMPA"). quad "(4)" $

    Here β#sub[AMPA] is one-timestep conductance retention, τ#sub[AMPA] is AMPA
    decay time, w#sub[probe] is conductance added per spike, and exp is the
    exponential function. Other symbols follow Equation 2. The primary 1.2 μS
    probe is the target architecture's nominal mean initial pixel-to-excitatory
    conductance; 0.6 and 2.4 μS test half and double scale for comparison.

    We integrated the membrane without thresholding or resetting:

    $ C_E (d v_i) / (d t) = g_"L,E" (E_L - v_i) + g_i^"pix" (t) (E_e - v_i). quad "(5)" $

    $ tau_"eff",i (t) = C_E / (g_"L,E" + g_i^"pix" (t)). quad "(6)" $

    Here C#sub[E] is excitatory capacitance; g#sub[L,E] is leak conductance;
    E#sub[L] and E#sub[e] are leak and AMPA reversal potentials; and
    τ#sub[eff,i] is the instantaneous conductance-dependent time constant. Other
    symbols follow Equation 1.

    Voltage began at E#sub[L] and conductance at zero. After one presentation,

    $ z_i = 1 / T integral_0^T (v_i (t) - E_L) dif t. quad "(7)" $

    where T is presentation duration and z#sub[i] is mean baseline-subtracted
    voltage. Neither the feature nor the decoder input included rate, so sparse
    inputs retained their weaker, noisier signal.

  2. *Characterized the empirical response moments.* For visualization and
    consistency checks, we used #diagnostic-k independent simulations per
    rate--intensity--conductance condition and seed. This was a practical
    diagnostic sample size, not a convergence gate or a requirement for ANN
    training. The empirical response table estimated the conditional mean

    $ hat(mu)_z (x, r, w_"probe") = 1 / K sum_(k=1)^K z^(k). quad "(8)" $

    and conditional variance

    $ hat(sigma)_z^2 (x, r, w_"probe") = 1 / (K - 1) sum_(k=1)^K (z^(k) - hat(mu)_z)^2. quad "(9)" $

    Here x, r, and w#sub[probe] are pixel-intensity, rate, and probe conductance;
    z#super[(k)] is draw k; K is the draw count; and Equations 8 and 9 estimate
    the conditional mean and variance.

    The table served only to plot moments and check consistency. Decoder inputs
    instead used a fresh spike train and direct synapse--membrane simulation on
    every presentation n:

    $ S_i^(n)(t) -> g_i^"pix,n"(t) -> v_i^(n)(t) -> z_i^(n) -> "ANN". quad "(10)" $

    Repetition therefore sampled the full conditional distribution of z#sub[i]
    without an intermediate noise model.

    Exact replay, finite values, physical bounds, zero-intensity resting
    response, independent streams, recomputed moments, fresh direct simulations,
    and float32 fidelity were checked.

  3. *Derived the analytical linear-filter response.* We calculated a local
    linear approximation to the synapse, membrane, and finite averaging window.
    This diagnostic was not used to select a training range.

    For mean spike rate λ, stationary mean conductance is

    $ bar(g)_lambda = lambda w_"probe" tau_"AMPA". quad "(11)" $

    and stationary mean voltage is

    $
      bar(v)_lambda = (g_"L,E" E_L + bar(g)_lambda E_e) /
      (g_"L,E" + bar(g)_lambda). quad "(12)"
    $

    Here λ is encoding rate times pixel-intensity; bar(g)#sub[λ] and
    bar(v)#sub[λ] are stationary mean conductance and voltage. Other symbols
    follow Equations 3--7.

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

    We evaluated the transfer functions at every distinct calibration point and
    validated low-, middle-, and high-drive gains by sinusoidally modulating the
    numerical probe. This stationary,
    continuous-time, linearized Poisson model approximates the finite, discrete
    Bernoulli simulation and remains diagnostic only.

    Without new simulation, we compared these predictions with the Step 2 table
    at every rate, intensity, and conductance. Analytical mean was Equation 12
    minus resting voltage; analytical SD was the square root of Equation 18.
    Predicted-versus-empirical plots used an identity line, Pearson correlation,
    mean absolute error, and median prediction/empirical ratio. Summary
    statistics excluded conditions where both responses were zero.

  4. *Constructed and compared complete feature images.* We used only the
    official 60,000-image MNIST training partition. Indices 0--54,999 and
    55,000--59,999 were reserved for decoder training and validation,
    respectively; the official MNIST test set was held out and not loaded. For each uint8
    pixel, the diagnostic sampler used its exact 0--255 intensity index and
    selected an authenticated empirical draw with independent, deterministic
    pixel and image streams. A fresh direct simulation provided the comparison.

    The comparison covered training images 0--15, all
    #s4.dataset.image_shape.at(1) × #s4.dataset.image_shape.at(2) pixels, eight
    replicates, three conductances, and 0.25, 3, and 25 Hz. Tests covered pooled
    and image-level moments, zero fractions, absolute and relative differences,
    and spatial correlation of per-pixel means. The low-rate image-level checks
    failed; this limitation was retained, while subsequent decoders used fresh
    direct simulation rather than the table.

  5. *Trained mixed-rate decoders.* Each simulated image produced a feature
    vector $bold(z) in RR^784$, containing one time-averaged membrane-voltage
    displacement per pixel. The primary ANN transformed this vector through a
    fully connected 1,024-unit hidden layer,

    $
      bold(h) = "ReLU"(W_h bold(z) + bold(b)_h), quad
      bold(s) = W_o bold(h) + bold(b)_o,
    $

    where $bold(h)$ is hidden activation, $bold(s) in RR^10$ contains one logit
    per digit, and elementwise $"ReLU"(x) = max(0, x)$ retains positive inputs
    and zeros negative ones. Both weight matrices and bias vectors were
    trainable; prediction used $arg max_c s_c$. The 784--1,024--10 network had
    no convolution, recurrence, normalization, dropout, or pretrained weights.

    The matched linear decoder omitted the hidden layer and computed

    $ bold(s) = W_"lin" bold(z) + bold(b)_"lin". $

    Its 784 × 10 weights and ten biases were trainable. “Linear” described the
    feature-to-logit map, not fitting: both models used backpropagation. Linear
    performance measured directly accessible digit evidence; the ANN advantage
    measured the benefit of its nonlinear representation. Neither received rate.

    Both models were fitted with ten-class cross-entropy. For logits
    $bold(s)$ and true digit label $y$, the loss for one image was

    $
      cal(L)_"CE"(bold(s), y)
      = -log(exp(s_y) / sum_(c=0)^9 exp(s_c)).
    $

    This penalized low probability for the correct digit; backpropagation trained
    each decoder's weights and biases.

    Each presentation sampled the registered rates uniformly, then generated
    z#sub[i] from a fresh spike train through Equations 3--7. Both decoders saw
    the same feature batch. Adam used learning rate 0.001, batch size 256, and 15
    epochs. The ANN used no weight decay; three linear candidates used L2 weight
    decays 10#super[-5], 10#super[-4], and 10#super[-3]; validation accuracy
    selected the decay and, independently for each decoder, the retained epoch.
    Seeds #seed-text initialized parameters and sampling streams. The MNIST test
    set remained held out through fitting and selection. The primary 1.2 μS
    ensemble and separate 0.6 and 2.4 μS ensembles were not pooled.

  6. *Evaluated frozen held-out psychometric curves.* All decoders, selected
    epochs, hyperparameters, checkpoint hashes, rates, direct-simulation seeds,
    and uncertainty rules were committed before the held-out MNIST test set was
    accessed. At each rate, every decoder seed received the same 10,000 held-out
    images and three reproducible fresh direct-simulation draws. The primary
    curve was

    $ A_r (r) = P("correct" | r, "mixed-rate nonlinear decoder"). quad "(19)" $

    Here A#sub[r] is held-out nonlinear-ANN accuracy, P is correct-classification
    probability, and r is rate. The linear decoder was diagnostic.

    Two thousand bootstrap repetitions resampled images, simulation draws, and
    decoder seeds to give the one-sided 95% lower bound L#sub[r]. The
    decoder-relative edge was

    $ r_"decode" = "lowest " r in cal(R) " satisfying " L_r (r) > 1 / N_"class". quad "(20)" $

    The practical training floor was

    $ r_"train" = "lowest " r in cal(R) " satisfying " L_r (r) >= a_"use". quad "(21)" $

    Here $cal(R)$ is the tested grid, N#sub[class] the class count,
    r#sub[decode] the lowest rate reliably above chance, a#sub[use] the 50%
    useful-accuracy target, and r#sub[train] its first reliable crossing. Primary
    thresholds used the 1.2 μS ANN; the linear decoder and other conductances
    measured sensitivity. Interpolated rates were not observations.

  7. *Selected the training range.* The decision reported r#sub[decode],
    r#sub[train], uncertainty, conductance sensitivity, model and artifact
    hashes, and every registered rule outcome. The later PING range began at
    r#sub[train] and ended at the registered 25 Hz ceiling. The rule would have
    reported a range of plausible floors if conductance shifted the practical
    floor by more than one adjacent grid point.

  #block(breakable: false)[
    == Results

    === Filter-matched feature generation

    #image("/artifacts/data/exp077/probe_dynamics.svg", width: 100%)

    _Finite-window timing sensitivity._ Panel A places one input spike at 20 or
    180 ms. Panels B and C pass those inputs through the registered AMPA synapse
    and subthreshold membrane. Panel D repeats the simulation across 101
    single-spike times and plots the presentation-averaged feature z. Conductance
    and voltage rise after each spike and then relax through their respective
    decay dynamics; z falls for late spikes because the fixed 200 ms window
    truncates more of their response.
  ]

  === Empirical response table

  #image(
    "/artifacts/data/exp077/response_library.png",
    width: 100%,
    alt: "Two panels show empirical feature mean and standard deviation against expected input spikes for three probe conductances.",
  )

  _Signal and variability across the empirical response table._ For every
  rate--intensity condition, Panel A plots the mean feature z and Panel B its
  standard deviation from #diagnostic-k independent simulations per condition
  and seed, against encoding rate × normalized pixel intensity ×
  #presentation-ms ms. Each point is one rate--intensity condition, and colour
  encodes probe conductance. Both moments rise because larger expected spike
  counts deliver more stochastic conductance; larger probes produce larger
  voltage excursions.

  === Analytical linear-filter response

  #image(
    "/artifacts/data/exp077/linear_filter.svg",
    width: 100%,
    alt: "Two analytically calculated Bode magnitude panels show the synapse-plus-membrane response and the complete response after 200 millisecond averaging at three mean input rates.",
  )

  _Analytically calculated linearized frequency response at the nominal 1.2 μS
  probe._ These analytical functions take modulation frequency, mean input rate,
  and probe conductance; they contain no simulated traces. Frequency is a small
  sinusoidal rate modulation around 0.25, 3, or 25 Hz. Panel A evaluates the
  magnitude of Equation 13 from 0.1--200 Hz, with $omega = 2 pi f$. Gain is

  $
    20 log_10 (abs(G_lambda(2 pi f)) / abs(G_(lambda_"low")(0))),
  $

  relative to the 0.25 Hz DC response. Thus the black curve begins near 0 dB;
  higher mean conductance lowers the red and cyan DC gains. Panel B plots
  Equation 15, $H_lambda = A_T G_lambda$, after the analytical #presentation-ms
  ms rectangular-window average.

  Panel A combines an AMPA low-pass pole at $1 / tau_"AMPA"$ with a membrane pole
  set by effective conductance and capacitance, smoothly suppressing fast
  modulation. Greater drive both shunts the membrane and depolarizes it toward
  $E_e$, reducing excitatory driving force and separating the curves.

  Panel B adds the averaging magnitude
  $abs(A_T(2 pi f)) = abs(sin(pi f T) / (pi f T))$. It is one at zero frequency
  and zero at $f = n / T$ for nonzero integer n. With $T =
  #presentation-ms / 1000$ s, integer cycles cancel at 5, 10, 15 Hz, and so on.
  Magnitude folds alternating sinc signs into positive lobes; the two low-pass
  terms attenuate successive lobes to form the falling envelope.

  #image(
    "/artifacts/data/exp077/linear_filter_empirical_comparison.svg",
    width: 100%,
    alt: "Two predicted-versus-empirical plots compare analytical and simulated feature means and standard deviations at three probe conductances.",
  )

  _Stationary analytical predictions versus the empirical response table._ Each
  point is one recorded rate--intensity condition; colour denotes probe
  conductance. Panel A compares the stationary mean displacement from Equation
  12 with the empirical 200 ms feature mean. Panel B compares the square root of
  the Equation 18 variance with empirical feature SD. The diagonal denotes exact
  agreement. The analytical model preserved the broad ordering of the mean
  responses (Pearson $r = #rounded(s3-comparison.mean.pearson_r)$) but predicted
  a median #rounded(s3-comparison.mean.median_predicted_empirical_ratio)-fold
  larger mean. SD agreement was weaker ($r =
  #rounded(s3-comparison.standard_deviation.pearson_r)$), with a median
  #rounded(s3-comparison.standard_deviation.median_predicted_empirical_ratio)-fold
  overprediction. Points therefore lie mainly below the identity line because
  the stationary, locally linear model omits the start-from-rest transient,
  finite-window nonstationarity, discrete Bernoulli input, and nonlinear
  conductance fluctuations present in the empirical simulations.

  Panel A bends upward because Equation 12 evaluates voltage at stationary mean
  conductance, whereas empirical trajectories begin at rest and average only
  #presentation-ms ms. Sparse presentations often contain no event or a late
  one; with greater drive, earlier and more consistent events let the empirical
  mean approach stationarity. Probe curves separate because a mean conductance
  can comprise many small events or fewer large ones. Larger jumps amplify
  fluctuations, and voltage saturation gives approximately $E[v(g)] < v(E[g])$.

  Panel B hooks because variability rises from zero as spike counts and timings
  diversify, then falls when averaging suppresses relative count fluctuations
  and shunting and saturation limit voltage excursions. Equation 18 likewise
  balances rising Poisson noise against falling local gain, but places the
  maximum elsewhere. Plotting these displaced maxima against each other creates
  the arches and backward branches; larger, less linear probe events enlarge
  them.

  === Complete feature images

  #image(
    "/artifacts/data/exp077/feature_images.png",
    width: 100%,
    alt: "Rows at 0.25, 3, and 25 hertz compare original MNIST images, empirical response table samples, and fresh direct simulations.",
  )

  _Empirical response table samples versus fresh direct simulations._ Rows show
  one original MNIST image, one authenticated empirical response table sample, and one fresh
  direct simulation at 0.25, 3, and 25 Hz for the nominal 1.2 μS probe, using
  common 0--65 mV limits. Low-rate images are sparse because most pixels receive
  no spike; increasing rate reveals the intensity pattern because spatial signal
  becomes larger relative to independent sampling noise.

  === Mixed-rate decoder training

  #image(
    "/artifacts/data/exp077/step5_training_history.svg",
    width: 100%,
    alt: "Validation accuracy across fifteen epochs for nonlinear and linear decoders at three probe conductances.",
  )

  _Mixed-rate validation histories._ Each panel plots mean validation accuracy
  across decoder seeds, with the seed range shaded, for models trained from
  fresh direct simulations at uniformly sampled registered rates. Curves rise
  rapidly and then flatten because most learnable class structure is acquired
  in the first several epochs; the three conductances overlap because scaling
  the subthreshold response preserves nearly the same spatial evidence.

  At the nominal probe, validation selected nonlinear epochs
  #s5.records.at(3).selected_nonlinear_epoch,
  #s5.records.at(4).selected_nonlinear_epoch, and
  #s5.records.at(5).selected_nonlinear_epoch, with accuracies
  #pct(s5.records.at(3).selected_nonlinear_validation_accuracy),
  #pct(s5.records.at(4).selected_nonlinear_validation_accuracy), and
  #pct(s5.records.at(5).selected_nonlinear_validation_accuracy). The matched
  linear decoders reached
  #pct(s5.records.at(3).selected_linear_validation_accuracy),
  #pct(s5.records.at(4).selected_linear_validation_accuracy), and
  #pct(s5.records.at(5).selected_linear_validation_accuracy).

  === ANN psychometric curve and thresholds

  #image(
    "/artifacts/data/exp077/psychometric.svg",
    width: 100%,
    alt: "Held-out accuracy against encoding rate, comparing nonlinear and linear decoders and three probe conductances with uncertainty bands.",
  )

  _Frozen held-out psychometrics._ Curves show accuracy on the same 10,000
  held-out MNIST images from three direct-simulation draws and three decoder
  seeds; bands show the registered hierarchical 95% interval. The dotted and
  dashed horizontal rules mark 10% chance and 50% useful accuracy. Accuracy
  rises with rate because more pixels receive spikes within 200 ms, while the
  near-overlap across probe conductances shows that the trained nonlinear
  decoder compensates for their response-scale difference.

  At 0.25 Hz, nominal nonlinear accuracy was #pct(nominal-quarter.accuracy),
  with a #pct(nominal-quarter.lower_95_one_sided) one-sided lower bound. At 0.5
  Hz it was #pct(nominal-half.accuracy), with a
  #pct(nominal-half.lower_95_one_sided) lower bound. The nominal linear decoder
  also first crossed the 50% lower-bound criterion at
  #s6.thresholds.at("linear_1.2").r_train_hz Hz.

  === Training-range decision

  The nonlinear decoder was reliably above chance from
  #decision.r_decode_hz Hz, and its practical floor was
  #decision.r_train_hz Hz. The practical floor remained
  #decision.conductance_floors_hz.at("0.6") Hz at 0.6 μS,
  #decision.conductance_floors_hz.at("1.2") Hz at 1.2 μS, and
  #decision.conductance_floors_hz.at("2.4") Hz at 2.4 μS. We therefore
  recommend rates from #decision.recommendation.floor_hz Hz to
  #decision.recommendation.ceiling_hz Hz for later variable-rate PING training.
  This is a decoder-relative practical
  range, not an absolute information limit or a PING accuracy result.

  == Relation to prior work

  Filtered conductance shot noise can violate fixed-time-constant models#cite(1),
  and Gaussian approximations can miss skewed responses#cite(2), supporting the
  empirical response table. Decoder performance is not absolute neural
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
  finite, discrete, conductance-dependent probe. It is retained as an analytical
  diagnostic rather than as the generator of ANN inputs.
]
