#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Filter-matched rate calibration for variable-rate PING training",
  date: "2026-08-05",
  description: "An exploratory continuation tests a linear variance approximation and validates empirical-library MNIST feature images after a preserved response-library gate failure.",
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
#let p = r.step2.combined_pilot
#let p-last = p.trajectory.last()
#let p-extension-first = r.step2.extension_pilot.trajectory.first()
#let mono = r.step2.validations.monotonic_means
#let low-dist = r.step2.representative_distributions.first()
#let boot = r.step2.bootstrap_stability
#let boot-1024 = boot.trajectory.at(4)
#let boot-2048 = boot.trajectory.last()
#let s3 = r.step3
#let s4 = r.step4
#let s3-nominal-low = s3.agreement_summaries.at(3)
#let s3-nominal-transition = s3.agreement_summaries.at(4)
#let s3-nominal-high = s3.agreement_summaries.at(5)
#let s4-nominal-low = s4.condition_records.at(3).comparison
#let s4-nominal-transition = s4.condition_records.at(4).comparison
#let s4-nominal-high = s4.condition_records.at(5).comparison
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

  This staged experiment tests which Poisson rates preserve MNIST class
  evidence. An artificial neural network (ANN) is planned to classify static features
  formed by passing each pixel's spike train through an AMPA synapse and
  non-spiking membrane, then averaging its voltage over the presentation.

  ANN accuracy against rate remains the planned primary result, but no decoder
  was trained here. An explicitly authorized exploratory continuation instead
  tested a linear transfer-function prediction and constructed complete sampled
  MNIST feature images from the preserved empirical library.

  Step 1 is complete. The local generator passed all
  #r.validations.len() focused checks, including agreement with the shared
  `tools/snn` cell below the registered numerical tolerance. The original Step 2
  pilot stopped at K = 512; an explicitly authorized extension then stabilized
  both empirical moments at K = #p.selected_K under the unchanged tolerances.
  The complete empirical library was then generated, but
  #mono.intensity_violations of #mono.intensity_comparison_count adjacent-intensity
  comparisons exceeded the locked Monte Carlo tolerance. Step 2 therefore
  failed its original gate. That failure remains unchanged. Under the later
  amendment, Step 3 found that the stationary linear model overpredicted
  finite-window variance, especially at low drive and larger conductance.
  Step 4 reproduced pooled pixel statistics and recognizable feature structure,
  but failed its locked low-rate image-level checks. Steps 5--7 remain pending.

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
    voltage. We did not divide by rate or disclose it to either decoder, so
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

  2. *Generated and validated the empirical response library.* Before
    generating a final library, we locked candidate draw counts K of 64, 128,
    256, and 512. None passed. After recording that failure, the user explicitly
    authorized an extension at K = 1,024 and 2,048 without changing the
    convergence rule. The deterministic pilot covered
    #p.evaluation_condition_count conditions: six intensities, five rates, all
    three probe conductances, and all three registered seeds. For each candidate,
    we compared the first K draws with the next K independent draws.

    Mean and unbiased-variance discrepancies were divided by locked absolute or
    relative tolerances, whichever was larger. Both metrics had to keep their
    95th percentile at or below 1 and their maximum at or below 2. The smallest
    passing K would have selected the final draw count. If none passed at the
    maximum, the protocol required stopping without changing the rule.

    The extension selected K = #p.selected_K. We then ran that many independent
    Step 1 draws for every grayscale level, registered rate, probe conductance,
    and seed. The retained float32 array has shape
    #m.library_shape.map(str).join(" × ") in the ordered axes
    seed, probe conductance, rate, intensity, and draw. Its payload occupies
    #m.library_payload_bytes bytes in local scratch and is authenticated by the
    SHA-256 digest `#m.library_sha256`.

    The planned final library would estimate the conditional mean

    $ hat(mu)_z (x, r, w_"probe") = 1 / K sum_(k=1)^K z^(k). quad "(8)" $

    and conditional variance

    $ hat(sigma)_z^2 (x, r, w_"probe") = 1 / (K - 1) sum_(k=1)^K (z^(k) - hat(mu)_z)^2. quad "(9)" $

    Here x, r, and w#sub[probe] are pixel-intensity, rate, and probe conductance;
    z#super[(k)] is draw k; K is the draw count; and Equations 8 and 9 estimate
    the conditional mean and variance.

    We retained all K simulated values as the primary empirical response
    library. During a later ANN feature-generation stage, the registered design
    would draw one value according to

    $ J tilde "DiscreteUniform"(1, dots.c, K), quad z_"sample" = z^(J). quad "(10)" $

    Here J is uniform on the K draws and z#sub[sample] is the sampled feature;
    other symbols follow Equations 8 and 9. Mean and variance summarize noise,
    but the ANN samples the retained empirical values because low-rate responses
    may be zero-heavy, skewed, and non-Gaussian.

    The generator wrote #m.chunking.total_chunks deterministic,
    independently authenticated intensity chunks. Exact replay passed for one
    predeclared chunk from every seed. The complete grid, finite values,
    physical bounds, zero-intensity resting response, independent streams,
    independently recomputed moments, fresh direct simulations, and float32
    fidelity all passed their checks. The low-rate representative condition had
    zero fraction #rounded(low-dist.zero_fraction), skewness
    #rounded(low-dist.skewness), and #low-dist.distinct_float32_values distinct
    retained values, preserving its discrete, non-Gaussian structure.

    The required monotonicity check did not pass. It compared every adjacent
    intensity and rate pair using the predeclared
    #(mono.standard_error_multiplier)-standard-error tolerance.
    #mono.intensity_violations of #mono.intensity_comparison_count intensity
    pairs exceeded it, while #mono.rate_violations of
    #mono.rate_comparison_count rate pairs did. The rule was not weakened after
    inspection, so this completed library is a preserved killed attempt rather
    than a validated input to Step 3.

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

    Under a timestamped exploratory amendment, we evaluated every distinct
    calibration point, compared predicted with empirical Step 2 variance, and
    validated low-, middle-, and high-drive gains by sinusoidally modulating the
    numerical probe. The amendment did not relabel the original Step 2 gate.
    This stationary,
    continuous-time, linearized Poisson model approximates the finite, discrete
    Bernoulli simulation and remains diagnostic only.

    The corresponding plot is defined in Results, Step 3.

  4. *Constructed and validated complete feature images.* We used only the
    official 60,000-image MNIST training partition. Indices 0--54,999 and
    55,000--59,999 remain the locked future decoder-training and validation
    partitions; the official test partition was not loaded. For each uint8
    pixel, the sampler used its exact 0--255 intensity index and selected one of
    the K = #p.selected_K authenticated empirical draws with independent,
    deterministic pixel and image streams.

    The direct comparison used training images 0--15, #s4.dataset.image_shape.at(1)
    × #s4.dataset.image_shape.at(2) pixels, eight independent replicates, all
    three conductances, and rates 0.25, 3, and 25 Hz. We compared pooled pixel
    moments, image-level moments, zero fractions, absolute and relative
    differences, and the spatial correlation of per-pixel means. The thresholds
    and streams were locked before outcomes. This validation was required before
    ANN training; its low-rate checks did not all pass, so no decoder followed.

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

    Step 1 is complete. The Step 2 convergence pilot selected K after an
    authorized extension. The full library was generated, then Step 2 was
    killed by its required monotonicity validation. A timestamped post-hoc
    amendment subsequently authorized exploratory Steps 3--4 without changing
    that failure. Step 3 completed; Step 4 stopped at its locked low-rate
    image-level validation. Steps 5--7 remain pending.

    === Step 1: filter-matched feature generation

    #image("/artifacts/data/exp077/probe_dynamics.svg", width: 100%)

    _Finite-window timing sensitivity._ Panel A compares the subthreshold
    response to one early and one late spike. Panel B places that single spike
    across the presentation and reports the resulting mean voltage feature.
    With the same one-spike count, a spike at
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

  #image(
    "/artifacts/data/exp077/response_library.png",
    width: 100%,
    alt: "Six panels show response mean, variability, zero probability, empirical distributions, convergence, and the monotonicity audit against expected input spikes.",
  )

  _Signal and variability across the complete empirical response library._
  Expected input spikes are encoding rate times normalized grayscale intensity
  times the #presentation-ms ms presentation. Panel A shows conditional mean
  feature z; Panel B shows its standard deviation in the same voltage units.
  Faint points are all rate--intensity conditions and strong curves are binned
  medians. Colour, marker, and line style identify probe conductance. The red
  crosses locate the #mono.intensity_violations isolated comparisons that failed
  the locked monotonicity rule. The small annotation reports how exact-zero mass
  falls across the registered full-intensity range; detailed distribution,
  replay, and K-stability audits remain in the recorded artifacts.

  The array contains #m.library_value_count float32 values with shape
  #m.library_shape.map(str).join(" × ") and a #(m.library_payload_bytes)-byte
  payload. Zero intensity returned the resting feature exactly, direct
  simulation agreed at every predeclared condition, and exact chunk replay
  passed for all registered seeds. These checks establish an authenticated
  empirical library and its failure mode only. They do not establish MNIST
  decodability, a training-rate floor, or PING accuracy.

  #block(breakable: false)[
    The post-hoc repeated-resampling diagnostic found that K =
    #boot-1024.K passed #boot-1024.pass_count of #boot.repetitions repetitions,
    whereas K = #boot-2048.K passed #boot-2048.pass_count, a frequency of
    #rounded(boot-2048.pass_frequency). This exceeded the locked escalation
    threshold, so no independent K = 4,096 comparison was run. The diagnostic
    supports describing K = #boot-2048.K as typically stable under the registered
    tolerances, not as an asymptotic plateau.
  ]

  === Step 3: linear-filter prediction

  #image(
    "/artifacts/data/exp077/linear_filter.svg",
    width: 100%,
    alt: "Two Bode magnitude panels show the synapse-plus-membrane response and the complete response after 200 millisecond averaging at low, transitional, and high drive.",
  )

  _Linearized frequency response at the nominal 1.2 μS probe._ Panel A shows
  the synapse-plus-membrane transfer magnitude for low, transitional, and high
  drive. Panel B applies the #presentation-ms ms finite-window averaging
  response to the same curves. Gain is expressed relative to the low-drive DC
  response, so vertical separation shows how mean conductance changes local
  gain while the additional roll-off and spectral nulls show what window
  averaging removes. The dotted horizontal line marks −3 dB.

  All #s3.gain_checks.len() numerical sinusoidal gain checks passed; the largest
  relative error was
  #rounded(calc.max(..s3.gain_checks.map(x => x.relative_error)), digits: 4).
  The maximum frequency-grid refinement change was
  #rounded(s3.quadrature.maximum_refinement_relative_change, digits: 8), and the
  widened-bound change was
  #rounded(s3.quadrature.maximum_bound_relative_change, digits: 8), both below
  the locked 0.2% tolerance. At the nominal 1.2 μS probe, median predicted to
  empirical variance ratios were
  #rounded(s3-nominal-low.median_predicted_empirical_ratio) at low drive,
  #rounded(s3-nominal-transition.median_predicted_empirical_ratio) at
  transitional drive, and #rounded(s3-nominal-high.median_predicted_empirical_ratio)
  at high drive. Thus the transfer functions and numerical implementation agree,
  but the stationary variance approximation remains quantitatively poor and is
  not a substitute for the empirical library.

  === Step 4: complete feature images

  #image(
    "/artifacts/data/exp077/feature_images.png",
    width: 100%,
    alt: "Rows at low, transitional, and high rates compare original MNIST images, empirical-library samples, fresh direct simulations, and signed differences.",
  )

  _Empirical-library samples versus fresh direct Step 1 simulations._ The image
  rows show 0.25, 3, and 25 Hz at the nominal 1.2 μS probe; comparable voltage
  images use identical 0--65 mV limits. The low-rate row retains the registered
  zero-heavy, discrete structure, while transitional and high rates preserve
  recognizable digit structure. The summary beneath reports spatial
  library--direct agreement for all three conductances. Filled markers passed
  every locked check; open red markers failed at least one. The dashed line is
  the rate-specific spatial-correlation minimum. The displayed samples are
  illustrative; validation used all three conductances, 16 fixed images, and
  eight replicates per condition.

  Six of nine probe--rate conditions passed every locked comparison. At the
  nominal probe, pooled-mean relative differences were
  #rounded(s4-nominal-low.metrics.pooled_mean_relative_difference),
  #rounded(s4-nominal-transition.metrics.pooled_mean_relative_difference), and
  #rounded(s4-nominal-high.metrics.pooled_mean_relative_difference) from low to
  high drive; spatial correlations were
  #rounded(s4-nominal-low.metrics.spatial_mean_correlation),
  #rounded(s4-nominal-transition.metrics.spatial_mean_correlation), and
  #rounded(s4-nominal-high.metrics.spatial_mean_correlation).
  All pooled pixel-moment and zero-fraction checks passed. The three low-rate
  conditions failed image-level mean and variance tolerances; the 2.4 μS
  low-rate condition also reached correlation
  #rounded(s4.condition_records.at(6).comparison.metrics.spatial_mean_correlation)
  against the locked 0.20 minimum. The thresholds and replicate count were not
  changed after inspection. Step 4 is therefore a preserved validation failure,
  and it establishes neither decodability nor classification accuracy.

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
