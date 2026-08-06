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
#let body = [
  == Abstract

  Poisson rate coding provides a simple interface between static images and
  spiking neural networks, but the encoding rate does not by itself determine
  the signal available to a downstream circuit. Synaptic decay,
  conductance-dependent membrane integration, and finite observation windows
  reshape both the strength and variability of the representation. This makes
  input-rate selection especially important for sparse conductance-based PING
  networks, where low rate regimes that are computationally efficient may also erase class
  structure before learning begins. We developed a filter-matched calibration
  of MNIST inputs by propagating pixel spike trains through the target AMPA and
  subthreshold membrane dynamics, measuring the resulting empirical response
  distributions, and deriving the corresponding linear transfer-function model.
  The empirical response table preserved non-Gaussian finite-window effects,
  while the analytical model described the frequency filtering imposed by the
  synapse, membrane, and averaging window. Sampled feature
  images recovered recognizable digit structure as rate increased, whereas the
  lowest-rate images remained sparse and did not satisfy all image-level
  validation tolerances. These results established an empirical basis for
  selecting candidate input rates before decoder or recurrent-network training;
  classification thresholds remain to be measured.

  == Purpose and scope

  The question is:

  #quote(block: true)[At a given encoding rate, does the temporally filtered
    image still contain enough information for an ANN to recognize the digit?]

  The target PING architecture has 784 pixel channels feeding conductance-based
  excitatory cells. Its fixed synaptic and membrane equations define the
  features, without trained input, recurrent, or output weights. The ANN thus
  measures filtered-input decodability independently of a trained PING network.

  The study was organized to:

  #enum(
    [Convert 0.25--25 Hz Poisson inputs into static, filter-matched features.],
    [Measure their distributions and test a linear variance prediction.],
    [Construct complete MNIST feature images from the empirical response table.],
    [Keep the official MNIST test set held out for subsequent decoder evaluation.],
  )

  #list(
    [Nonlinear ANN above chance: this decoder can access class evidence.],
    [Nonlinear ANN at chance: this representation and decoder cannot extract
      reliable evidence.],
    [Nonlinear ANN succeeds but linear decoder fails: the evidence is not
      linearly accessible.],
  )

  The completed analyses characterized the input representation rather than
  classification performance. Any later ANN threshold would be a
  decoder-relative decodability edge, not an absolute information boundary or a
  prediction of PING accuracy.

  == Methods

  1. *Generated and validated filter-matched pixel features.* Every image used
    a #presentation-ms ms presentation and #dt-ms ms timestep. The tested rates
    were #training-rate-text Hz, with dense sampling below 5 Hz and an upper
    bound of 25 Hz, which has previously been show to have good performance in trained PING networks. Seeds #seed-text defined independent feature and empirical
    response table runs.

    The rate grid and deterministic MNIST training and validation indices were
    recorded before analysis; the official MNIST test set was held out and not
    loaded. Pixel intensities x lay in [0, 1]. If r was
    the encoding rate at pixel intensity one, the expected pixel rate was r x.

    Each pixel drove an independent AMPA synapse and uncoupled, non-spiking
    excitatory membrane, capturing AMPA decay, conductance-dependent integration,
    and within-window timing that raw Poisson encoded spike counts would omit. This probe has no threshold,
    reset, recurrence, or trained weights.

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

    We initialized voltage at E#sub[L] and conductance at zero, simulated one
    complete presentation, and calculated

    $ z_i = 1 / T integral_0^T (v_i (t) - E_L) dif t. quad "(7)" $

    Here T is presentation duration and z#sub[i] is mean baseline-subtracted
    voltage. The feature definition did not divide by rate, and future decoder
    inputs will not include rate, so lower rates retain their weaker, noisier
    signal.

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

    The table was used only to plot these moments and support consistency checks.
    It was not used as a sampling distribution for ANN
    inputs. For every future image presentation n, the ANN input will instead be
    generated directly by drawing a fresh Poisson spike train and rerunning the
    synapse and membrane equations:

    $ S_i^(n)(t) -> g_i^"pix,n"(t) -> v_i^(n)(t) -> z_i^(n) -> "ANN". quad "(10)" $

    Repeated presentations therefore sample the full conditional distribution
    of z#sub[i] without fitting or resampling an intermediate noise model.

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

  4. *Constructed and compared complete feature images.* We used only the
    official 60,000-image MNIST training partition. Indices 0--54,999 and
    55,000--59,999 were reserved for decoder training and validation,
    respectively; the official MNIST test set was held out and not loaded. For each uint8
    pixel, the diagnostic sampler used its exact 0--255 intensity index and
    selected an authenticated empirical draw with independent, deterministic
    pixel and image streams. A fresh direct simulation provided the comparison.

    The direct comparison used training images 0--15, #s4.dataset.image_shape.at(1)
    × #s4.dataset.image_shape.at(2) pixels, eight independent replicates, all
    three conductances, and rates 0.25, 3, and 25 Hz. We compared pooled pixel
    moments, image-level moments, zero fractions, absolute and relative
    differences, and the spatial correlation of per-pixel means. The thresholds
    and streams were locked before outcomes. This validation was required before
    ANN training; its low-rate checks did not all pass, so no decoder followed.

  5. *Planned mixed-rate decoder training.* The primary ANN will have 784
    inputs, one 1,024-unit rectified-linear hidden layer, and ten outputs, and
    will learn both weight matrices from voltage features. A regularized linear
    softmax decoder trained on the same features will test linear accessibility.
    Neither model will receive the encoding rate or pretrained weights.

    Every image presentation will sample a rate uniformly from the registered
    grid. Each pixel will then receive a fresh Poisson spike train, which will be
    propagated directly through Equations 3--7 to produce z#sub[i]. The empirical
    response table will not generate decoder inputs. Seeds #seed-text will use
    Adam, learning rate 0.001, batch size 256, and initially at most 15 epochs.
    Validation alone will control model selection, regularization, early
    stopping, and any epoch extension. The official MNIST test set will remain
    held out until Method 6. Each configuration, selected epoch, and training
    history will be recorded.

    The primary ensemble will use the 1.2 μS probe; separate 0.6 and 2.4 μS runs
    will test conductance sensitivity. Conductance conditions will not be pooled
    or supplied to either decoder.

  6. *Planned inference-only psychometric evaluation.* All decoders will be
    frozen before the held-out MNIST test set is accessed. At each tested rate,
    decoder seeds will use the same held-out images and reproducible fresh
    direct-simulation draws; additional direct draws will measure encoding
    variability. The primary curve will be

    $ A_r (r) = P("correct" | r, "mixed-rate nonlinear decoder"). quad "(19)" $

    Here A#sub[r] is held-out nonlinear-ANN accuracy, P is correct-classification
    probability, and r is rate. The linear decoder will remain diagnostic.

    Bootstrap resampling of held-out images, direct-simulation draws, and decoder
    seeds will give lower confidence bound L#sub[r]. The decoder-relative edge
    will be

    $ r_"decode" = "lowest " r in cal(R) " satisfying " L_r (r) > 1 / N_"class". quad "(20)" $

    The practical training floor will be

    $ r_"train" = "lowest " r in cal(R) " satisfying " L_r (r) >= a_"use". quad "(21)" $

    Here $cal(R)$ is the tested grid; N#sub[class] is the number of classes;
    r#sub[decode] is the lowest rate reliably above chance; a#sub[use] is the 50%
    useful-accuracy target; and r#sub[train] is the lowest rate whose lower
    confidence bound reaches it. Primary thresholds will use the 1.2 μS
    nonlinear ANN. The linear decoder and 0.6 and 2.4 μS runs will show
    sensitivity to decoder capacity and conductance. Interpolated rates will not
    be treated as observations.

  7. *Planned training-range decision.* The final `decision.json` will report
    r#sub[decode], r#sub[train], their uncertainty, probe-conductance sensitivity,
    decoder and artifact hashes, and all rule outcomes. The recommended later
    PING range will be r#sub[train]--25 Hz. If the floor shifts by more than one
    adjacent grid point across conductances, the result will report plausible
    floors rather than one value. A separate experiment will train and test PING
    over the resulting range.

  #block(breakable: false)[
    == Results

    We characterized the filter-matched pixel response, empirical
    response table, linear approximation, and complete feature images. The
    response table was used only for characterization and consistency checks;
    the complete feature images did not meet all low-rate image-level
    tolerances. Decoder
    training, psychometric evaluation, and a training-range decision were not
    performed and remain incomplete.

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
  probe._ The plotted functions take modulation frequency, mean input spike rate,
  and probe conductance as inputs; no simulated traces are plotted. Frequency
  here is the frequency of a small sinusoidal modulation of the input spike rate,
  about mean operating rates of 0.25, 3, or 25 Hz. Panel A evaluates the
  synapse-plus-membrane transfer function on a 0.1--200 Hz frequency grid. Panel
  B multiplies it by the analytical #presentation-ms ms rectangular-window
  averaging response. Gain is referenced to the low-drive DC value. Mean
  conductance lowers and shifts the membrane response, the AMPA and membrane
  terms produce the smooth low-pass roll-off, and rectangular averaging
  introduces nulls at multiples of 5 Hz.

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

  _Not yet performed._

  === ANN psychometric curve and thresholds

  _Not yet performed._

  === Training-range decision

  _Not yet performed._

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
