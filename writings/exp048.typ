#import "/.demolab/lib.typ": cite, numbers-table, provenance-footer, reference-list

#let meta = (
  title: "Temporal and spatial evidence limits of trained PING",
  date: "2026-06-08",
  description: "Streaming and spatial-masking psychometric curves identify the presentation durations, encoding rates, and foreground evidence that support classification in a frozen trained PING network.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let r = json("/artifacts/data/exp048/numbers.json")
#let rs = json("/artifacts/data/exp065/numbers.json")
#let cfg = r.config
#let scfg = rs.config
#let rate-at(rate) = r.encoding_rate_psychometric.curve.filter(x => x.input_rate_hz == rate).at(0)
#let mask-at(q) = rs.matched_masking.rows.filter(x => x.q == q).at(0)
#let p05 = rate-at(0.5)
#let p2 = rate-at(2.0)
#let p3 = rate-at(3.0)
#let p5 = rate-at(5.0)
#let p10 = rate-at(10.0)
#let m1 = mask-at(1.0)
#let m02 = mask-at(0.2)
#let m01 = mask-at(0.1)
#let m005 = mask-at(0.05)
#let m002 = mask-at(0.02)
#let m0005 = mask-at(0.005)
#let q01-rate = rs.config.matched_rate_hz * m01.q
#let varying-correct = r.varying_headline.seg_correct.fold(0, (total, x) => total + x)
#let varying-conditions = (
  r
    .varying_headline
    .segments
    .map(
      x => str(x.at(0)) + " ms at " + str(x.at(1)) + " Hz",
    )
    .join("; ")
)
#let varying-predictions = (
  range(r.varying_headline.labels.len())
    .map(
      i => str(r.varying_headline.labels.at(i)) + "→" + str(r.varying_headline.seg_preds.at(i)),
    )
    .join(", ")
)
#let grid-cells = cfg.tau_grid_ms.len() * cfg.rate_grid_hz.len()
#let segments-per-cell = cfg.n_grid_streams * cfg.n_per_stream * cfg.train_seeds.len()

#let body = [
  == Abstract

  A frozen pyramidal-interneuron gamma (PING) network, whose trained weights
  remain fixed during evaluation, is tested under
  complementary temporal and spatial reductions of Modified National Institute
  of Standards and Technology (MNIST) digit evidence. It classifies continuously
  streamed digits without retraining, but a duration × input-rate sweep reveals
  a failure floor below #cfg.tau_grid_ms.at(1) ms. At fixed
  #r.encoding_rate_psychometric.presentation_ms ms presentation and
  readout windows, performance remains at #(cfg.n_classes)-class chance through
  #p05.input_rate_hz Hz
  and becomes clearly informative by #p2.input_rate_hz Hz. Separately,
  foreground pixels are permanently removed from binarized images and presented
  to both PING and a width-matched artificial neural network (ANN).
  PING is competitive at intermediate deletion but reaches chance by retention
  q = #m002.q. Together, the curves delimit temporal, event-rate, and spatial
  evidence regimes for a future variable-rate training experiment.

  == Methods

  === Streaming duration and encoding rate

  The trained baseline from #link("/exp025/")[the canonical PING experiment]
  contains #cfg.n_e excitatory (E) and #cfg.n_i inhibitory (I) cells. It was
  trained on one MNIST digit per #cfg.trained_t_ms ms trial using random seeds
  #cfg.train_seeds.map(str).join(", "). Each seed identifies an independent
  training run with a reproducible random initialization and data order.
  Everything here is *inference-only* at
  the trained timestep Δt = #cfg.dt ms; the weights are never updated.
  The two-dimensional sweep averages over all #cfg.train_seeds.len() seeds; the
  single-stream examples use seed #cfg.seed.

  A stream of digits, each shown for τ ms, is classified in one forward pass. The _only_ change from training is a sliding readout window:

  1. *Encode* each digit as a Poisson spike train over τ ms across
    #cfg.n_in input channels, one per pixel. Each channel generates independent
    random spikes. The stated encoding rate is the expected number of spikes
    per second for a full-intensity pixel; lower pixel intensities reduce that
    rate proportionally.
  2. *Concatenate* the per-digit trains into one input stream. At segment index
    k, the Poisson rate switches instantaneously at the boundary

    $ t_k = k tau. quad "(1)" $

    Here t#sub[k] is the boundary time of segment k, k is the segment index,
    and τ is the duration of each segment.

  3. *Run once* through the trained network at the trained Δt, without
    retraining.
  4. *Integrate evidence* in a non-spiking output leaky integrator, one unit per
    class. A leaky integrator accumulates incoming spikes while gradually
    discounting older input:

    $ v_"out" (t) = beta_"out" v_"out" (t-1) + (1 - beta_"out") / (Delta t) bold(s)^E (t-1) W_"out". quad "(2)" $

    Here t is the discrete timestep; v#sub[out] (t) is the vector of
    output-unit states; β#sub[out] is their leak factor, the fraction of the
    previous output state retained for one timestep; Δt
    is the simulation timestep; s#super[E] (t−1) is the E-cell spike
    vector at the preceding timestep; and W#sub[out] is the trained
    E-to-output weight matrix.

  5. *Read a sliding window.* Average v#sub[out] over the trailing τ-window and
    apply a softmax:

    $ "logits"(t) = (Delta t) / tau sum_(u=t-w+1)^(t) v_"out" (u). quad "(3)" $

    $ p("class", t) = "softmax"("logits"(t)). quad "(4)" $

    Here logits(t) is the class-evidence vector; u indexes
    timesteps in the window; τ is the presentation duration;
    w is its number of timesteps; and p(class,t) is
    the softmax-normalized class-probability vector. Softmax converts the logits
    into non-negative class probabilities that sum to one. The readout-window duration
    is matched exactly to the current digit's presentation duration:

    $ T_"readout" = T_"presentation" = tau. quad "(5)" $

    Here T#sub[readout] is the duration over which output evidence is averaged,
    T#sub[presentation] is the time for which the digit is shown, and τ is that
    common duration.

    The corresponding window length is

    $ w = tau / (Delta t). quad "(6)" $

    Every digit is therefore read over exactly
    its own presentation duration; readout duration is not varied independently.
    At training the average ran over the _whole_ trial; the trailing
    matched-duration window is the single change.
  6. *Predict per segment* at the end of the digit's τ-window according to

    $ hat(c)(t) = arg max_c p("class"=c, t). quad "(7)" $

    Here c indexes the #cfg.n_classes digit classes, and arg max selects the
    class with the largest probability.

  The output leak is

  $ beta_"out" = exp(-Delta t \/ tau_"out"). quad "(8)" $

  Here τ#sub[out] is the output-unit time constant, which controls how quickly
  accumulated output evidence decays, and exp denotes the exponential
  function. The probability trace p(class,t) is the network's online class
  confidence.

  The grid uses presentation durations
  #cfg.tau_grid_ms.map(x => str(x) + " ms").join(", ") and input rates
  #cfg.rate_grid_hz.map(x => str(x) + " Hz").join(", ") per channel. This gives
  #grid-cells cells with #segments-per-cell classified segments per cell.

  To resolve the encoding-rate floor below the grid, additional evaluations use
  rates #r.encoding_rate_psychometric.new_rates_hz.map(x => str(x) + " Hz").join(", ") while holding both presentation and readout at
  #r.encoding_rate_psychometric.presentation_ms ms. Each cell contains
  #(r.encoding_rate_psychometric.new_streams_per_seed) streams of
  #(r.encoding_rate_psychometric.digits_per_stream) digits for every trained
  seed. The #cfg.rate_grid_hz.map(x => str(x) + " Hz").join(", ") points use
  the same fixed-duration protocol and come from the corresponding grid row.

  === Foreground-retention calibration

  The spatial protocol uses the same MNIST split and has two parts:

  1. Train #(scfg.seeds.len()) seeds of a width-matched artificial neural
    network (ANN) with #(scfg.n_input) inputs, one rectified-linear hidden
    layer of #(scfg.n_hidden) units, and #(scfg.n_classes) outputs. The hidden
    width matches the PING E population, not its recurrent E/I architecture.
    A rectified-linear unit outputs zero for a negative input and otherwise
    passes the input unchanged. Training uses #(scfg.epochs) epochs, or complete
    passes through the training set, batches of #(scfg.batch_size) images per
    weight update, and learning rate #(scfg.learning_rate), the step size of
    each update.
  2. Binarize each held-out image, meaning an image excluded from training, at
    intensity #(scfg.binarize_threshold): pixels above the threshold become
    unit-valued foreground and all others become zero-valued background. Retain
    every foreground pixel independently with probability q. Retention q = 1
    leaves the foreground intact and q = 0 removes it. The ANN calibration uses
    #(scfg.mask_draws) independent mask realizations per image. The matched comparison uses
    #(scfg.matched_images) fixed held-out examples and identical masks for
    every ANN and PING seed; PING encodes them at
    #(scfg.matched_rate_hz) Hz for
    #(scfg.matched_presentation_ms) ms.

  == Results

  === Streaming classification and temporal evidence

  #figure(
    image(
      "/artifacts/data/exp048/varying_headline_stream.png",
      width: 100%,
      alt: "A digit stream where each segment has its own duration and input rate, with errors marked in red.",
    ),
    caption: [Classification when presentation duration and encoding rate vary
      between segments. The segment conditions are #varying-conditions. Thumbnail
      opacity increases with encoding rate. The middle panels plot E- and I-cell
      spike rasters against time (ms); the lower panel plots class probability
      against time (ms), with the true class emphasized in red. The
      label-to-prediction pairs are #varying-predictions, giving
      #varying-correct of #r.varying_headline.labels.len() correct segments.],
  )

  #figure(
    image(
      "/artifacts/data/exp048/acc_grid_tau_rate.png",
      width: 100%,
      alt: "A duration-by-input-rate accuracy heatmap beside a fixed-duration encoding-rate psychometric curve.",
    ),
    caption: [Temporal and encoding-rate limits of the frozen PING classifier.
      *(A)* Per-segment accuracy (%) is shown for presentation duration (ms,
      horizontal) and Poisson encoding rate (Hz per channel, vertical), using
      #segments-per-cell segments per cell. *(B)* Probability of a correct
      classification (%) is plotted against encoding rate (Hz) with presentation
      and readout fixed at #r.encoding_rate_psychometric.presentation_ms ms. The
      inset enlarges the linear
      #r.encoding_rate_psychometric.new_rates_hz.first()–#p10.input_rate_hz
      Hz interval without changing the axis scale. The
      dashed line marks #(cfg.n_classes)-class chance and the dotted line the
      #r.encoding_rate_psychometric.trained_rate_hz Hz training rate. Accuracy
      stays at its empty-input floor, the accuracy obtained when almost no input
      spikes arrive, through #p05.input_rate_hz Hz, becomes
      informative by #p2.input_rate_hz Hz, and reaches
      #calc.round(100 * p5.accuracy, digits: 1)% at #p5.input_rate_hz Hz.],
  )

  The fixed-duration rate curve distinguishes a nonviable encoder regime from ordinary
  classification errors under weak evidence. In the variable-condition stream,
  the first failed segment received #p10.input_rate_hz Hz for
  #r.varying_headline.segments.at(0).at(0) ms, yet that condition
  reaches #calc.round(100 * p10.accuracy, digits: 1)% across the population. Its
  error is therefore natural trial-level variation, not evidence that
  #p10.input_rate_hz Hz is intrinsically too low. The other failed segment,
  presented at #r.varying_headline.segments.at(4).at(1) Hz for
  #r.varying_headline.segments.at(4).at(0) ms, is likewise above the empty-input
  rate floor, although its shorter window supplies less total evidence. Rates
  below #p05.input_rate_hz Hz are not useful
  operating points; #p2.input_rate_hz Hz is the lowest clearly informative tested
  rate and #p5.input_rate_hz Hz is a practical lower bound for future sweeps.

  === Spatial evidence calibration

  The architecture-matched ANN remains above chance until foreground retention
  falls to q = #m0005.q, fewer than one visible foreground pixel per image on
  average.

  #figure(
    image(
      "/artifacts/data/exp065/ann_masking_calibration.svg",
      width: 100%,
      alt: "ANN probability of correct classification against foreground-pixel retention probability.",
    ),
    caption: [Held-out ANN accuracy as foreground evidence is removed.
      Probability of a correct classification (%) is plotted against foreground
      retention q, the independent probability that a foreground pixel
      remains visible. Points are means across #scfg.seeds.len() ANN seeds and the
      band is one standard error, the estimated uncertainty of that mean across
      seeds. The dashed line marks
      #(scfg.n_classes)-class chance; the dotted line marks the measured
      chance-region bound at q = #rs.chance_bound.q, the highest tested retention
      whose 95% confidence interval across seeds still contains chance accuracy.],
  )

  Under identical masks, neither classifier is uniformly better. With
  #calc.round(m02.mean_visible_foreground_pixels, digits: 1) visible pixels on
  average (q = #m02.q), PING reaches
  #calc.round(100 * m02.ping_accuracy, digits: 1)% against the ANN's
  #calc.round(100 * m02.ann_accuracy, digits: 1)%. They coincide near
  #calc.round(100 * m01.ping_accuracy, digits: 1)% at q = #m01.q. PING still
  leads at q = #m005.q, then reaches chance at q = #m002.q while the ANN
  remains above it.

  #figure(
    image(
      "/artifacts/data/exp065/matched_masking.svg",
      width: 100%,
      alt: "ANN and frozen PING classification accuracy against foreground retention on the same held-out examples and masks.",
    ),
    caption: [Width-matched ANN and frozen PING accuracy under identical spatial
      deletion. Probability of a correct classification (%) is plotted against
      foreground retention q. Black circles denote ANN and red squares PING;
      bands show one standard error across trained seeds. Each point uses
      #(m1.n_images) fixed held-out examples and identical independently sampled
      foreground masks. PING runs at #(scfg.matched_rate_hz) Hz for
      #(scfg.matched_presentation_ms) ms. Neither classifier dominates across the
      full retention range.],
  )

  #figure(
    image(
      "/artifacts/data/exp065/masking_diagnostics.png",
      width: 100%,
      alt: "Example masked digits and row-normalized ANN and PING confusion matrices at five foreground-retention levels.",
    ),
    caption: [Stimulus and error structure across the matched masking curve.
      Rows progress from intact input through intermediate masking to the blank
      control. Left panels show five binarized examples and their mean visible
      foreground-pixel count. The ANN and PING panels show confusion matrices
      with true digit on the vertical axis and predicted digit on the horizontal
      axis; each row is normalized to show the distribution of predictions for
      one true class. The lowest-retention rows reveal collapse toward
      model-specific default classes rather than structured digit confusions.],
  )

  Binarization maps every nonzero antialiased MNIST pixel, including
  intermediate-intensity pixels introduced to smooth digit edges, to full intensity, so
  spatial retention is not a second measurement of grayscale contrast.
  Nevertheless, expected input-event count gives a useful first-order bridge.
  For an otherwise identical binary image encoded at the masking experiment's
  #rs.config.matched_rate_hz Hz ceiling, retaining fraction q gives the same
  expected event count as retaining the full foreground and using

  $ r_"equiv" = q dot #rs.config.matched_rate_hz " Hz". quad "(9)" $

  Here r#sub[equiv] is the full-foreground Poisson rate with the same
  expected event count, meaning the expected total number of input spikes
  across the image and presentation; q is foreground retention; and
  #scfg.matched_rate_hz Hz is the masking experiment's reference encoding rate.

  Thus q = #m01.q maps to r#sub[equiv] = #q01-rate Hz, within the
  #p2.input_rate_hz–#p3.input_rate_hz Hz transition of the fixed-duration rate
  curve. At that
  retention, PING reaches #calc.round(100 * m01.ping_accuracy, digits: 1)%,
  compared with #calc.round(100 * p2.accuracy, digits: 1)% at
  #p2.input_rate_hz Hz in the grayscale rate sweep. This numerical alignment
  supports including a rate near #q01-rate Hz in variable-rate training, but it
  is not an equivalence of corruptions: spatial masking permanently removes
  locations, whereas lowering Poisson rate preserves all locations in
  expectation and changes temporal sampling noise.

  == Appendix: Proposed filter-matched variable-rate calibration

  Prior ANN-to-SNN matching work establishes the pieces of this proposal without
  answering its diagnostic question. Hunsberger and Eliasmith train an ANN
  against variability estimated from synaptically filtered LIF spike trains,
  and show that the filtered-response distribution changes shape with input
  current#cite(1). Rueckauer et al. derive finite-window rate-approximation
  errors and report that Poisson image encoding introduces enough variability
  to impair converted networks, which they avoid by using analog input
  currents#cite(2). Tang et al. map accumulated SNN spikes into a weight-sharing
  ANN branch for efficient training#cite(3), but that shared branch is not an
  independent test of whether the input remains informative. The Neural
  Engineering Framework instead provides the relevant decoding principle:
  filter population spikes with a postsynaptic response and fit a decoder to
  the resulting continuous representation#cite(4). The present proposal
  combines these ideas for a different purpose. It uses PING's exact frozen
  feedforward filter and an independently trained decoder to locate a
  decoder-relative information floor as encoding rate falls.

  === High-level overview

  The proposal asks a simple question:

  #quote(block: true)[
    At a given encoding rate, does the signal delivered to the PING excitatory
    cells still contain enough information to recognize the digit?
  ]

  The primary experiment trains a normal artificial neural network (ANN) on a
  static summary of the same Poisson input that drives PING. The summary is not
  a raw spike count. Each encoded image passes through the trained input
  projection, the AMPA synaptic conductance, and a bank of uncoupled,
  non-spiking excitatory-cell membranes. The membrane voltage is then averaged
  over the same #scfg.matched_presentation_ms ms presentation used by PING.
  This produces one static feature per excitatory cell with the mean and
  sampling variability created by the actual input filter. Training on a
  filtered representation follows the same broad strategy as noise-aware
  ANN-to-SNN matching#cite(1), but the exact simulation is retained because
  low-rate filtered responses need not have a Gaussian distribution.

  The probe excludes recurrent excitation, recurrent inhibition, spike
  threshold, reset, and the trained output layer. It therefore asks whether
  PING's own feedforward input stage preserves decodable digit evidence before
  the recurrent circuit acts on it. The new *variable-rate ANN* is trained on
  these filter-matched features while the encoding rate changes from example to
  example. Its held-out accuracy curve is the primary calibration used to
  choose the lower rate for variable-rate PING training. A linear softmax
  decoder, whose class scores are weighted sums of the probe features, is
  trained on the same examples as a diagnostic. It tests whether the available
  digit evidence is already linearly accessible, following the population
  decoding logic of the Neural Engineering Framework#cite(4). The nonlinear
  variable-rate ANN remains the model used to set the information floor.

  This construction cannot use one fixed membrane leak factor. The quoted
  passive membrane time constant applies only when synaptic conductance is
  zero. Once input arrives, the total conductance changes both the membrane gain
  and its effective time constant. The exact conductance-based membrane
  equation is therefore used to generate ANN features. A local linear
  approximation is reserved for the complementary transfer-function and Bode
  analysis.

  The proposal has three outputs:

  - The *variable-rate ANN psychometric curve* is the primary result. It
    locates the lowest tested encoding rate at which the filter-matched
    feedforward representation remains reliably classifiable.
  - The *linear-decoder psychometric curve* is a diagnostic result. Agreement
    with the variable-rate ANN indicates that the surviving evidence is
    linearly accessible; a gap shows that nonlinear decoding is required.
  - The *filter and Bode analysis* is a complementary mechanistic result. It
    shows how synaptic and membrane filtering shape signal bandwidth, mean,
    variance, and signal-to-noise ratio across encoding rates.
  - The existing *foreground-retention ANN curve* in Figure 3 is a
    complementary spatial-evidence result and an implementation sanity check.
    It does not set the training range.

  The interpretation is:

  - If the variable-rate ANN and PING both fail, the tested feedforward
    representation contains too little evidence for either decoder.
  - If the variable-rate ANN succeeds but PING fails, the evidence reaches the
    excitatory-cell input stage, but PING's spiking, recurrent dynamics,
    readout, or training does not exploit it.
  - If both succeed, the rate is already viable.
  - If PING succeeds but the variable-rate ANN fails, the probe or ANN is an
    inadequate surrogate. That outcome invalidates the calibration rather than
    demonstrating that PING created information.

  Within the ANN control, failure of both the linear and nonlinear decoders
  provides stronger evidence for a feedforward information floor. Success of
  the nonlinear ANN alongside failure of the linear decoder instead means that
  digit evidence survives but is not linearly accessible.

  This is a decoder-relative information floor, not a proof that no conceivable
  decoder could recover information at a lower rate.

  === Filter-matched static signal

  For pixel i, the discrete encoder draws

  $ S_i (t) tilde "Bernoulli"(r Delta t x_i). quad "(10)" $

  Here S#sub[i] (t) is the binary input spike at timestep t, r is the maximum
  encoding rate in spikes per second, Δt is the simulation timestep in seconds,
  and x#sub[i] is the grayscale intensity of pixel i between zero and one.

  For excitatory probe cell j, the feedforward AMPA conductance is

  $ g_j^"ff" (t) = alpha_"AMPA" g_j^"ff" (t-1) + sum_i S_i (t) W_"in" (i,j). quad "(11)" $

  The synaptic decay factor is

  $ alpha_"AMPA" = exp(-(Delta t) / tau_"AMPA"). quad "(12)" $

  Here g#super[ff]#sub[j] is the feedforward excitatory conductance of probe cell
  j; W#sub[in] (i,j) is the trained weight from pixel i to excitatory cell j;
  α#sub[AMPA] is the fraction of AMPA conductance retained for one timestep; and
  τ#sub[AMPA] is the AMPA conductance time constant. Equation 11 is the same
  input-conductance update used by PING, with recurrent contributions omitted.

  Each probe cell follows the non-spiking conductance-based membrane equation

  $ C_E (d v_j) / (d t) = g_"L,E" (E_L - v_j) + g_j^"ff" (t) (E_e - v_j). quad "(13)" $

  Its instantaneous effective membrane time constant is

  $ tau_"eff",j (t) = C_E / (g_"L,E" + g_j^"ff" (t)). quad "(14)" $

  Here v#sub[j] is the probe-cell voltage; C#sub[E] is the excitatory-cell
  capacitance; g#sub[L,E] is its leak conductance; E#sub[L] is the leak reversal
  potential; E#sub[e] is the excitatory reversal potential; and
  τ#sub[eff,j] is its conductance-dependent effective time constant. The passive
  time constant is only the zero-input limit of Equation 14. Input conductance
  shortens the effective time constant, so a fixed membrane β would be wrong.

  The static feature supplied to the variable-rate ANN is

  $ z_j = 1 / T integral_0^T (v_j (t) - E_L) d t. quad "(15)" $

  Here z#sub[j] is the time-averaged, baseline-subtracted voltage of probe cell
  j and T is the presentation duration. Subtracting the fixed resting potential
  only centres the features. The feature is not divided by encoding rate:
  PING is not told the rate, and removing the rate-dependent change in mean
  drive would no longer reproduce its input conditions. Equation 15 implements
  the requested normalization by elapsed time.

  Across independent Poisson draws, characterize each feature by

  $ mu_j (r, bold(x)) = E[z_j | r, bold(x)], quad sigma_j^2 (r, bold(x)) = "Var"[z_j | r, bold(x)]. quad "(16)" $

  A dimensionless repeatability signal-to-noise summary is

  $ "SNR"_j (r, bold(x)) = abs(mu_j (r, bold(x))) / (sigma_j (r, bold(x))). quad "(17)" $

  Here the bold x in Equation 16 denotes the complete grayscale image;
  μ#sub[j] is the mean static feature; σ#super[2]#sub[j] is its variance;
  σ#sub[j] is its standard
  deviation; E denotes an average over repeated Poisson encodings; Var denotes
  variance over those encodings; and SNR is signal-to-noise ratio. In this
  ratio, signal means the mean drive produced by one fixed image and noise
  means variation across its independent Poisson encodings. It measures
  repeatability, not separation between digit classes. The decoder
  psychometric curves provide the population-level test of class information.
  Report the distribution of these quantities across cells and images rather
  than hiding their heterogeneity in one grand average.

  The primary psychometric curve is

  $ A_r (r) = P("correct" | r, "filter-matched features"). quad "(18)" $

  Its information floor is

  $ r_"floor" = "lowest " r in cal(R) " satisfying " L_r (r) > 1 / N_"class". quad "(19)" $

  Here A#sub[r] is held-out variable-rate ANN accuracy. P denotes probability.
  The calligraphic R in Equation 19 is the evaluated rate grid. L#sub[r] is the
  lower confidence bound for A#sub[r]. N#sub[class] is the number of digit
  classes. The result r#sub[floor] is the lowest tested rate whose lower
  confidence bound exceeds chance.

  === Complementary transfer-function analysis

  The exact feature generator uses Equations 10–15. For interpretation only,
  linearize the membrane around the mean conductance observed for a particular
  rate, image, and cell. After normalizing to unit gain at zero temporal
  frequency, the synapse-membrane cascade is

  $ H_j (f; r, bold(x)) = 1 / ((1 + i 2 pi f tau_"AMPA") (1 + i 2 pi f macron(tau)_"eff",j (r, bold(x)))). quad "(20)" $

  Here H#sub[j] is the local transfer function; f is temporal modulation
  frequency in hertz; i is the imaginary unit; and the overbarred effective
  time constant in Equation 20 is Equation 14 evaluated at the mean feedforward
  conductance for the selected operating point. Encoding rate is not the
  horizontal axis of a Bode plot. It selects the operating point and therefore
  selects one member of a family of transfer curves. This explicit dependence
  on the temporal filter is consistent with population-decoding approaches in
  which the assumed postsynaptic response is part of the decoding
  problem#cite(4).

  The two local corner frequencies are

  $ f_"AMPA" = 1 / (2 pi tau_"AMPA"), quad f_"mem",j = 1 / (2 pi macron(tau)_"eff",j). quad "(21)" $

  Here f#sub[AMPA] is the AMPA corner frequency and f#sub[mem,j] is the local
  membrane corner frequency. The larger time constant produces the lower
  corner. Because the membrane time constant changes across rates, images, and
  cells, plot median Bode magnitude with an interval across those operating
  points rather than presenting one membrane curve as universal.

  The finite presentation average in Equation 15 adds the boxcar magnitude

  $ B_T (f) = abs(sin(pi f T) / (pi f T)). quad "(22)" $

  Here B#sub[T] is the magnitude response of a T-second averaging window, with
  value one at zero frequency by continuity. Show both the intrinsic cascade in
  Equation 20 and the task-level response formed by multiplying Equations 20
  and 22. This separates cellular filtering from the additional low-pass effect
  of averaging over the full presentation.

  For independent, locally linear filtered Poisson inputs, the stationary
  moments provide an analytic check:

  $ E[y_j] = sum_i lambda_i integral_0^infinity h_"ij" (u) d u, quad "Var"[y_j] = sum_i lambda_i integral_0^infinity h_"ij" (u)^2 d u. quad "(23)" $

  Here y#sub[j] is the locally linear filtered signal at cell j; λ#sub[i] is the
  Poisson event rate of pixel channel i; h#sub[ij] is the impulse response from
  pixel i to cell j, including its input weight; and u is time. Compare these
  predicted moments with a long simulation of the exact non-spiking
  conductance model. Agreement validates the local linear approximation.
  Disagreement does not invalidate the ANN experiment, which uses the exact
  simulation, but it limits interpretation of the Bode and analytic
  signal-to-noise results.

  === Steps

  1. *Lock the three roles before implementation.* Treat A#sub[r], the
    variable-rate ANN curve, as the primary calibration. Treat the Bode and
    moment analysis as a complementary mechanistic result. Retain A#sub[q], the
    existing foreground-retention curve in Figure 3, as a complementary spatial
    result and sanity check only.
  2. *Reuse the established data partitions and baselines.* Use the MNIST
    training and held-out partitions already used by the foreground-retention
    experiment. Reuse the three frozen PING baselines and their trained input
    matrices. Do not use held-out labels while designing the probe, selecting a
    normalization, or checking the transfer approximation.
  3. *Fix the task timing and rate grid.* Use
    T = #scfg.matched_presentation_ms ms for every example, matching the
    presentation and readout window in Figure 2. Evaluate 0.25, 0.5, 0.75, 1,
    1.5, 2, 2.5, 3, 4, 5, 10, and 25 Hz. These points resolve the current
    #p2.input_rate_hz–#p3.input_rate_hz Hz PING transition and retain the
    #scfg.matched_rate_hz Hz trained-rate endpoint.
  4. *Build the exact feedforward probe.* For each frozen PING seed, route
    Poisson input spikes through its trained W#sub[in] and AMPA conductance
    using Equations 10–12. Apply the same conductance-based excitatory membrane
    update used by PING, but disable recurrence, threshold, reset, refractory
    state, adaptation, and the output layer. Vectorize this as
    #scfg.n_hidden uncoupled probe cells. This is the primary feature generator;
    no fixed membrane time constant or single membrane β is inserted.
  5. *Validate the probe implementation.* At zero input, verify that every
    probe remains at its resting potential. For controlled constant
    conductances, compare the numerical voltage update and effective time
    constant with Equations 13 and 14. For Poisson drive, verify the AMPA
    conductance trace against Equation 11. These unit checks must pass before
    any classifier is trained.
  6. *Separate stationary filter characterization from the finite task
    window.* For the mechanistic analysis, drive the non-spiking probe for long
    enough to discard its initial transient, then estimate stationary
    conductance and voltage distributions across training images, cells, rates,
    and repeated Poisson draws. For the ANN dataset, restart the probe from the
    same initial state used by PING and retain exactly the first
    #scfg.matched_presentation_ms ms. The long run estimates steady-state
    filter statistics; the finite run generates the task-matched ANN feature.
    Do not substitute one for the other. Finite-window rate estimates converge
    with integration time and can accumulate approximation errors through a
    spiking network#cite(2), so the task-matched feature must include the same
    startup transient and observation window as PING.
  7. *Measure the equivalent static signal.* Apply Equation 15 to every finite
    probe trace. Across repeated encodings, estimate the mean, variance, and SNR
    in Equations 16 and 17, together with the median, interquartile range, and
    skewness of each feature distribution. The interquartile range spans the
    middle half of repeated encodings; skewness records distribution
    asymmetry. Plot their distributions against encoding rate. This establishes
    whether the low-rate regime is dominated by sparse shot-to-shot
    fluctuations, whether an approximately Gaussian middle regime emerges, and
    whether relative variability falls as rate increases. Do not assume that
    absolute variance approaches zero at high rate: filtered-spike
    distributions can change both scale and shape with drive#cite(1).
  8. *Produce the complementary Bode analysis.* At representative low, middle,
    and high encoding rates, compute the local effective time constants from
    the stationary probe states. Plot the normalized magnitude of Equation 20
    against temporal frequency on logarithmic axes, with median and interval
    across cells and training images. Mark both corner frequencies from
    Equation 21. Add the task-level response including Equation 22. Validate the
    local moment prediction in Equation 23 against the exact simulation and
    label the Bode result explicitly as a local approximation.
  9. *Generate training features for the variable-rate ANN.* For every MNIST
    training presentation, sample one rate uniformly from the Step 3 grid,
    generate a fresh Poisson encoding, run the exact finite-window probe, and
    use the #scfg.n_hidden values from Equation 15 as the static input vector.
    Do not give the sampled rate to the ANN as an extra feature. Fresh spike
    draws on every epoch prevent memorization of particular realizations.
  10. *Train the variable-rate ANN and linear diagnostic.* Use a conventional
      classifier with one
      rectified-linear hidden layer of #scfg.n_hidden units and
      #scfg.n_classes outputs. Its input dimension is #scfg.n_hidden because it
      decodes probe-cell features after W#sub[in], whereas the existing
      foreground-retention ANN receives #scfg.n_input pixel values. Pair each
      decoder with one frozen PING input projection and repeat for seeds
      #scfg.seeds.map(str).join(", "). On the same features, also train a linear
      softmax decoder with no hidden layer. Keep its curve diagnostic rather
      than using it to set the training range. Unlike weight-sharing ANN-to-SNN
      training#cite(3), neither decoder shares its learned classification
      weights with PING.
  11. *Generate A#sub[r] by inference only.* Freeze every variable-rate ANN and
      linear decoder.
      For each rate, evaluate the same fixed held-out images using fixed
      Poisson draws shared across ANN seeds where possible, plus additional
      independent draws to measure sampling variation. Plot both probabilities
      of a correct classification against rate, identifying the nonlinear ANN
      curve as A#sub[r] and the linear-decoder curve as a diagnostic. No decoder
      weights are updated during this sweep.
  12. *Fit A#sub[r] and determine the primary information floor.* Fit isotonic
      regression, a monotonic best-fit curve that can only stay level or
      increase, to A#sub[r] and to the linear-decoder curve. Bootstrap held-out
      images, Poisson draws, and model seeds to obtain confidence intervals.
      Apply Equation 19 to the nonlinear A#sub[r] curve and round the resulting
      floor upward to the next tested rate. Report the linear curve and its
      transition separately.
  13. *Compare the primary curve with frozen PING.* Compare A#sub[r] with the
      fixed-duration PING rate curve in Figure 2 without retraining PING. Report
      both floors and the gap between them. Interpret that gap as a limitation
      downstream of the frozen input projection, not as a pure measure of
      recurrence alone. Compare the linear and nonlinear decoder curves at the
      same time: their gap distinguishes linearly accessible evidence from
      evidence that requires nonlinear decoding.
  14. *Retain foreground masking as a complementary sanity check.* Reuse
      A#sub[q] from Figure 3 without rerunning its ANN. It was produced by
      freezing an ANN trained on uncorrupted grayscale images and evaluating
      independently masked, binarized held-out images. Compare its transition
      region with A#sub[r] and with the event-budget bridge in Equation 9.
      An accuracy-matched diagnostic rate is

      $ r_"info" (q) = A_r^(-1) (A_q (q)). quad "(24)" $

      Here A#sub[q] is the existing foreground-retention ANN curve;
      A#super[−1]#sub[r] is the inverse of the fitted variable-rate curve; and
      r#sub[info] is the lowest rate at which A#sub[r] reaches the accuracy
      observed at retention q. This lowest-rate definition handles flat
      sections of the fitted curve. Evaluate Equation 24 at q = #m01.q and
      compare it with the
      #q01-rate Hz equal-event prediction. This is an order-of-magnitude check,
      not a required equality: spatial deletion, temporal undersampling, input
      representation, and ANN training differ. A gross disagreement should
      trigger inspection of the encoders, weights, feature scaling, and data
      pairing; a modest disagreement is scientifically expected.
  15. *Choose the variable-rate PING training range.* Use the rounded
      A#sub[r] information floor from Step 12 as the lower endpoint and
      #scfg.matched_rate_hz Hz as the upper endpoint. Sample uniformly from the
      retained discrete rate grid so every included operating condition
      receives equal exposure.

  #pagebreak()

  === TODO

  1. [ ] Replace Figure 1 with a representative streaming example. Generate a
    fixed set of candidate streams under the same conditions, select the stream
    whose error count is closest to that expected from the measured cell
    accuracies, and record the candidate set, selection rule, and chosen seed.
    Prefer one failure, or at most two, without selecting by digit identity.
  2. [x] Reuse A#sub[q] from Figure 3 without rerunning the
    foreground-retention ANN.
  3. [ ] Implement and unit-test the uncoupled non-spiking feedforward probe in
    Steps 4 and 5 without changing the production PING engine.
  4. [ ] Run the stationary and finite-window probe characterizations in Steps
    6 and 7, then report feature mean, variance, SNR, and distribution shape
    against encoding rate.
  5. [ ] Produce the local Bode and analytic-moment validation in Step 8.
  6. [ ] Generate filter-matched training features and train the variable-rate
    ANN and linear softmax diagnostic for seeds
    #scfg.seeds.map(str).join(", ").
  7. [ ] Freeze both decoders and run the inference-only rate sweep to generate
    A#sub[r] and the linear-decoder curve.
  8. [ ] Fit and bootstrap both curves, then report the nonlinear A#sub[r]
    information floor and the linear transition.
  9. [ ] Compare A#sub[r] with the frozen PING curve in Figure 2 and select the
    lower endpoint for variable-rate PING training.
  10. [ ] Compare A#sub[q] from Figure 3 with A#sub[r] and the Equation 9
      event-budget bridge as a complementary sanity check.
  11. [ ] After review, run the variable-rate PING training experiment over the
      selected rate range.

  #reference-list((
    (
      text: [Hunsberger & Eliasmith — _Training Spiking Deep Networks for Neuromorphic Hardware_. 2016.],
      doi: "10.48550/arXiv.1611.05141",
    ),
    (
      text: [Rueckauer, Lungu, Hu, Pfeiffer & Liu — _Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification_. Frontiers in Neuroscience, 2017.],
      doi: "10.3389/fnins.2017.00682",
    ),
    (
      text: [#link("https://arxiv.org/abs/2206.09449")[Tang, Lai, Xie, Yang & Zheng — _SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks_]. 2022.],
    ),
    (
      text: [Bekolay, Bergstra, Hunsberger, DeWolf, Stewart, Rasmussen, Choo, Voelker & Eliasmith — _Nengo: A Python Tool for Building Large-Scale Functional Brain Models_. Frontiers in Neuroinformatics, 2014.],
      doi: "10.3389/fninf.2013.00048",
    ),
  ))
]
