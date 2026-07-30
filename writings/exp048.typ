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

  == Appendix: Neo-proposal for filter-matched variable-rate calibration

  === Neo-proposal

  1. *Ask one operational question.* At fixed
    #scfg.matched_presentation_ms ms presentation, what is the lowest Poisson
    encoding rate at which PING's frozen feedforward input stage still supports
    useful digit classification? The answer will set the lower endpoint of a
    subsequent variable-rate PING training run.
  2. *Feature generation.* Per-pixel spike counts would discard event timing
    and therefore omit the AMPA decay and finite-window membrane response.
    Generate features from the frozen feedforward dynamics so the ANN instead
    receives the filter-matched signal reaching PING's hidden layer. For each
    frozen PING seed, route the same Poisson input through its trained input
    projection, AMPA conductance, and #scfg.n_hidden uncoupled non-spiking
    excitatory membranes. Exclude recurrence, threshold, reset, refractory
    state, adaptation, and the trained output layer. For pixel i, draw

    $ S_i (t) tilde "Bernoulli"(r Delta t x_i). quad "(10)" $

    Update the feedforward conductance of probe cell j according to

    $ g_j^"ff" (t) = beta_"AMPA" g_j^"ff" (t-1) + sum_i S_i (t) W_"in" (i,j). quad "(11)" $

    The AMPA decay factor is

    $ beta_"AMPA" = exp(-(Delta t) / tau_"AMPA"). quad "(12)" $

    Each probe membrane follows

    $ C_E (d v_j) / (d t) = g_"L,E" (E_L - v_j) + g_j^"ff" (t) (E_e - v_j). quad "(13)" $

    Its instantaneous effective time constant is

    $ tau_"eff",j (t) = C_E / (g_"L,E" + g_j^"ff" (t)). quad "(14)" $

    Here S#sub[i] (t) is the input spike from pixel i at timestep t; r is the
    maximum encoding rate in spikes per second; Δt is the simulation timestep
    in seconds; x#sub[i] is grayscale pixel intensity; g#super[ff]#sub[j] is the
    feedforward conductance of probe cell j; W#sub[in] (i,j) is the frozen input
    weight; β#sub[AMPA] is the conductance retained for one timestep;
    τ#sub[AMPA] is the AMPA time constant; C#sub[E] is excitatory-cell
    capacitance; v#sub[j] is probe voltage; g#sub[L,E] is leak conductance;
    E#sub[L] and E#sub[e] are the leak and excitatory reversal potentials; and
    τ#sub[eff,j] is the conductance-dependent membrane time constant. Equation
    14 is why a single fixed membrane leak factor is not an adequate surrogate.
    Restart the probe from the same initial state used by PING and retain
    exactly the first #scfg.matched_presentation_ms ms. Reduce each presentation
    to one static feature vector by computing, for every probe cell,

    $ z_j = 1 / T integral_0^T (v_j (t) - E_L) dif t. quad "(15)" $

    Here z#sub[j] is the time-averaged baseline-subtracted voltage and T is
    presentation duration. The feature is not divided by encoding rate because
    PING is not given that rate and experiences its rate-dependent change in
    mean drive.
  3. *Build the variable-rate feature dataset.* Apply the feature-generation
    recipe in Step 2 to the established MNIST training and held-out partitions,
    using the three frozen PING seeds and their trained input matrices. Evaluate
    0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10, and 25 Hz at fixed
    #scfg.matched_presentation_ms ms presentation. The grid ends at 25 Hz
    because the existing PING results establish it as a high-accuracy trained
    endpoint; the denser low-rate samples are intended to locate the lower
    information boundary. Generate fresh Poisson realizations during decoder
    training. Split the training images into decoder-training and validation
    subsets before selecting regularization or stopping epochs; leave held-out
    test images untouched.
  4. *Train only two mixed-rate decoders initially.* The primary decoder is a
    conventional ANN with one rectified-linear hidden layer of
    #scfg.n_hidden units and #scfg.n_classes outputs. Train a regularized linear
    softmax decoder on the same features as a diagnostic of linear
    accessibility. Neither decoder receives the numerical rate or shares
    learned classification weights with PING.
  5. *Generate the psychometric curves by inference only.* Freeze both
    decoders. At every rate, evaluate the same held-out images using fixed
    Poisson draws shared across compatible decoder and PING seeds, with
    additional draws used to measure sampling variation. The primary curve is

    $ A_r (r) = P("correct" | r, "mixed-rate nonlinear decoder on Equation 15"). quad "(16)" $

    Here A#sub[r] is held-out nonlinear-decoder accuracy and P denotes
    probability. Plot the linear curve beside it, but do not use the linear
    decoder to set the training range.
  6. *Report an informative edge and a practical training floor.* Bootstrap
    held-out images, Poisson draws, and model seeds to obtain a lower confidence
    bound L#sub[r] at every tested rate. Define the informative edge as

    $ r_"edge" = "lowest " r in cal(R) " satisfying " L_r(r) > 1 / N_"class". quad "(17)" $

    Define the practical training floor as

    $ r_"train" = "lowest " r in cal(R) " satisfying " L_r(r) >= a_"use". quad "(18)" $

    Here the calligraphic R is the tested rate grid; N#sub[class] is the number
    of digit classes; r#sub[edge] is the lowest rate reliably above chance;
    a#sub[use] is a predeclared useful-accuracy target; and r#sub[train] is the
    lower endpoint used for training. Use 50% correct as the provisional
    a#sub[use] value and lock it before inspecting the curve. Report
    r#sub[edge] descriptively, but use r#sub[train] for the training decision.
  7. *Compare with the frozen PING curve.* Compare the nonlinear ANN curve with
    the fixed-duration PING rate curve in Figure 2. If both fail, the tested
    feedforward representation contains little usable evidence for either
    decoder. If the ANN succeeds but PING fails, the evidence reaches the input
    stage but the trained PING network does not exploit it. If PING succeeds
    where the ANN fails, the probe summary or decoder is inadequate.
  8. *Keep the existing spatial results as sanity checks.* Reuse the
    foreground-retention curve in Figure 3 and the equal-event bridge in
    Equation 9 without rerunning them. Equal-event mapping compares expected
    input-spike count, not class information. Because Figure 3 uses binarized
    images while the rate curve uses grayscale inputs, treat the simple mapping
    in Equation 9 as an order-of-magnitude comparison rather than an exact
    equivalence.
  9. *Escalate only if the primary comparison demands it.* Add a coarse
    temporal decoder only if PING succeeds where the time-averaged ANN fails.
    Train rate-specific decoders only if the mixed-rate decoder shows an
    unexplained failure, initially restricting them to rates around that
    transition. These are debugging controls, not prerequisites for choosing
    the first variable-rate training range.

  === Relation to prior work

  - Wolff and Lindner show that transient voltage statistics under filtered
    conductance shot noise can depart from a fixed-effective-time-constant
    approximation, especially at low rates#cite(1). This supports using the
    exact finite-window conductance probe in Step 2.
  - Brigham and Destexhe show that nonstationary filtered shot noise can produce
    voltage distributions whose skew is missed by a Gaussian
    approximation#cite(2). This supports fresh Poisson simulation rather than
    synthetic additive ANN noise.
  - Quian Quiroga and Panzeri emphasize held-out decoder evaluation and warn
    that failure of one decoder does not prove that a neural response contains
    no information#cite(3). This is why the reported floor is explicitly
    decoder-relative.
  - Warland et al. compare linear and nonlinear population decoders#cite(4).
    This motivates the one linear diagnostic beside the primary nonlinear ANN.

  === Why Bode analysis is deferred

  A Bode analysis would describe the temporal bandwidth of a locally linearized
  synapse–membrane cascade. It would not directly determine whether digit
  identity remains classifiable or which rate should bound variable-rate
  training. The exact finite-window probe already answers the calibration
  question without relying on a linear approximation.

  In addition, PING's effective membrane time constant changes with conductance,
  so there is no single universal transfer curve. A defensible Bode result would
  require a family of operating-point-dependent curves and exact gain-and-phase
  validation. That remains useful mechanistic and thesis material, but it is a
  separate follow-up rather than a prerequisite for the neo-proposal.

  === TODO

  1. [ ] Implement and unit-test the exact finite-window feedforward probe in
    Step 2 without changing the production PING engine.
  2. [ ] Generate time-averaged filter-matched features over the fixed rate
    grid for seeds #scfg.seeds.map(str).join(", ").
  3. [ ] Train the mixed-rate nonlinear ANN and linear softmax diagnostic.
  4. [ ] Freeze both decoders and run the shared held-out rate sweep.
  5. [ ] Bootstrap both curves and report r#sub[edge] and r#sub[train].
  6. [ ] Compare the primary ANN curve with frozen PING Figure 2, Figure 3, and
    the equal-event bridge in Equation 9.
  7. [ ] Select the first variable-rate PING training range using
    r#sub[train] as its lower endpoint.
  8. [ ] Run temporal, per-rate, or Bode follow-ups only if the primary result
    triggers the conditions in Step 9.

  #reference-list((
    (
      text: [Wolff & Lindner — _Mean, Variance, and Autocorrelation of Subthreshold Potential Fluctuations Driven by Filtered Conductance Shot Noise_. Neural Computation, 2010.],
      doi: "10.1162/neco.2009.02-09-958",
    ),
    (
      text: [Brigham & Destexhe — _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
    (
      text: [Quian Quiroga & Panzeri — _Extracting Information from Neuronal Populations: Information Theory and Decoding Approaches_. Nature Reviews Neuroscience, 2009.],
      doi: "10.1038/nrn2578",
    ),
    (
      text: [Warland, Reinagel & Meister — _Decoding Visual Information From a Population of Retinal Ganglion Cells_. Journal of Neurophysiology, 1997.],
      doi: "10.1152/jn.1997.78.5.2336",
    ),
  ))
]
