#let meta = (
  title: "ANN masking and PING encoding-rate psychometric curves",
  date: "2026-07-15",
  description: "An architecture-matched artificial neural network calibrates how MNIST accuracy falls as foreground pixels are removed, alongside encoding-rate and matched-mask psychometric curves for the frozen trained PING network.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let r = json("/artifacts/data/exp065/numbers.json")
#let ping-at(rate) = r.ping_rate.curve.filter(x => x.input_rate_hz == rate).at(0)
#let p05 = ping-at(0.5)
#let p1 = ping-at(1.0)
#let p2 = ping-at(2.0)
#let p3 = ping-at(3.0)
#let p5 = ping-at(5.0)
#let p10 = ping-at(10.0)
#let mask-at(q) = r.matched_masking.rows.filter(x => x.q == q).at(0)
#let m1 = mask-at(1.0)
#let m02 = mask-at(0.2)
#let m01 = mask-at(0.1)
#let m005 = mask-at(0.05)
#let m002 = mask-at(0.02)
#let m0005 = mask-at(0.005)

#let body = [
  == Abstract

  A pyramidal--interneuron gamma (PING) network trained on clean grayscale
  handwritten digits encounters two different forms of weak evidence: fewer
  Poisson-encoded spikes, or permanent removal of foreground pixels. An
  architecture-matched artificial neural network (ANN) provides the non-spiking
  reference. The frozen PING classifier stays at chance through
  #p05.input_rate_hz Hz and reaches
  #calc.round(100 * p2.accuracy, digits: 1)% correct at #p2.input_rate_hz Hz.
  Under matched spatial masking, PING outperforms the ANN at intermediate
  deletion but reaches chance sooner: at retention $q = #m002.q$, PING is at
  #calc.round(100 * m002.ping_accuracy, digits: 1)% while the ANN remains at
  #calc.round(100 * m002.ann_accuracy, digits: 1)%. The calibrated range now
  bounds a follow-up training experiment with variable evidence.

  == Methods

  The dataset is MNIST, a ten-class collection of grayscale handwritten digits.
  The standard corpus is split into #(r.dataset.train_samples) training and
  #(r.dataset.test_samples) held-out examples. The protocol has three ordered
  parts:

  + *Calibrate foreground deletion with an ANN.* Train
    #(r.config.seeds.len()) seeds of a non-spiking classifier with
    #(r.config.n_input) inputs, one rectified-linear-unit (ReLU) hidden layer of
    #(r.config.n_hidden) units (matching the PING excitatory population), and
    #(r.config.n_classes) outputs. Training uses #(r.config.epochs) epochs,
    batches of #(r.config.batch_size), and learning rate
    #(r.config.learning_rate). Binarize each held-out image at intensity
    #(r.config.binarize_threshold), then retain every foreground pixel
    independently with probability $q$. Here $q = 1$ leaves the foreground
    intact and $q = 0$ removes it. Evaluate #(r.config.mask_draws) mask draws per
    image at every registered $q$.
  + *Measure the frozen PING encoding-rate curve.* Hold presentation at
    #(r.ping_rate.presentation_ms) ms and sweep the maximum per-pixel Poisson
    encoding rate. Newly simulate the low-rate cells and reuse the matching
    published cells from #link("/exp048/")[exp048] at higher rates. No PING
    weight changes during this experiment.
  + *Compare spatial masking directly.* Select
    #(r.config.matched_images) fixed held-out images. Present the identical
    binarized images and Bernoulli masks to every ANN and PING seed. PING uses a
    maximum encoding rate of #(r.config.matched_rate_hz) Hz and its trained
    presentation duration. This paired subset makes differences attributable to
    the classifiers rather than different corruptions.

  == Results

  === The ANN needs almost complete foreground removal to reach chance

  The architecture-matched ANN remains above chance until foreground retention
  falls to $q = #m0005.q$, fewer than one visible foreground pixel per image on
  average.

  #figure(
    image("/artifacts/data/exp065/ann_masking_calibration.svg", width: 100%,
      alt: "ANN probability of correct classification against foreground-pixel retention probability."),
    caption: [Held-out ANN classification accuracy as foreground evidence is removed. The horizontal axis is foreground-retention probability $q$ (dimensionless); the vertical axis is the probability of a correct classification, in percent. Points are means across the architecture-matched ANN seeds and the band is one standard error across seeds. The dashed red line is ten-class chance performance; the dotted line marks the chance-region bound read from the run. Accuracy stays above chance until almost every foreground pixel is absent.],
  )

  === PING has an empty-input floor followed by a steep rate transition

  The frozen network remains on its zero-input floor through
  #p05.input_rate_hz Hz. Its accuracy then rises from
  #calc.round(100 * p1.accuracy, digits: 1)% at #p1.input_rate_hz Hz to
  #calc.round(100 * p3.accuracy, digits: 1)% at #p3.input_rate_hz Hz, before
  joining exp048's #calc.round(100 * p5.accuracy, digits: 1)% point at
  #p5.input_rate_hz Hz.

  #figure(
    image("/artifacts/data/exp065/ping_encoding_rate.svg", width: 100%,
      alt: "Frozen PING probability of correct classification against Poisson encoding rate on a linear axis."),
    caption: [Held-out classification accuracy of the existing frozen PING networks against input strength. The horizontal axis is the maximum per-pixel Poisson encoding rate in hertz; the vertical axis is the probability of a correct classification, in percent. Presentation lasts #(r.ping_rate.presentation_ms) ms. Red circles are newly simulated low-rate cells; black squares reuse published #link("/exp048/")[exp048] cells, with marker shape preserving the source distinction without colour. Points are means across the trained seeds and the band is one standard error. The dashed line marks ten-class chance and the dotted line the #(r.ping_rate.trained_rate_hz) Hz training rate. Below the onset the encoder usually supplies no spikes; classification then rises steeply between #p1.input_rate_hz and #p5.input_rate_hz Hz.],
  )

  === PING is competitive at intermediate masking but reaches chance sooner

  The matched comparison does not have one uniformly better classifier. With
  #calc.round(m02.mean_visible_foreground_pixels, digits: 1) visible pixels on
  average ($q = #m02.q$), PING reaches
  #calc.round(100 * m02.ping_accuracy, digits: 1)% against the ANN's
  #calc.round(100 * m02.ann_accuracy, digits: 1)%. They coincide near
  #calc.round(100 * m01.ping_accuracy, digits: 1)% at $q = #m01.q$. PING still
  leads at $q = #m005.q$, then reaches chance at $q = #m002.q$ while the ANN
  remains above it. The PING transition is steeper, not uniformly worse.

  #figure(
    image("/artifacts/data/exp065/matched_masking.svg", width: 100%,
      alt: "ANN and frozen PING classification accuracy against foreground retention on the same held-out examples and masks."),
    caption: [Architecture-matched ANN and frozen PING classification under identical spatial deletion. The horizontal axis is foreground-retention probability $q$ (dimensionless); the vertical axis is the probability of a correct classification, in percent. Black circles denote ANN and red squares PING, so marker shape also identifies each classifier in grayscale. Each point uses #(m1.n_images) fixed held-out examples and the same Bernoulli foreground masks; uncertainty is one standard error across trained seeds. PING runs at #(r.config.matched_rate_hz) Hz for #(r.ping_rate.presentation_ms) ms. PING is competitive at intermediate retention but crosses into its chance region before the ANN.],
  )

  #figure(
    image("/artifacts/data/exp065/masking_diagnostics.png", width: 100%,
      alt: "Example masked digits and row-normalized ANN and PING confusion matrices at five foreground-retention levels."),
    caption: [Stimulus and error structure across the matched masking curve. Rows progress from intact input through intermediate masking, the PING and ANN chance regions, and the blank control; the left panel in each row shows five example digits and reports $q$ plus the mean number of visible foreground pixels. The middle and right panels are row-normalized ANN and PING confusion matrices: rows are true classes and columns predicted classes. The blank-input columns reflect each model's constant prediction and the class prevalence in this fixed subset, not evidence extracted from a blank image.],
  )

  #emph[Caveat.] Binarization maps every nonzero antialiased MNIST pixel to full
  intensity. At $q = #m1.q$ that drives PING more strongly than the original
  grayscale image at the same encoding-rate ceiling. This curve therefore
  measures robustness to permanent spatial deletion from a binarized stimulus;
  it is not a second measurement of grayscale contrast.

  === The chance bounds define the training range and explain exp048's failures

  The ANN reaches chance near $q = #m0005.q$; matched PING by $q = #m002.q$.
  These bounds set the range for variable-evidence training. Encoding-rate
  reduction and spatial masking remain distinct because they remove different
  evidence. For rate sweeps, #p2.input_rate_hz Hz is the lowest tested point
  with clearly useful population-level information, while #p5.input_rate_hz Hz
  is a more practical lower operating point; denser sampling below
  #p05.input_rate_hz Hz would mostly characterize empty encoder output.

  This also changes the interpretation of #link("/exp048/")[exp048]'s two errors
  in its variable-condition stream. The failed 5 was presented for 200 ms at
  #p10.input_rate_hz Hz, a condition at which the population accuracy here is
  #calc.round(100 * p10.accuracy, digits: 1)%, far above chance. The failed 7 at
  15 Hz and 75 ms likewise lies above the empty-input rate floor, although its
  shorter integration window supplies less total evidence. Those examples are
  therefore natural trial-level classification failures under weak evidence,
  not evidence that their encoding rates are generally too low to represent the
  digits. Encoding rate should only be discarded as uninformative below the
  population-level transition measured here, rather than whenever an individual
  example is misclassified.
]
