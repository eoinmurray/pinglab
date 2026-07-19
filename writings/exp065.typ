#let meta = (
  title: "ANN-calibrated spatial masking in PING",
  date: "2026-07-15",
  description: "An architecture-matched artificial neural network calibrates how MNIST accuracy falls as foreground pixels are removed, then receives the same masks as the frozen trained PING network.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let r = json("/artifacts/data/exp065/numbers.json")
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
  handwritten digits encounters permanent removal of foreground pixels. An
  architecture-matched artificial neural network (ANN) calibrates when the
  damaged stimulus itself ceases to support classification. Under identical
  spatial masks, PING outperforms the ANN at intermediate
  deletion but reaches chance sooner: at retention $q = #m002.q$, PING is at
  #calc.round(100 * m002.ping_accuracy, digits: 1)% while the ANN remains at
  #calc.round(100 * m002.ann_accuracy, digits: 1)%. The calibrated range now
  bounds a follow-up training experiment with variable evidence.

  == Methods

  The dataset is MNIST, a ten-class collection of grayscale handwritten digits.
  The standard corpus is split into #(r.dataset.train_samples) training and
  #(r.dataset.test_samples) held-out examples. The protocol has two ordered
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
    caption: [Architecture-matched ANN and frozen PING classification under identical spatial deletion. The horizontal axis is foreground-retention probability $q$ (dimensionless); the vertical axis is the probability of a correct classification, in percent. Black circles denote ANN and red squares PING, so marker shape also identifies each classifier in grayscale. Each point uses #(m1.n_images) fixed held-out examples and the same Bernoulli foreground masks; uncertainty is one standard error across trained seeds. PING runs at #(r.config.matched_rate_hz) Hz for #(r.config.matched_presentation_ms) ms. PING is competitive at intermediate retention but crosses into its chance region before the ANN.],
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

  === The chance bounds define the spatial training range

  The ANN reaches chance near $q = #m0005.q$; matched PING by $q = #m002.q$.
  These bounds set the range for variable-evidence training. Encoding-rate
  reduction and spatial masking remain distinct because they remove different
  evidence. The interval between the PING and ANN chance bounds identifies
  extreme corruptions where some class information remains usable by the ANN but
  not by PING; at moderate masking, PING is competitive rather than uniformly
  fragile. Encoding-rate viability is treated separately in
  #link("/exp048/")[exp048].
]
