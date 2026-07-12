#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "What the Spiking Heidelberg Digits look like",
  date: "2026-07-11",
  description: "A first look at the SHD event-based audio benchmark before training on it: raw spike rasters of spoken digits across 700 cochlear channels, one per class and several within a class.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every dataset number below is read from this run's numbers.json, never
// hand-typed, so the prose cannot drift from what the run measured.
#let r = json("/artifacts/data/exp059/numbers.json")

#let body = [
  == Abstract

  The Spiking Heidelberg Digits (SHD)#cite(1) benchmark delivers spoken digits
  as cochlear spikes. Unlike the static MNIST images the gamma-gated-sparsity
  collection Poisson-encodes into spike trains before its
  #link("/ar003/")[conductance-based spiking network (COBANet)] ever sees them,
  SHD arrives _already_ as events, with no image and no dense array anywhere:
  each sample is a spoken digit passed through a model of the inner ear, so it
  is a list of events, each an event time and the cochlear channel that fired.
  Before training a network on it, this entry looks at the raw data: one
  utterance per class, then several utterances of a single digit. Different
  classes paint visibly distinct time–frequency signatures and repeated
  utterances of one digit share a family resemblance, so the data is
  well-behaved and separable enough to be worth training on.

  == Methods

  Nothing here runs the network: every raster is drawn straight from the raw SHD
  event lists, with no binning and no model.

  + Load the SHD train split as raw events: $N = #(r.n_utterances)$ training
    utterances (a held-out test split adds more), each a spoken digit recorded
    from several speakers and converted to spikes by a Cramer et al.#cite(1)
    cochlea model, a computational model of the inner ear that maps sound onto
    firing across an array of frequency-tuned channels.
  + Draw the class gallery: the first utterance of each of the #(r.n_classes)
    classes (the digits #(r.config.digit_min)–#(r.config.digit_max) spoken in
    German, labels #(r.config.german_label_min)–#(r.config.german_label_max),
    then English, labels #(r.config.english_label_min)–#(r.config.english_label_max)),
    each as a spike raster.
  + Draw the within-class spread: the first #(r.config.spread_panels) utterances
    of class #(r.config.spread_class), _null_, side by side.

  Every utterance is a set of events ${(t_k, u_k)}$, where:

  - $t_k$, the *spike time* of event $k$, in seconds (utterances run
    ≈ #calc.round(r.duration_min_s, digits: 1)–#calc.round(r.duration_max_s, digits: 1) s,
    median ≈ #calc.round(r.duration_median_s, digits: 1) s);
  - $u_k in {#(r.config.channel_min), ..., #(r.n_channels - 1)}$, the *cochlear channel* that fired: a
    place code for frequency, where a low channel ≈ low pitch and a high channel
    ≈ high pitch.

  So an utterance carries $C = #(r.n_channels)$ input channels and a median of
  ≈ #calc.round(r.events_per_utterance_median) events per utterance (range
  ≈ #(r.events_per_utterance_min)–#(r.events_per_utterance_max)).

  == Results

  #figure(
    image("/artifacts/data/exp059/class_gallery.png", width: 100%,
      alt: "A " + str(r.config.gallery_nrows) + "-by-" + str(r.config.gallery_ncols) + " grid of spike rasters, one spoken digit per panel, time on the x-axis and cochlear channel on the y-axis; each digit shows a distinct time-frequency pattern."),
    caption: [
      One utterance per class (German #(r.config.digit_min)–#(r.config.digit_max),
      then English #(r.config.digit_min)–#(r.config.digit_max)), each a raster of
      time (ms) against cochlear channel. *What we expect.* If SHD is learnable,
      different words should paint different time–frequency signatures. *What we
      see.* They do: a compact high-channel onset for _zwei_, a long two-lobe
      sweep for _sieben_/_seven_, a low-channel tail on _sechs_. A diffuse haze
      of isolated events sits under every panel: the cochlea model's spontaneous
      background firing, which the classifier has to see past.
    ],
  )

  #figure(
    image("/artifacts/data/exp059/within_class_spread.png", width: 100%,
      alt: str(r.config.spread_panels) + " spike rasters of the digit null side by side; all share a broad descending contour but differ in density, duration, and detail from utterance to utterance."),
    caption: [
      #(r.config.spread_panels) different utterances of class #(r.config.spread_class), _null_, each a raster of time (ms)
      against cochlear channel: the within-class spread. *What we expect.*
      Different speakers saying the same word should share a family resemblance
      while differing in the particulars. *What we see.* Exactly that: all #(r.config.spread_panels)
      carry the same broad high-channel onset falling into a mid-channel body,
      but they differ in duration, event density, and the fine structure of the
      low-channel tail. This speaker variability is what a classifier trained on
      SHD must generalise over.
    ],
  )

  == Next steps

  The data is well-behaved and visibly class-separable, so it is worth training
  on. The trainer accepts SHD directly: it bins these events onto the model's
  integration grid (a spike tensor of shape $T_"steps" times #(r.n_channels)$ at
  the integration timestep $Delta t$) and feeds them to the PING
  (pyramidal–interneuron gamma) network, the excitatory and inhibitory
  populations of the COBANet, where:

  - $T_"steps"$, the number of time bins on the integration grid;
  - $Delta t$, the integration timestep, in seconds.

  The next entries take that path: a first PING network trained on SHD, then the
  firing-rate regulariser (the upper firing-rate bound from Cramer et al.#cite(1),
  with target $theta_u = #(r.config.rate_target_theta_u)$ and weight $s_u = #(r.config.rate_weight_s_u)$) that event-based
  benchmarks need to keep the hidden-layer rates in check, where:

  - $theta_u$, the target upper bound on a unit's firing rate;
  - $s_u$, the strength of the penalty applied above that bound.

  #reference-list((
    (
      text: [Cramer, Stradmann, Schemmel & Zenke — _The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks_. IEEE Transactions on Neural Networks and Learning Systems, 2020.],
      doi: "10.1109/TNNLS.2020.3044364",
    ),
  ))
]
