#let meta = (
  title: "What the Spiking Heidelberg Digits look like",
  date: "2026-07-11",
  description: "A first look at the SHD event-based audio benchmark before training on it: raw spike rasters of spoken digits across 700 cochlear channels, one per class and several within a class.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)


#let body = [
  == Introduction

  The gamma-gated-sparsity collection trains the #link("/ar003/")[COBANet] on
  static MNIST images that are Poisson-encoded into spikes. #link("https://doi.org/10.1109/TNNLS.2020.3044364")[Spiking Heidelberg Digits (SHD)]
  is a different animal: the data is _already_ spikes. Each sample is a spoken
  digit passed through a model of the inner ear, so it arrives as a list of
  events — a spike time and the cochlear channel that fired — with no image and
  no dense array anywhere. Before training a network on it, this entry just
  looks at the raw data.

  == Data

  SHD is $N = #[8156]$ training utterances (a held-out test split adds more),
  each a spoken digit recorded from several speakers and converted to spikes by
  a #link("https://doi.org/10.1109/TNNLS.2020.3044364")[Cramer et al.] cochlea
  model. Every utterance is a set of events ${(t_k, u_k)}$, where:

  - $t_k$, the *spike time* of event $k$, in seconds (utterances run ≈ 0.7–1.0 s);
  - $u_k in {0, ..., 699}$, the *cochlear channel* that fired — a place code for
    frequency, low channel ≈ low pitch, high channel ≈ high pitch;
  - so an utterance carries $C = 700$ input channels and a median of ≈ 7600
    events (range ≈ 2400–14900), and there are 20 classes: the digits 0–9 spoken
    in German (labels 0–9) then English (labels 10–19).

  Nothing here runs the network — the rasters are drawn straight from the raw
  HDF5. Binning these events onto the model's integration grid (a
  $(T_"steps", 700)$ spike tensor at the run's $Delta t$) is the trainer's job,
  not this entry's.

  == Results

  #figure(
    image("/artifacts/data/exp059/class_gallery.png", width: 100%,
      alt: "A 4-by-5 grid of spike rasters, one spoken digit per panel, time on the x-axis and cochlear channel on the y-axis; each digit shows a distinct time-frequency pattern."),
    caption: [
      One utterance per class (German 0–9, then English 0–9), each a raster of
      time (ms) against cochlear channel. *What we expect.* If SHD is learnable,
      different words should paint different time–frequency signatures. *What we
      see.* They do: a compact high-channel onset for _zwei_, a long two-lobe
      sweep for _sieben_/_seven_, a low-channel tail on _sechs_. A diffuse haze
      of isolated events sits under every panel — the cochlea model's spontaneous
      background firing, which the classifier has to see past.
    ],
  )

  #figure(
    image("/artifacts/data/exp059/within_class_spread.png", width: 100%,
      alt: "Four spike rasters of the digit null side by side; all share a broad descending contour but differ in density, duration, and detail from utterance to utterance."),
    caption: [
      Four different utterances of class 0, _null_ — the within-class spread.
      *What we expect.* Different speakers saying the same word should share a
      family resemblance while differing in the particulars. *What we see.*
      Exactly that: all four carry the same broad high-channel onset falling into
      a mid-channel body, but they differ in duration, event density, and the
      fine structure of the low-channel tail. This speaker variability is what a
      classifier trained on SHD must generalise over.
    ],
  )

  == Next steps

  The data is well-behaved and visibly class-separable, so it is worth training
  on. The trainer already accepts it — _tool.py train --dataset shd_ bins these
  events to spikes at the run's $Delta t$ and feeds them to the PING network. The
  next entries take that path: a first PING trained on SHD, then the firing-rate
  regulariser (Cramer et al.'s $theta_u = 100$, $s_u = 0.06$) that event-based
  benchmarks need to keep the hidden rates in check.
]
