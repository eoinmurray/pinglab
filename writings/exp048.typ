#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Trained PING streams sequential digits without retraining",
  date: "2026-06-08",
  description: "A PING network trained one digit at a time classifies a stream of digits at a fraction of the time each, with no retraining — only the readout window changes.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)


#let body = [
  == Abstract

  A PING network trained one digit at a time on 200 ms trials can classify a _stream_ of digits at a fraction of that time each, with no retraining — only the readout's integration window changes. At τ = 50 ms (≈ 2 gamma cycles) all 5 sequential MNIST digits classify correctly, the readout flipping within one cycle of each transition; a 2-D sweep over (τ, input rate) reveals a sub-cycle failure floor below ≈15 ms and a broad high-accuracy plateau above it, where short presentation time and weak drive trade off along iso-accuracy diagonals.

  == Method

  Canonical #link("/exp025/")[exp025] PING baselines — 1,024 excitatory + 256 inhibitory cells, trained one MNIST digit per 200 ms trial, three seeds (42, 43, 44). Everything here is *inference-only* at the trained Δt = 0.10 ms — the weights are never updated. The 2-D sweep averages over the three seeds; the single-stream demos use seed 42.

  A stream of digits, each shown for τ ms, is classified in one forward pass. The _only_ change from training is a sliding readout window:

  + *Encode* each digit as a Poisson spike train over τ ms — 784 input channels, one per pixel, firing rate proportional to pixel intensity at the chosen input rate.
  + *Concatenate* the per-digit trains into one input stream. Transitions are _hard switches_ — the Poisson rate flips instantaneously at $t = k tau$, with no blending.
  + *One forward pass* of the trained network over the whole stream (no retraining), at the trained Δt.
  + *Integrate evidence* in a non-spiking output leaky-integrator, one unit per class:

    $ v_"out" (t) = beta_"out" v_"out" (t-1) + (1 - beta_"out") / (Delta t) bold(s)^E (t-1) W_"out". $

  + *Read a sliding window.* Average $v_"out"$ over the trailing τ-window and softmax it:

    $ "logits"(t) = (Delta t) / tau sum_(u=t-w+1)^(t) v_"out" (u), quad p("class", t) = "softmax"("logits"(t)), $

    At training the average ran over the _whole_ trial; the trailing window is the single change.
  + *Predict per segment* as $arg max_c p("class"=c, t)$, read at the end of each digit's τ-window.

  Here $bold(s)^E (t)$ is the excitatory spike vector, $W_"out"$ the trained readout (unchanged), and $beta_"out" = exp(-Delta t \/ tau_"out")$ the output leak with $tau_"out" = 2$ ms. The probability trace $p("class", t)$ — the network's online confidence in each digit — is what the figures plot.

  The grid sweeps τ over 10, 15, 25, 40, 50, 75, 100, 200 ms and input rate over 5, 10, 25, 50, 100, 200 Hz per channel — 8 × 6 cells, each 40 streams × 10 digits × 3 seeds (1200 segments per cell). It extends down to τ = 10 ms (≈ 0.36 of a gamma cycle) to resolve the sub-cycle regime.

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Five sequential digits (5, 6, 0, 7, 3) at τ = 50 ms each — ≈ 2 gamma cycles per digit, 250 ms total, 25 Hz input. The gamma cadence (≈ 28 ms) is preserved across the stream; the readout flips to the new digit within one cycle of each transition and reaches near 1.0 by the segment's end. *5/5 correct.* The figure makes four things visible at once: sparse E firing, the I-burst clock, class-tracking readout, and re-identification within ≈ 1 cycle of each switch.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [A harder test: each digit gets its _own_ duration and input rate within one stream (durations 25–200 ms, rates 10–200 Hz). Thumbnail opacity ∝ input rate (faint = weak drive, bold = strong); the sliding window uses each segment's own τ, so every digit's prediction respects its presentation window. *5/5 correct.*],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Per-segment accuracy over the full grid (3 seeds × 1200 segments per cell). Three things stand out. *(1) Sub-cycle failure below τ ≈ 15 ms:* at τ = 10 ms even 200 Hz input reaches only 59% — the network cannot classify within less than one cycle regardless of drive, the cleanest evidence that the *gamma cycle is the temporal quantum* of its classification ability. *(2) Above one cycle, accuracy ≈ f(τ · rate):* iso-accuracy contours run diagonal, so τ and input rate substitute for each other — more drive compensates for shorter presentation and vice versa. *(3) The trained operating point* (200 ms, 25 Hz) sits at 88%, interior to the plateau — the corners gain only ≈3 pp, so there is headroom in both directions.],
  )
]
