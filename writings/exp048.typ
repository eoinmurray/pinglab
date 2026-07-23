#import "/.demolab/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Trained PING streams sequential digits without retraining",
  date: "2026-06-08",
  description: "A PING network trained one digit at a time classifies a stream of digits at a fraction of the time each, with no retraining: only the readout window changes.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let r = json("/artifacts/data/exp048/numbers.json")
#let rate-at(rate) = r.encoding_rate_psychometric.curve.filter(x => x.input_rate_hz == rate).at(0)
#let p05 = rate-at(0.5)
#let p2 = rate-at(2.0)
#let p5 = rate-at(5.0)
#let p10 = rate-at(10.0)

#let body = [
  == Abstract

  A PING network trained one digit at a time on 200 ms trials can classify a _stream_ of digits at a fraction of that time each, with no retraining: only the readout's integration window changes. At τ = 50 ms (≈ 2 gamma cycles) 4 of the 5 sequential MNIST digits classify correctly, the readout flipping within one cycle of each transition. A 2-D sweep over (τ, input rate) reveals a sub-cycle failure floor below ≈ 15 ms and a broad high-accuracy plateau above it. At fixed 200 ms presentation and readout windows, performance remains at chance through #p05.input_rate_hz Hz and becomes clearly informative by #p2.input_rate_hz Hz, locating the encoder's low-rate information floor.

  == Method

  Canonical #link("/exp025/")[exp025] PING baselines (1,024 excitatory + 256 inhibitory cells, trained one MNIST digit per 200 ms trial, three seeds 42, 43, 44). Everything here is *inference-only* at the trained Δt = 0.10 ms; the weights are never updated. The 2-D sweep averages over the three seeds; the single-stream demos use seed 42.

  A stream of digits, each shown for τ ms, is classified in one forward pass. The _only_ change from training is a sliding readout window:

  + *Encode* each digit as a Poisson spike train over τ ms across 784 input channels, one per pixel, firing rate proportional to pixel intensity at the chosen input rate.
  + *Concatenate* the per-digit trains into one input stream. Transitions are _hard switches_: the Poisson rate flips instantaneously at $t = k tau$, with no blending.
  + *One forward pass* of the trained network over the whole stream (no retraining), at the trained Δt.
  + *Integrate evidence* in a non-spiking output leaky-integrator, one unit per class:

    $ v_"out" (t) = beta_"out" v_"out" (t-1) + (1 - beta_"out") / (Delta t) bold(s)^E (t-1) W_"out". $

  + *Read a sliding window.* Average $v_"out"$ over the trailing τ-window and softmax it:

    $ "logits"(t) = (Delta t) / tau sum_(u=t-w+1)^(t) v_"out" (u), quad p("class", t) = "softmax"("logits"(t)), $

    Here the readout-window duration is matched exactly to the current digit's
    presentation duration: $T_"readout" = T_"presentation" = tau$, with
    $w = tau / Delta t$ timesteps. Thus a digit presented for 25, 50, or 200 ms
    is read over 25, 50, or 200 ms, respectively; the readout duration is not
    varied independently. At training the average ran over the _whole_ trial;
    the trailing matched-duration window is the single change.
  + *Predict per segment* as $arg max_c p("class"=c, t)$, read at the end of each digit's τ-window.

  Here $bold(s)^E (t)$ is the excitatory spike vector, $W_"out"$ the trained readout (unchanged), and $beta_"out" = exp(-Delta t \/ tau_"out")$ the output leak with $tau_"out" = 2$ ms. The probability trace $p("class", t)$, the network's online confidence in each digit, is what the figures plot.

  The grid sweeps τ over 10, 15, 25, 40, 50, 75, 100, 200 ms and input rate over 5, 10, 25, 50, 100, 200 Hz per channel, giving 8 × 6 cells, each 40 streams × 10 digits × 3 seeds (1200 segments per cell). It extends down to τ = 10 ms (≈ 0.36 of a gamma cycle) to resolve the sub-cycle regime.

  To resolve the encoding-rate floor below the grid, additional evaluations sweep rates from 0.01 to 3 Hz while holding both the presentation duration and readout window fixed at 200 ms. Each cell contains #(r.encoding_rate_psychometric.new_streams_per_seed) streams of #(r.encoding_rate_psychometric.digits_per_stream) digits across the same three trained seeds. The published 5–200 Hz points use the same fixed 200 ms protocol and come from the corresponding row of the grid.

  == Results

  #figure(
    image("/artifacts/data/exp048/headline_stream.png", width: 100%,
      alt: "A five-digit stream at fixed τ=50 ms: input thumbnails, E and I rasters, and the readout probability traces flipping to each new digit."),
    caption: [Five sequential digits (5, 6, 0, 7, 3) at τ = 50 ms each (≈ 2 gamma cycles per digit, 250 ms total, 25 Hz input). The gamma cadence (≈ 28 ms) is preserved across the stream; the readout flips to the new digit within one cycle of each transition and reaches near 1.0 by the segment's end. *4/5 correct* (the leading 5 is misread). The figure makes four things visible at once: sparse E firing, the I-burst clock, class-tracking readout, and re-identification within ≈ 1 cycle of each switch.],
  )

  #figure(
    image("/artifacts/data/exp048/varying_headline_stream.png", width: 100%,
      alt: "A five-digit stream where each segment has its own duration and input rate; two of the five thumbnails are boxed red for a wrong prediction."),
    caption: [A harder test: each digit gets its _own_ duration and input rate within one stream (durations 25–200 ms, rates 10–200 Hz). Thumbnail opacity ∝ input rate (faint = weak drive, bold = strong); the sliding window uses each segment's own τ, so every digit's prediction respects its presentation window. *3/5 correct* here: the two errors fall on the weakest-drive segments (a 5 at 10 Hz and a 7 at 15 Hz), the two lowest input rates in the stream.],
  )

  #figure(
    image("/artifacts/data/exp048/acc_grid_tau_rate.png", width: 100%,
      alt: "Two-panel figure combining the duration-by-input-rate accuracy heatmap with a psychometric curve measured at fixed 200-ms presentation and readout windows."),
    caption: [Temporal and encoding-rate limits of the frozen PING classifier. *(A)* Per-segment accuracy over the full duration × rate grid (3 seeds × 1200 segments per cell). The sub-cycle regime fails even under strong drive, while above one cycle duration and rate trade off along diagonal iso-accuracy contours. *(B)* Probability of a correct classification versus encoding rate with both presentation duration and readout window fixed at 200 ms. All points belong to the same psychometric curve. The dashed line marks ten-class chance and the dotted line the 25 Hz training rate. Accuracy remains on its empty-input floor through #p05.input_rate_hz Hz, is clearly informative by #p2.input_rate_hz Hz, and reaches #calc.round(100 * p5.accuracy, digits: 1)% at #p5.input_rate_hz Hz.],
  )

  The fixed-200 ms rate curve distinguishes a nonviable encoder regime from ordinary
  classification errors under weak evidence. In the variable-condition stream,
  the failed 5 received #p10.input_rate_hz Hz for 200 ms, yet that condition
  reaches #calc.round(100 * p10.accuracy, digits: 1)% across the population. Its
  error is therefore natural trial-level variation, not evidence that
  #p10.input_rate_hz Hz is intrinsically too low. The failed 7 at 15 Hz and 75 ms
  is likewise above the empty-input rate floor, although its shorter window
  supplies less total evidence. Rates below #p05.input_rate_hz Hz are not useful
  operating points; #p2.input_rate_hz Hz is the lowest clearly informative tested
  rate and #p5.input_rate_hz Hz is a practical lower bound for future sweeps.
]
