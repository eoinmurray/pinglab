#let meta = (
  title: "Perturbations: gamma gates rates not just mean inhibition",
  date: "2026-06-02",
  description: "Overriding the I-stream of trained PING at inference shows what gates the E rate is the timing of inhibition, the rhythm, not its average level.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

// Provenance (HOUSESTYLE H9): every run number below is read from the run's
// numbers.json, never hand-typed, so a re-run updates the prose automatically.
#let run = json("/artifacts/data/exp042/numbers.json")
#let cfg = run.config
#let mean(a) = a.sum() / a.len()

// Condition-level results (baseline / phase-shuffle / Poisson), averaged over seeds.
#let by_cond(cond, key) = mean(
  run.results.filter(r => r.condition == cond).map(r => r.at(key))
)
#let base_e = calc.round(by_cond("baseline", "e_rate_hz"), digits: 1)
#let shuf_e = calc.round(by_cond("phase_shuffled_i", "e_rate_hz"), digits: 1)
#let pois_acc = calc.round(by_cond("poisson_matched_i", "acc"), digits: 1)

// Sweep helpers: average a metric over seeds at a given σ.
#let cell_at(s, key) = mean(
  run.cell_jitter_sweep.filter(r => calc.abs(r.sigma_ms - s) < 0.001).map(r => r.at(key))
)
#let cyc_at(s, key) = mean(
  run.jitter_sweep.filter(r => calc.abs(r.sigma_ms - s) < 0.001).map(r => r.at(key))
)

// Per-I-cell jitter: E rate + accuracy along the collapse.
#let cell_e_half = calc.round(cell_at(0.5, "e_rate_hz"), digits: 1)
#let cell_e1 = calc.round(cell_at(1.0, "e_rate_hz"), digits: 1)
#let cell_e5 = calc.round(cell_at(5.0, "e_rate_hz"), digits: 1)
#let cell_acc05 = calc.round(cell_at(0.5, "acc"), digits: 1)
#let cell_acc1 = calc.round(cell_at(1.0, "acc"), digits: 1)
#let cell_acc2 = calc.round(cell_at(2.0, "acc"), digits: 1)
#let cell_acc5 = calc.round(cell_at(5.0, "acc"), digits: 1)
#let cell_acc9 = calc.round(cell_at(9.0, "acc"), digits: 1)

// Cycle-coherent jitter: baseline → high-σ E rise, accuracy plateau.
#let cyc_e_hi = calc.round(cyc_at(100.0, "e_rate_hz"), digits: 1)
#let cyc_acc_hi = calc.round(cyc_at(100.0, "acc"), digits: 1)

// Rate-matched anchor for the compound figure and the strict same-mean claim:
// σ = 14 ms — a single magnitude used for BOTH arms so the compound reads as one
// manipulation strength, opposite outcome. It is a measured grid point on each
// sweep; there the displaced-burst tail has barely reached the trial edge, so
// realised I is still within a few percent of baseline while the per-cell E has
// fully collapsed and the cycle-coherent E has risen well above baseline.
#let anchor_sigma = 14
#let cyc_e_anchor = calc.round(cyc_at(14.0, "e_rate_hz"), digits: 1)
#let cyc_acc_anchor = calc.round(cyc_at(14.0, "acc"), digits: 1)
#let cyc_i_anchor = calc.round(cyc_at(14.0, "i_rate_hz"), digits: 1)

// Realised (measured) I rates for the "mean inhibition held fixed" check:
// both arms hold it at the anchor; only cycle-coherent at σ = 100 ms drops it
// (finite-window truncation of the most-displaced bursts).
#let base_i = calc.round(by_cond("baseline", "i_rate_hz"), digits: 1)
#let cell_i5 = calc.round(cell_at(5.0, "i_rate_hz"), digits: 1)
#let cell_i_anchor = calc.round(cell_at(14.0, "i_rate_hz"), digits: 1)
#let cyc_i_hi = calc.round(cyc_at(100.0, "i_rate_hz"), digits: 1)
#let cyc_i_drop_pct = calc.round(100 * (base_i - cyc_i_hi) / base_i)
#let cyc_i_anchor_drop_pct = calc.round(100 * (base_i - cyc_i_anchor) / base_i)

// Gamma period 1/f_γ: the predicted transition timescale.
#let period_ms = calc.round(1000 / cfg.f_gamma_reference_hz, digits: 1)

#let body = [
  == Abstract

  #link("/exp025/")[exp025]'s rate gap admits a cheap reading: PING fires less because
  the I-loop delivers more inhibition. This entry forecloses that reading and
  identifies _what_ about the I-stream is doing the suppressing: qualitatively
  (rhythm vs mean), and quantitatively (which temporal precision is required).
  Scaffolded by #link("/ar009/")[ar009] §Leg 1 item 2.

  == Methods

  Pure inference on the trained #link("/exp025/")[exp025] PING baseline (seed 42,
  $theta_u =$ off). For each batch the I-population spike tensor
  $bold(s)^I_"base" in {0,1}^(T times B times N_I)$ is recorded from a baseline
  forward pass, then an override tensor replaces it in a second pass via the
  #link("/exp037/")[exp037] hidden-perturbation hook. The E-population experiences only
  the override I-stream through $W^(I E)$; the readout consumes the perturbed E
  spikes.

  Every perturbation only _moves_ spikes in time (or, for Poisson, redraws at the
  matched count) — none adds or removes them — so the mean per-cell I rate is
  matched to baseline by construction: exactly for phase-shuffle, and to within
  ≈ #cyc_i_anchor_drop_pct% for the jitter families across the range where each
  result is read. The one exception is cycle-coherent jitter at the largest $sigma$:
  a Gaussian block offset with $sigma = 100$ ms displaces part of each burst past
  the ends of the fixed presentation window, where it is clamped and lost, so the
  _realised_ I rate falls to #cyc_i_hi Hz (#cyc_i_drop_pct% below the #base_i Hz
  baseline). Realised I is therefore plotted on every sweep, and the strict
  same-mean-inhibition comparison (the compound figure below) is anchored at
  $sigma = #anchor_sigma$ ms, where realised I is still within
  #cyc_i_anchor_drop_pct% of baseline on both arms.

  Five perturbation families:

  + *Baseline*: no override; trained PING dynamics.
  + *Cycle-coherent jitter*: partition the trial into blocks of length
    $1 \/ f_gamma$ (≈ #period_ms ms at the trained operating point from
    #link("/exp041/")[exp041]). For each (trial, block), draw a single Gaussian offset
    $Delta tilde cal(N)(0, sigma^2)$ and shift every I-spike in that block by
    $Delta$. Within-burst cross-cell synchrony is preserved exactly; only the
    _placement_ of each burst is perturbed. Sweep
    $sigma in {0, 1, 3, 7, 14, 21, 28, 42, 60, 100}$ ms.
  + *Per-I-cell jitter*: draw an independent Gaussian offset
    $Delta_(b,n,k) tilde cal(N)(0, sigma^2)$ for _every_ I-spike and shift each
    spike by its own offset. Destroys within-burst cross-cell synchrony while keeping
    the mean per-cell rate; the within-burst counterpart of cycle-coherent jitter.
    Sweep $sigma in {0, 0.5, 1, 2, 5, 9, 14, 21, 50}$ ms.
  + *Phase-shuffle*: per-trial permutation $pi_b$ of the time axis applied to all
    I-cells together: $bold(s)^I_"shuf"[t, b, n] = bold(s)^I_"base"[pi_b (t), b, n]$.
    Preserves cross-cell co-firing within a timestep; destroys all phase structure.
  + *Rate-matched Poisson*: per-(trial, cell) Bernoulli with
    $p = "count"_(b,n) \/ T$. Destroys both temporal and cross-cell structure; tests
    the $g_i$ variance limit.

  == Results

  #figure(
    image(
      "/artifacts/data/exp042/rhythm_compound.png",
      width: 100%,
      alt: "A two-by-two panel; both top rasters use the same jitter magnitude, sigma 14 ms, differing only in the kind of jitter. Top row: two single-trial rasters of trained PING (E spikes black, I spikes red). Top left, per-I-cell jitter, where the I-bursts have dissolved into continuous asynchronous firing and E is silent. Top right, cycle-coherent jitter, where the I-bursts stay sharp but are displaced and E firing appears in the gaps; both panels have near-identical realised I rates. Bottom row: two twin-axis line plots of hidden E rate (black diamonds) and accuracy (red squares) versus jitter sigma, each with a grey realised mean I-rate trace. Bottom left (per-cell) E and accuracy fall to near zero while realised I stays flat near 53 Hz; bottom right (cycle-coherent) E rate rises and accuracy stays high, while the grey realised-I trace holds flat then droops at the largest sigma.",
    ),
    caption: [
      Two inference-time perturbations of the trained-PING I-stream, *both holding
      the mean per-cell I rate fixed at ≈ #base_i Hz*, push the E rate in _opposite_
      directions, which a mean-inhibition account cannot produce. *Both columns use
      the same jitter magnitude, σ = #anchor_sigma ms* — only the _kind_ of jitter
      differs. *Left column, smear the bursts:* per-I-cell jitter (realised I
      #cell_i_anchor Hz) scatters the spikes within each burst, destroying synchrony
      while leaving the mean untouched; the burst dissolves into a continuous shunt,
      the E rate falls to zero, and accuracy collapses toward chance (#pois_acc% at the
      Poisson limit). *Right column, move the bursts:* cycle-coherent jitter (realised
      I #cyc_i_anchor Hz — within #cyc_i_anchor_drop_pct% of the left column) displaces
      each gamma burst bodily but keeps its within-burst synchrony; the I-stream opens
      gaps and the E rate _rises_ from #base_e Hz at baseline to #cyc_e_anchor Hz,
      accuracy holding near #cyc_acc_anchor%. Same jitter magnitude, same mean
      inhibition (both ≈ #base_i Hz), opposite outcome: what gates the E rate is the
      _timing_ of inhibition, the rhythm, not its average level. The cycle-coherent
      rise continues past the full phase-shuffle level to #cyc_e_hi Hz by σ = 100 ms
      (bottom-right sweep), but there the finite trial window truncates the
      most-displaced bursts and realised I falls #cyc_i_drop_pct%, so the strict
      rate-matched reading is taken at σ = #anchor_sigma ms; see Methods.
    ],
  )

  === Per-I-cell jitter

  #figure(
    image(
      "/artifacts/data/exp042/cell_jitter_sweep.svg",
      width: 100%,
      alt: "Twin-axis line plot: hidden E rate (black diamonds, left axis) and test accuracy (red squares, right axis) versus per-I-cell jitter sigma in milliseconds on a symlog axis. Both fall steeply from baseline at small sigma; E rate is essentially zero by 5 ms and accuracy reaches chance by 9 ms. A grey line shows the realised mean I rate, flat near 53 Hz across the whole sweep and dipping only slightly to about 49 Hz by 50 ms.",
    ),
    caption: [
      Per-I-cell jitter sweep, three seeds. Each spike receives an independent
      Gaussian offset; mean per-cell I rate is preserved exactly. *E rate (black
      diamonds, left axis) falls monotonically from baseline* (#base_e Hz), already
      more than halved to #cell_e_half Hz by $sigma = 0.5$ ms and essentially zero
      (#cell_e5 Hz) by $sigma ≈ 5$ ms, below $tau_"GABA" = 6$ ms. *Accuracy (red
      squares, right axis) holds at ≈ #cell_acc05% up to $sigma = 0.5$ ms*, then
      collapses through #cell_acc1% (σ = 1), #cell_acc2% (σ = 2) and #cell_acc5%
      (σ = 5), bottoming at chance (#cell_acc9%) by $sigma = 9$ ms. The grey trace is
      the realised mean I rate, held flat near #base_i Hz across the sweep: the E
      collapse happens under matched inhibition. The $sigma -> oo$ asymptote is the
      rate-matched Poisson regime: E silent, accuracy at chance.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp042/cell_jitter_raster_strip.png",
      width: 100%,
      alt: "Five stacked single-trial rasters of trained PING under per-I-cell jitter at sigma 0, 1, 5, 9 and 50 ms; E spikes black, I spikes red. Crisp vertical I-bursts at sigma 0 smear at sigma 1 and dissolve into continuous asynchronous I firing by sigma 5, while E firing goes silent.",
    ),
    caption: [
      Single trial replayed at five per-cell jitter levels. At $sigma = 0$ the
      I-bursts are crisp vertical bands. *At $sigma = 1$ ms* the bursts visibly smear
      into a few-ms-wide cluster and E firing already collapses to the low single
      digits (per-trial E annotated on each panel; sweep mean #cell_e1 Hz). *At
      $sigma >= 5$ ms* the I-stream looks indistinguishable from a continuous
      low-variance shunt, and E is silenced. Per-cell jitter doesn't release E; it
      destroys the bursty structure that gave E its recovery troughs in the first
      place.
    ],
  )

  === Cycle-coherent jitter

  #figure(
    image(
      "/artifacts/data/exp042/jitter_sweep.svg",
      width: 100%,
      alt: "Twin-axis line plot on a symlog sigma axis: hidden E rate (black diamonds, left axis) rises monotonically from about 9 Hz to 66 Hz as cycle-coherent jitter sigma grows, while test accuracy (red squares, right axis) declines gently from about 91 to 82 percent. A grey line shows the realised mean I rate: flat near 53 Hz up to about sigma 14 ms, then drooping to about 40 Hz by sigma 100 ms.",
    ),
    caption: [
      E rate (black diamonds) and accuracy (red squares) vs cycle-coherent jitter
      $sigma$, three seeds. As $sigma$ grows the displaced bursts open wider gaps and
      the E rate climbs from baseline (#base_e Hz) _past_ the full phase-shuffle level
      (#shuf_e Hz, the reference with within-burst structure destroyed) to #cyc_e_hi Hz
      by $sigma = 100$ ms, with the sharpest rise near the predicted transition
      timescale $sigma = 1 \/ f_gamma$ ≈ #period_ms ms; accuracy declines only gently,
      holding near #cyc_acc_hi%. The grey trace is the realised mean I rate: it holds
      within #cyc_i_anchor_drop_pct% of baseline through $sigma ≈ #anchor_sigma$ ms —
      where the E rate has already risen to #cyc_e_anchor Hz — then droops to
      #cyc_i_hi Hz (#cyc_i_drop_pct% below baseline) by $sigma = 100$ ms, as the finite
      trial window truncates the most-displaced bursts. The strict same-mean-inhibition
      comparison is read at the smaller $sigma$, where the rate is matched and the E
      rise is already unambiguous.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp042/jitter_raster_strip.png",
      width: 100%,
      alt: "Five stacked single-trial rasters of trained PING under cycle-coherent jitter at sigma 0, 7, 14, 28 and 100 ms; E spikes black, I spikes red. The red I-bursts stay sharp and vertical at every sigma but shift position, while E firing (black) grows denser and fills the widening gaps as sigma increases.",
    ),
    caption: [
      Single trial replayed at five jitter levels ($sigma = 0, 7, 14, 28, 100$ ms;
      seed 42, MNIST digit 0 sample 0). Per-trial E rate annotated on each panel.
      *The I-bands stay vertical and crisp at every $sigma$:* within-burst synchrony
      is preserved exactly. What changes is _where_ each burst lands: at larger
      $sigma$ the bursts are displaced bodily from their phase-locked positions,
      opening longer gaps in the I-stream that E fires through, and the E rate climbs
      accordingly.
    ],
  )

  == Next steps

  *Toroidal (wrapped) jitter, to extend the rate-matched range and disentangle
  release from truncation.* The strict same-mean-inhibition claim is now anchored at
  $sigma = #anchor_sigma$ ms, where realised I holds within #cyc_i_anchor_drop_pct% of
  baseline on both arms and the E rate has already risen to #cyc_e_anchor Hz — the
  qualitative result stands on rate-matched ground. Beyond $sigma ≈ 30$ ms, though,
  the cycle-coherent E-rate rise and the realised-I droop become confounded: some of
  the extra E firing is genuine gap-opening, and some is simply less inhibition
  delivered, because the finite window clamps and loses the most-displaced bursts.
  Wrapping each block offset modulo the trial length would restore exact spike-count
  preservation at every $sigma$ and separate the two, at the cost of re-injecting a
  wrapped burst at the opposite trial edge — a phase artifact of its own, so the
  wrapped sweep is a robustness check, not a replacement. The prediction: if the
  wrapped E rate still climbs past the phase-shuffle level (#shuf_e Hz), the release at
  large $sigma$ is real; if it flattens there, part of the #cyc_e_hi Hz overshoot at
  $sigma = 100$ ms was the truncation. Either way the anchored rhythm-vs-mean
  conclusion is unaffected.
]
