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
  spikes. Mean per-cell I rate is matched to the baseline to four decimals across
  every perturbation.

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
      alt: "A two-by-two panel. Top row: two single-trial rasters of trained PING (E spikes black, I spikes red). Top left, per-I-cell jitter at sigma 5 ms, where the I-bursts have dissolved into continuous asynchronous firing and E is silent. Top right, cycle-coherent jitter at sigma 100 ms, where the I-bursts stay sharp but are displaced and dense E firing fills the gaps. Bottom row: two twin-axis line plots of hidden E rate (black diamonds) and accuracy (red squares) versus jitter sigma. Bottom left (per-cell) both fall to near zero; bottom right (cycle-coherent) E rate rises while accuracy stays high.",
    ),
    caption: [
      Two inference-time perturbations of the trained-PING I-stream, *both holding
      mean per-cell I rate fixed*, push the E rate in _opposite_ directions, which a
      mean-inhibition account cannot produce. *Left column, smear the bursts:*
      per-I-cell jitter (σ = 5 ms shown) scatters the spikes within each burst,
      destroying synchrony while leaving the mean untouched; the burst dissolves into
      a continuous shunt, the E rate falls to zero, and accuracy collapses toward
      chance (#pois_acc% at the Poisson limit). *Right column, move the bursts:*
      cycle-coherent jitter (σ = 100 ms shown) displaces each gamma burst bodily but
      keeps its within-burst synchrony; the I-stream opens gaps and the E rate _rises_
      from #base_e Hz at baseline to #cyc_e_hi Hz, accuracy holding near #cyc_acc_hi%.
      Same mean inhibition, opposite outcome: what gates the E rate is the _timing_ of
      inhibition, the rhythm, not its average level.
    ],
  )

  === Per-I-cell jitter

  #figure(
    image(
      "/artifacts/data/exp042/cell_jitter_sweep.svg",
      width: 100%,
      alt: "Twin-axis line plot: hidden E rate (black diamonds, left axis) and test accuracy (red squares, right axis) versus per-I-cell jitter sigma in milliseconds. Both fall steeply from baseline near zero sigma; E rate is essentially zero by 5 ms and accuracy reaches chance by 9 ms. Horizontal dashed lines mark baseline 9.1 Hz and the rate-matched Poisson floor; a dotted vertical line marks tau_GABA = 6 ms.",
    ),
    caption: [
      Per-I-cell jitter sweep, three seeds. Each spike receives an independent
      Gaussian offset; mean per-cell I rate is preserved exactly. *E rate (black
      diamonds, left axis) falls monotonically from baseline* (#base_e Hz), already
      more than halved to #cell_e_half Hz by $sigma = 0.5$ ms and essentially zero
      (#cell_e5 Hz) by $sigma ≈ 5$ ms, below $tau_"GABA" = 6$ ms. *Accuracy (red
      squares, right axis) holds at ≈ #cell_acc05% up to $sigma = 0.5$ ms*, then
      collapses through #cell_acc1% (σ = 1), #cell_acc2% (σ = 2) and #cell_acc5%
      (σ = 5), bottoming at chance (#cell_acc9%) by $sigma = 9$ ms. The
      $sigma -> oo$ asymptote is the rate-matched Poisson regime: E silent, accuracy
      at chance.
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
      alt: "Twin-axis line plot on a symlog sigma axis: hidden E rate (black diamonds, left axis) rises monotonically from about 9 Hz to 66 Hz as cycle-coherent jitter sigma grows, while test accuracy (red squares, right axis) declines gently from about 91 to 82 percent. Horizontal dashed lines mark baseline 9.1 Hz and the phase-shuffle level 25.9 Hz; a dotted vertical line marks 1 over f_gamma = 22.8 ms.",
    ),
    caption: [
      E rate (black diamonds) and accuracy (red squares) vs cycle-coherent jitter
      $sigma$, three seeds. Horizontal dashed lines mark the baseline rate (#base_e Hz)
      and the full phase-shuffle level (#shuf_e Hz, the reference with within-burst
      structure destroyed); the vertical dotted line sits at $sigma = 1 \/ f_gamma$
      ≈ #period_ms ms, the predicted transition timescale. As $sigma$ grows the
      displaced bursts open wider gaps and the E rate climbs _past_ the phase-shuffle
      level to #cyc_e_hi Hz by $sigma = 100$ ms, while accuracy declines only gently,
      holding near #cyc_acc_hi%.
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
]
