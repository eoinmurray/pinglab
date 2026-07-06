#let meta = (
  title: "Perturbations: gamma gates rates not just mean inhibition",
  date: "2026-06-02",
  description: "Overriding the I-stream of trained PING at inference shows what gates the E rate is the timing of inhibition — the rhythm — not its average level.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)


#let body = [
  == Abstract

  #link("/exp025/")[exp025]'s rate gap admits a cheap reading: PING fires less because
  the I-loop delivers more inhibition. This entry forecloses that reading and
  identifies _what_ about the I-stream is doing the suppressing — qualitatively
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

  Four perturbation families:

  + *Baseline* — no override; trained PING dynamics.
  + *Cycle-coherent jitter* — partition the trial into blocks of length
    $1 \/ f_gamma$ (≈ 28 ms at the trained operating point from
    #link("/exp041/")[exp041]). For each (trial, block), draw a single Gaussian offset
    $Delta tilde cal(N)(0, sigma^2)$ and shift every I-spike in that block by
    $Delta$. Within-burst cross-cell synchrony preserved exactly; only the
    _placement_ of each burst is perturbed. Sweep
    $sigma in {0, 1, 3, 7, 14, 21, 28, 42, 60, 100}$ ms.
  + *Phase-shuffle* — per-trial permutation $pi_b$ of the time axis applied to all
    I-cells together: $bold(s)^I_"shuf"[t, b, n] = bold(s)^I_"base"[pi_b (t), b, n]$.
    Preserves cross-cell co-firing within a timestep; destroys all phase structure.
  + *Rate-matched Poisson* — per-(trial, cell) Bernoulli with
    $p = "count"_(b,n) \/ T$. Destroys both temporal and cross-cell structure; tests
    the $g_i$ variance limit.

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Two inference-time perturbations of the trained-PING I-stream, *both holding
      mean per-cell I rate fixed* (annotated on each raster). They push the E rate in
      _opposite_ directions, which a mean-inhibition account cannot produce. *Left
      column — smear the bursts.* Per-I-cell jitter scatters the spikes within each
      burst, destroying synchrony while leaving the mean untouched; the burst
      dissolves into a continuous shunt and the E rate _falls to zero_, accuracy
      collapsing to chance. *Right column — move the bursts.* Cycle-coherent jitter
      displaces each gamma burst bodily but keeps its within-burst synchrony; the
      I-stream opens gaps and the E rate _rises_ from ≈ 8 Hz toward ≈ 50 Hz as σ
      grows, accuracy holding near 80%. Same mean inhibition, opposite outcome: what
      gates the E rate is the _timing_ of inhibition — the rhythm — not its average
      level.
    ],
  )

  === Per-I-cell jitter

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Per-I-cell jitter sweep, three seeds. Each spike receives an independent
      Gaussian offset; mean per-cell I rate is preserved exactly. *E rate (cyan,
      left axis) falls monotonically from baseline* — already halved by
      $sigma = 0.5$ ms — and is essentially zero by $sigma ≈ 5$ ms, _well below_
      $tau_"GABA" = 9$ ms. *Accuracy (red, right axis) holds at ≈ 85% up to
      $sigma = 1$ ms*, then collapses through 71% (σ = 2) and 26% (σ = 5), bottoming
      at chance (≈ 10%) by $sigma = 9$ ms. The $sigma -> oo$ asymptote is the
      rate-matched Poisson regime: E silent, accuracy at chance.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Single trial replayed at five per-cell jitter levels. At $sigma = 0$ the
      I-bursts are crisp vertical bands. *At $sigma = 1$ ms* the bursts visibly smear
      into a few-ms-wide cluster and E firing already collapses to 1.6 Hz. *At
      $sigma >= 5$ ms* the I-stream looks indistinguishable from a continuous
      low-variance shunt, and E is silenced. Per-cell jitter doesn't release E; it
      destroys the bursty structure that gave E recovery troughs in the first place.
    ],
  )

  === Cycle-coherent jitter

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      E rate (black) and accuracy (red) vs cycle-coherent jitter $sigma$, seed 42.
      Horizontal dashed lines mark the baseline rate (≈ 8 Hz) and the full
      phase-shuffle ceiling (≈ 29 Hz, the $sigma -> oo$ asymptote with within-burst
      structure destroyed). Vertical dotted line at $sigma = 1 \/ f_gamma ≈ 28$ ms —
      the predicted transition timescale.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Single trial replayed at five jitter levels (seed 42, MNIST digit 0 sample 0).
      Per-trial E rate annotated on each panel. *The I-bands stay vertical and crisp
      at every $sigma$* — within-burst synchrony is preserved exactly. What changes
      is _where_ each burst lands: at larger $sigma$ the bursts get displaced bodily
      from their phase-locked positions, opening longer gaps in the I-stream that E
      fires through.
    ],
  )
]
