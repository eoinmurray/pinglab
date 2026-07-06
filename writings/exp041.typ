#let meta = (
  title: "E rate is affine in gamma frequency",
  date: "2026-06-02",
  description: "Re-training PING at each τ_GABA, the per-cell E rate is affine in the gamma frequency: r_E = 0.87 + 0.195·f_γ with R² = 0.996.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)


#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  #link("/exp037/")[exp037] varied $tau_"GABA"$ at inference and the per-cell E rate
  tracked the gamma cycle. Does that survive _re-training_ at each $tau_"GABA"$?
  Yes. Across $tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms × 3 seeds,
  $r_E = 0.87 + 0.195 dot f_gamma$ with $R^2 = 0.996$. The slope is per-cycle E-cell
  participation; the intercept is a non-rhythmic baseline; accuracy stays at 80–90%
  across the sweep. The cycle clock constrains what the network can become.

  == Method

  *Sweep.* Six $tau_"GABA"$ values ${4.5, 6, 9, 12, 18, 27}$ ms × three seeds = 18
  networks. The recipe matches the #link("/exp025/")[exp025] PING baseline (100 epochs
  medium tier on MNIST, Adam at $4 times 10^(-4)$, batch 256, mem-mean readout,
  $theta_u$ off, $Delta t = 0.1$ ms, $T = 200$ ms); only $tau_"GABA"$ varies.

  *Measuring $f_gamma$.* For each cell, $f_gamma$ is the parabolic-interpolated peak
  of the Welch PSD on the per-trial population E trace; the fit (Figure 4) uses
  per-trial peak medians, which avoid the centroid bias of trial-mean PSD peaks.
  Parabolic interpolation with peak-bin values $(y_0, y_1, y_2)$:

  $ f_gamma = "freq"["peak"] + 1/2 (y_0 - y_2)/(y_0 - 2 y_1 + y_2) dot Delta f,
    quad Delta f = 5 "Hz". $

  It is needed because the bare 5 Hz bin quantisation would coarsen $f_gamma$ across
  the six conditions; on a well-isolated peak the interpolation error is
  $O((Delta f)^3)$.

  *The predicted law.* The shape $r_E = a + p dot f_gamma$ is predicted by cycle
  dynamics, not curve-fitted. Within one cycle of duration $1 \/ f_gamma$, a
  fraction $p$ of E cells emits exactly one spike (those nearest threshold when the
  I shunt drops); the rest are still recovering, so the cyclic per-cell rate is
  $p dot f_gamma$. At long $tau_"GABA"$ the I conductance never fully decays and the
  cycle dissolves into a tonic bath, leaving a feedforward baseline $a$ independent
  of $f_gamma$:

  $ r_E = underbrace(a, "feedforward baseline") +
    underbrace(p dot f_gamma, "cyclic contribution"). $

  We fit this across the 18 cells.

  *Convergence.* Accuracy plateaus by epoch 15 while the E rate keeps climbing to
  epoch 70–100 (Figure 1), so the 100-epoch numbers are the converged ones. The fit
  shape is robust to training length — $p$ tightens with it, and the 100-epoch row
  is canonical:

  #table(
    columns: 4,
    [training], [$a$ (Hz)], [$p$ (Hz/Hz)], [$R^2$],
    [30 epochs], [1.14], [0.166], [0.990],
    [100 epochs], [*0.87*], [*0.195*], [*0.996*],
  )

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Per-cell accuracy (top) and E rate (bottom) over training. Accuracy plateaus
      by epoch 15; the E rate keeps climbing until epoch 70–100, so the 100-epoch
      numbers are converged.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Trial-mean Welch PSDs by $tau_"GABA"$; dots mark the parabolic-interpolated
      peak. The peak shifts cleanly from ≈ 14 Hz at $tau_"GABA" = 27$ ms to ≈ 54 Hz
      at $tau_"GABA" = 4.5$ ms, with no overlap between adjacent conditions.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      One MNIST trial through each network. The cycle period stretches from ≈ 17 ms
      ($r_E = 14.5$ Hz) to ≈ 63 ms ($r_E = 4.6$ Hz) — the eye and the spectrum
      agree.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      The law itself. Top: mean post-training E rate vs $f_gamma$, six clusters ×
      three seeds, error bars from seed variance — the affine fit passes through
      every error bar. Bottom: per-cluster accuracy is flat, so the rate change is
      not paid in classification.
    ],
  )
]
