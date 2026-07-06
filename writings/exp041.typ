#let meta = (
  title: "E rate is affine in gamma frequency",
  date: "2026-06-02",
  description: "Re-training PING at each τ_GABA, the per-cell E rate is affine in the gamma frequency: r_E ≈ 1.15 + 0.18·f_γ with R² ≈ 0.99.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let run = json("/artifacts/data/exp041/numbers.json")
#let fit = run.fit
#let fa = calc.round(fit.a_affine, digits: 2)
#let fp = calc.round(fit.p_affine, digits: 3)
#let fr = calc.round(fit.r2_affine, digits: 3)
#let po = calc.round(fit.p_origin, digits: 3)
#let pr = calc.round(fit.r2_origin, digits: 3)
#let fgs = run.results.map(x => x.f_gamma_hz)
#let ers = run.results.map(x => x.e_rate_hz)
#let fg_lo = calc.round(calc.min(..fgs))
#let fg_hi = calc.round(calc.max(..fgs))
#let er_lo = calc.round(calc.min(..ers), digits: 1)
#let er_hi = calc.round(calc.max(..ers), digits: 1)
#let accs = run.results.map(x => x.acc)
#let acc_lo = calc.round(calc.min(..accs))
#let acc_hi = calc.round(calc.max(..accs))

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  #link("/exp037/")[exp037] varied $tau_"GABA"$ at inference and the per-cell E rate
  tracked the gamma cycle. Does that survive _re-training_ at each $tau_"GABA"$?
  Yes. Across $tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms × 3 seeds,
  $r_E = #fa + #fp dot f_gamma$ with $R^2 = #fr$. The slope is per-cycle E-cell
  participation; the intercept is a non-rhythmic baseline; accuracy stays at ≈ #acc_lo–#acc_hi%
  across the sweep. The cycle clock constrains what the network can become.

  == Method

  *Sweep.* Six $tau_"GABA"$ values ${4.5, 6, 9, 12, 18, 27}$ ms × three seeds = 18
  networks, trained in the shared hub to the gamma standard (50 epochs on MNIST,
  Adam at $4 times 10^(-4)$, batch 256, mem-mean readout, no spike budget,
  $Delta t = 0.1$ ms, $T = 200$ ms); only $tau_"GABA"$ varies.

  *Measuring $f_gamma$.* For each cell, $f_gamma$ is the parabolic-interpolated peak
  of the Welch PSD on the per-trial population E trace; the fit (Figure 4) uses
  per-trial peak medians, which avoid the centroid bias of trial-mean PSD peaks.
  Parabolic interpolation with peak-bin values $(y_0, y_1, y_2)$:

  $
    f_gamma = "freq"["peak"] + 1/2 (y_0 - y_2)/(y_0 - 2 y_1 + y_2) dot Delta f,
    quad Delta f = 5 "Hz".
  $

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

  $
    r_E = underbrace(a, "feedforward baseline") +
    underbrace(p dot f_gamma, "cyclic contribution").
  $

  We fit this across the 18 cells.

  *Convergence.* Accuracy plateaus by ≈ epoch 15 while the E rate keeps climbing
  through training (Figure 1), so the fit uses the final-epoch rates. Fitting
  $r_E = a + p dot f_gamma$ across the 18 cells is tight, and forcing the intercept
  through zero barely loosens it, so the law is not an artefact of the free intercept:

  #table(
    columns: 4,
    [fit], [$a$ (Hz)], [$p$ (Hz/Hz)], [$R^2$],
    [affine], [#fa], [#fp], [#fr],
    [through origin], [0], [#po], [#pr],
  )

  == Results

  #figure(
    image(
      "/artifacts/data/exp041/training_curves.svg",
      width: 100%,
      alt: "Per-cell accuracy and E-rate over training epochs across the τ_GABA sweep.",
    ),
    caption: [
      Per-cell accuracy (top) and E rate (bottom) over training, one line per cell.
      Accuracy plateaus by ≈ epoch 15; the E rate keeps climbing through the 50
      epochs, so the final-epoch rates are the ones fit.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp041/psds.svg",
      width: 100%,
      alt: "Population-E power spectra by τ_GABA, gamma peak shifting with the inhibitory time constant.",
    ),
    caption: [
      Trial-mean Welch PSDs by $tau_"GABA"$; dots mark the parabolic-interpolated
      peak. The peak shifts cleanly from ≈ #fg_lo Hz at $tau_"GABA" = 27$ ms to
      ≈ #fg_hi Hz at $tau_"GABA" = 4.5$ ms, with no overlap between adjacent conditions.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp041/raster_strip.png",
      width: 100%,
      alt: "One MNIST trial through each τ_GABA network; the gamma cycle period lengthens with τ_GABA.",
    ),
    caption: [
      One MNIST trial through each network. The cycle period stretches from ≈ 20 ms
      ($f_gamma ≈ #fg_hi$ Hz) at short $tau_"GABA"$ to ≈ 50 ms ($f_gamma ≈ #fg_lo$ Hz)
      at long, the eye and the spectrum agreeing.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp041/rate_vs_fgamma.svg",
      width: 100%,
      alt: "Post-training E rate against gamma frequency; points lie on the affine fit line.",
    ),
    caption: [
      The law itself. Top: mean post-training E rate vs $f_gamma$, six clusters ×
      three seeds, error bars from seed variance; the affine fit passes through
      every error bar. Bottom: per-cluster accuracy is flat, so the rate change is
      not paid in classification.
    ],
  )
]
