#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "A PING rhythmicity metric",
  date: "2026-06-15",
  description: "A single bounded scalar — the lobe–trough contrast of the spike-time autocorrelation — for how rhythmic a spiking network is, made rate-invariant by private per-cell input.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)


#let body = [
  == Abstract

  A single bounded scalar for how rhythmic a spiking network is — the *lobe–trough contrast* of its spike-time autocorrelation, $("lobe" - "trough") \/ ("lobe" + "trough") in [0,1)$. Driving each excitatory cell with its own private Poisson input makes the metric rate-invariant by construction; across untrained PING networks it reads 0 along the COBA edges and rises smoothly to 0.98 through the PING interior — a rankable gradient that tracks the gamma-gated collapse of the E firing rate from 95 to 3 Hz.

  == Methods

  The networks here are untrained PING populations driven by external input; the rhythmicity metric is read off the resulting excitatory raster. Write $r(t)$ for the binned population spike count ($n$ bins of width $Delta t$) and $ell$ for the lag. The metric is read off in a fixed sequence of steps:

  + *Drive each E cell with its own private Poisson channel.* Every excitatory cell receives an independent homogeneous Poisson spike train at 100 Hz through a one-to-one identity input weight — there is no shared, dense $W_"in"$ projection, so no two cells share an input channel. Private input removes the input-driven spike coincidence that would otherwise inflate the metric at low firing, which is what makes the contrast rate-invariant by construction (Figures 4–5) with no post-hoc correction.
  + *Bin to a population count.* Sum spikes across cells into one count per $Delta t$ bin, giving the population trace $r(t)$ of length $n$.
  + *Raw autocorrelation.* Form the lag product $sum_t r(t) r(t+ell)$, which counts spike pairs separated by lag $ell$. All lags are computed at once in $O(n log n)$ via the Wiener–Khinchin route — zero-pad $r$, take its FFT, multiply by the conjugate, inverse-transform — rather than the $O(n^2)$ direct sum.
  + *Correct for finite overlap.* Only $n-ell$ bin-pairs exist at lag $ell$ (the last $ell$ samples have no partner), so the raw sum tapers toward zero with lag simply from running out of overlap; dividing by that per-lag overlap $n-ell$ converts it to the _average_ product per available pair and flattens the taper.
  + *Set the chance level.* Divide by the mean rate squared $⟨ r ⟩^2$ so rate-matched independent firing sits at $A = 1$, and drop the self-paired zero lag. The result is the normalised autocorrelogram

    $ A(ell) = 1 / (⟨ r ⟩^2) 1 / (n - ell) sum_t r(t) r(t+ell). $

  + *Locate the Mexican hat.* A rhythmic $A(ell)$ has a "Mexican-hat" profile: a central lobe above 1 (spikes recur a cycle apart) flanked by a dip below 1 where firing is suppressed between volleys. Scanning out from zero lag, take the trough as the first local minimum of a lightly smoothed $A(ell)$ — it falls near the half-period — and the lobe as the highest point at a shorter lag. Both searches start one bin past zero, so the self-paired zero-lag value dropped in step 5 is excluded from the lobe height: the lobe is read from the first real lag onward, never the trivial self-correlation.
  + *Read the contrast.* The metric is the *lobe–trough contrast*

    $ "contrast" = ("lobe" - "trough") / ("lobe" + "trough") in [0, 1), $

    zero when lobe equals trough (no structure) and approaching 1 as the trough goes silent. It is bounded by construction — no trough floor needed.

  In words: $A(ell)$ is how much more (or less) likely a spike is to be followed by another one $ell$ ms later than under independent firing, with $A = 1$ the chance floor. A central lobe above 1 says spikes _cluster_ in volleys; a trough below 1 near the half-period says firing is _suppressed between_ them. The contrast is *0* when the spikes carry no such structure (asynchronous) and approaches *1* as sharp volleys separate against near-silence; because it reads the _shape_ of $A(ell)$, it registers a rhythm whether or not its frequency holds still.

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [The anchor result: gamma switches on smoothly across the recurrent-weight plane, read as scalar maps over the $W_(E I) times W_(I E)$ grid (untrained networks, private per-cell Poisson input) with example rasters beneath. *Top* — three per-cell summaries, every cell labelled. *E rate* is high along both zero edges (the loop is broken, E fires at the input-driven ≈95 Hz) and gated down through the interior; *I rate* is silent where $W_(E I) = 0$, runs away along $W_(I E) = 0$ (clipped so the interior is legible), and is controlled once the loop closes; *lobe–trough contrast* (the rhythm scored 0–1) reads exactly 0 along both COBA edges and rises smoothly toward strong coupling, with three points marked along the $W_(I E) = 2 W_(E I)$ diagonal. *Bottom* — E/I rasters (E black, I red above) at those points: *A* the fully-off origin ($W_(E I) = W_(I E) = 0$), asynchronous with no I; *B* weak coupling, emerging volleys (contrast 0.27, below the half-way mark); *C* strong coupling, sharp volleys (contrast 0.98). E rate falling, I rate rising, and contrast rising are three readings of the same loop engaging. The full per-cell detail — every raster and autocorrelogram — is in the figures below; the rate-fairness behind the contrast metric is in Figures 4–5.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [E/I rasters for a 6×6 subset of the grid (every other cell; the heatmaps below use all 121). The two zero edges are the controls: $W_(E I) = 0$ (left column) leaves I silent and E asynchronous; $W_(I E) = 0$ (bottom row) lets I fire but not inhibit E, so both stay dense — neither is rhythmic. The interior shows clear gamma volleys (E black, I red) that sharpen as either weight grows — the rhythm the contrast (Figure 1) scores.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [The E-population autocorrelogram $A(ell)$ for a 6×6 subset of the grid (lag 0–50 ms; dotted line = chance, $A = 1$), with the located lobe (▲) and trough (▼) marked. The two zero edges are flat at 1 — asynchronous firing, no structure. Through the interior the Mexican hat emerges: a sharp central lobe at ≈1 ms over a trough near the half-period, with a secondary peak at the full period further out. The contrast in Figure 1 is exactly this lobe-versus-trough read off as one number.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [The rate-invariance test that justifies the private-input choice. Each line is a *non-rhythmic null network* (no inhibitory loop, no rhythm at any drive) scanned over input rate; a rate-invariant metric should read ≈0 everywhere. With *private input* (black) it does — flat at ≤0.04 across all firing rates. With *shared input* (grey dashed) it instead *climbs to 1.00 as firing thins*: cells sharing input channels fire coincidentally, and the metric reads that as rhythm. The real PING cells (red) sit well above the private-input null, so their contrast is genuine — no correction needed.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Why shared input fails and private input does not, at matched low firing rates. *Top (shared input):* coincident spikes from shared channels leave a central peak over a shallow dip — a spurious hat with no inhibitory loop behind it — and the metric marks a lobe (▲) and trough (▼) and reports a non-zero contrast. *Bottom (private input):* the same firing rates, but with one channel per cell the central peak is gone; $A(ell)$ is flat shot-noise around chance and the contrast collapses to ≈0. Same rate, same spike counts — the only difference is whether cells share input.],
  )

  == The onset over its mean-field bifurcation

  The turn-on above is _what_ the network does; the #link("/exp033/")[exp033] 4D conductance mean-field is _why_. This final section stacks the two into one manuscript figure — the empirical maps and example rasters directly over the mean-field bifurcation that predicts them. It recomputes the #link("/exp033/")[exp033] numerics (Hopf crossing, hysteresis sweep, gamma-vs-$tau_"GABA"$) and reuses this notebook's own map and raster rendering, so restyling Figure 1 propagates here automatically — no figures are copied. (This is the anchor that was formerly its own entry.)

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Panels are lettered *A–I* in reading order. *Top (empirics, A–C).* Across the $W_(E I) times W_(I E)$ plane the E rate falls (*A*), the I rate rises (*B*), and the lobe–trough contrast rises (*C*) — three readings of one loop engaging, 0 along both COBA edges and smoothly up toward strong coupling. The contrast map circles three points along the $W_(I E) = 2 W_(E I)$ diagonal, shown as rasters *D/E/F*: the fully-off origin (*D*), emerging volleys (*E*), and sharp gamma volleys (*F*). *Bottom (theory, G–I, #link("/exp033/")[exp033]).* The same onset from a 4D conductance mean-field calibrated from the biophysics: a complex-conjugate eigenvalue pair crosses into the right half-plane at $I^* = 0.60$ nA (a Hopf, *G*), the amplitude rises continuously with coinciding up/down branches (supercritical and reversible, not a hard switch, *H*), and the predicted gamma frequency falls with $tau_"GABA"$ in step with the #link("/exp041/")[exp041] spiking measurement (*I*). The smooth empirical turn-on is exactly what a supercritical Hopf predicts.],
  )
]
