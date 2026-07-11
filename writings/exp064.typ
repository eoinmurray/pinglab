#let meta = (
  title: "Does a state clamp stabilise the free net?",
  date: "2026-07-11",
  description: "The lead the SHD stability queue pointed to: with the free signed net's NaN traced to the exp-Euler v_inf = (…)/g_tot blowing up when signed weights drive g_tot ≤ 0, test whether a forward-pass state clamp that floors conductances at 0 removes the divergence while keeping the weights signed — and whether clamp + strong decay beats both the free net's accuracy and Dale's law's stability.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp064/numbers.json")
#let cell-by(nm) = r.cells.filter(c => c.name == nm).at(0)
#let free = cell-by("free__seed42")
#let clamp = cell-by("clamp__seed42")
#let clampwd = cell-by("clampwd__seed42")

#let results-caption = [_What we see_ is written in the Reading section below.]
#let reading-body = [_(finalised once the run's numbers are in.)_]

#let body = [
  == What this checks

  The stability queue converged on a mechanism. #link("/exp061/")[exp061] (Δt),
  #link("/exp063/")[exp063] (weight decay) both failed to stabilise the free
  signed net, and #link("/exp062/")[exp062] showed only Dale's law works — at an
  accuracy cost. Reading the forward pass explains why: the conductance-based LIF
  integrates in closed form to
  $ v_infinity = (g_L E_L + g_e E_e + g_i E_i) \/ g_"tot", quad g_"tot" = g_L + g_e + g_i, $
  and with *signed* weights a conductance can go negative, so $g_"tot"$ can cross
  zero and $v_infinity$ diverges to NaN. That is why exp060 saw a NaN at epoch 2
  with tiny weights: it is the *sign*, not the magnitude.

  So the plan's reserved fix is a *forward-pass state clamp*: floor the
  conductances at 0 each timestep (physical — a conductance cannot be negative),
  which keeps $g_"tot" >= g_L > 0$ and bounds $v_infinity$ between the reversal
  potentials, while the *weights stay signed and expressive*. This tests it in
  three cells at Δt = 1.0 ms, single seed, full scale:

  + *free (baseline)* — _--no-dales-law_, the net that diverges;
  + *free + state clamp* — _--state-clamp_ — does the clamp remove the NaN?
  + *free + clamp + decay 0.1* — exp063's strongest decay reached 63.4% but could
    not stabilise; with the clamp doing the stabilising, does clamp + decay top
    the program?

  #quote(block: true)[
    *Success.* The clamp trains the free net NaN-free with bounded dynamics, at
    accuracy at least matching Dale's law — stability that keeps the signed net's
    expressivity. *Kill:* if the clamp leaves NaN present, the $g_"tot" <= 0$
    mechanism is not the whole cause.
  ]

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet, 256 hidden, all four recurrent blocks trainable, signed (no Dale's law)],
    [Cells], [free · free + clamp · free + clamp + weight-decay 0.1],
    [State clamp], [floor conductances at 0, cap magnitude at 100 µS each timestep],
    [Integration], [$Delta t = #(r.dt_ms)$ ms, $T = 1000$ ms],
    [Seeds], [1 (seed #r.seed)],
    [Training], [#r.epochs epochs, lr 0.001, batch 32, surrogate-gradient BPTT],
    [Compute], [#r.compute],
  )

  == Compute

  RunPod fan-out, one pod per cell, collected off the shared volume over its S3
  HTTPS API (the collector from #link("/exp061/")[exp061]).

  == Results

  #figure(
    image("/artifacts/data/exp064/clamp_bars.svg", width: 100%,
      alt: "Three bar panels — NaN-epoch rate, max recurrent-weight norm, best accuracy — for free, free+clamp, and clamp+decay."),
    caption: [
      The three cells across NaN-epoch rate, max ‖W_ee‖, and best accuracy (free
      in red, clamped in black). *What we expect if the clamp is the fix.* The
      free bar carries the NaN rate; the clamped bars drop it to zero with bounded
      W_ee, and clamp + decay lifts accuracy. #results-caption
    ],
  )

  #figure(
    image("/artifacts/data/exp064/loss_traces.svg", width: 100%,
      alt: "Test loss per epoch for the three cells; the free baseline shows NaN gaps, the clamped cells run continuously."),
    caption: [
      Test loss per epoch. The free baseline leaves gaps where it diverges; the
      clamped cells should run continuously start to finish.
    ],
  )

  == Reading

  #reading-body
]
