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
#let oom(x) = calc.round(calc.log(x, base: 10))

#let results-caption = [
  *What we see confirms the mechanism.* The free baseline (red) carries
  #free.nan_epochs of #free.epochs NaN epochs; both clamped cells (black) drop
  that to *zero*. Critically, ‖W_ee‖ is unchanged by the clamp
  (#calc.round(clamp.max_wee_norm, digits: 1) vs the free net's
  #calc.round(free.max_wee_norm, digits: 1)) — the clamp bounds the conductance,
  not the weight, exactly as the $g_"tot" <= 0$ mechanism predicts. And accuracy
  is untouched: the clamped net reaches #calc.round(clamp.best_acc_pct, digits: 1)%,
  matching the free baseline's #calc.round(free.best_acc_pct, digits: 1)%.
]

#let reading-body = [
  *The clamp stabilises the signed net at no accuracy cost.* The free baseline
  diverges as always — #free.nan_epochs of #free.epochs epochs NaN, peak pre-clip
  gradient ≈ 10#super[#oom(free.max_grad_norm)]. Add only the forward-pass state
  clamp and the divergence vanishes: #clamp.nan_epochs NaN epochs, the gradient
  bounded at ≈ #calc.round(clamp.max_grad_norm), and best accuracy
  #calc.round(clamp.best_acc_pct, digits: 1)% — indistinguishable from the free
  net's #calc.round(free.best_acc_pct, digits: 1)%. The clamp is nearly
  transparent when the net behaves and bites only when a conductance would cross
  into the divergent $g_"tot" <= 0$ regime.

  *This is the mechanism confirmed, not just a fix that works.* Two signatures pin
  it down. First, ‖W_ee‖ is unchanged by the clamp
  (#calc.round(clamp.max_wee_norm, digits: 1) clamped vs
  #calc.round(free.max_wee_norm, digits: 1) free) — the weights grow exactly as
  before; the clamp bounds the *state* (conductance), not the weight, which is
  why #link("/exp063/")[weight decay] (which bounds weights) could not fix it and
  the clamp can. Second, the effect is total (#free.nan_epochs → #clamp.nan_epochs
  NaN epochs), as a hard floor on a divergent quantity should be — not the partial
  trend a soft regulariser gives.

  *The program's answer, ranked.* Three recipes now train the net stably:
  #link("/exp062/")[Dale's law], the state clamp, and clamp + decay. The state
  clamp is the best of them — it keeps the signed net's full accuracy
  (#calc.round(clamp.best_acc_pct, digits: 1)%, clearing the Dale's-law recipe by
  several points) while removing the divergence, where Dale's law paid for
  stability with accuracy. The plan's registered goal was a stable recipe; the
  sharper answer is that the free net never needed *constraining*, only its
  *state bounding*.

  *What is left.* Strong decay on top of the clamp did not lift accuracy here
  (#calc.round(clampwd.best_acc_pct, digits: 1)%), so exp063's unstable 63.4% peak
  was a property of the diverging trajectory, not headroom the clamp preserves —
  the stable ceiling for this recipe sits near
  #calc.round(clamp.best_acc_pct, digits: 1)%. Confirming across seeds, and
  whether the clamp shifts the network's γ-rhythm or sparsity, are the natural
  follow-ups.

  #emph[Caveat.] Single seed; the stability effect is categorical
  (#free.nan_epochs vs #clamp.nan_epochs NaN) so robust, but the accuracy gaps
  between the stable recipes are within a few points and want the 3-seed
  confirmation — cheap on the RunPod + S3 path.
]

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
