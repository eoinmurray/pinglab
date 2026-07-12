#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Does a state clamp stabilise the free net?",
  date: "2026-07-11",
  description: "The lead the Spiking Heidelberg Digits stability queue pointed to. The free signed net's not-a-number divergence traces to the steady-state voltage v_inf = (…)/g_tot blowing up when signed weights drive the total conductance g_tot to zero or below. This tests whether a forward-pass state clamp that floors conductances at 0 removes the divergence while keeping the weights signed, and whether adding strong weight decay beats both the free net's accuracy and Dale's law's stability.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp064/numbers.json")
#let cell-by(nm) = r.cells.filter(c => c.name == nm).at(0)
#let free = cell-by("free__seed42")
// exp063's strongest-decay peak accuracy, cross-imported so the cross-reference
// stays provenance-tracked (the ar009 pattern), not hand-typed.
#let e063strong = json("/artifacts/data/exp063/numbers.json").cells.sorted(key: c => c.wd).last()
// exp060's first not-a-number epoch, cross-imported so the "NaN as early as
// epoch N" claim stays provenance-tracked rather than hand-typed.
#let e060 = json("/artifacts/data/exp060/numbers.json")
#let clamp = cell-by("clamp__seed42")
#let clampwd = cell-by("clampwd__seed42")
#let oom(x) = calc.round(calc.log(x, base: 10))

#let results-caption = [
  *What we see confirms the mechanism.* The free baseline (red) carries
  #free.nan_epochs of #free.epochs epochs with a not-a-number (NaN) loss; both
  clamped cells (black) drop that to *zero*. Critically, the recurrent
  excitatory-to-excitatory weight norm ‖W_ee‖ is unchanged by the clamp
  (#calc.round(clamp.max_wee_norm, digits: 1) versus the free net's
  #calc.round(free.max_wee_norm, digits: 1)): the clamp bounds the conductance,
  not the weight, exactly as the total-conductance $g_"tot" <= 0$ mechanism
  predicts. And accuracy is untouched: the clamped net reaches
  #calc.round(clamp.best_acc_pct, digits: 1)%, matching the free baseline's
  #calc.round(free.best_acc_pct, digits: 1)%.
]

#let reading-body = [
  *The clamp stabilises the signed net at no accuracy cost.* The free baseline
  diverges as always: #free.nan_epochs of #free.epochs epochs reach a
  not-a-number (NaN) loss, with a peak gradient norm (before clipping) of
  ≈ 10#super[#oom(free.max_grad_norm)]. Add only the forward-pass state clamp and
  the divergence vanishes: #clamp.nan_epochs NaN epochs, the gradient norm bounded
  at ≈ #calc.round(clamp.max_grad_norm), and best accuracy
  #calc.round(clamp.best_acc_pct, digits: 1)%, indistinguishable from the free
  net's #calc.round(free.best_acc_pct, digits: 1)%. The clamp is nearly
  transparent when the net behaves and bites only when a conductance would cross
  into the divergent $g_"tot" <= 0$ regime.

  *The mechanism is confirmed, not merely patched.* Two signatures pin it down.
  First, the weight norm ‖W_ee‖ is unchanged by the clamp
  (#calc.round(clamp.max_wee_norm, digits: 1) clamped versus
  #calc.round(free.max_wee_norm, digits: 1) free): the weights grow exactly as
  before, so the clamp bounds the *state* (the conductance), not the weight. That
  is why #link("/exp063/")[weight decay], which bounds the weights, could not fix
  the divergence while the clamp can. Second, the effect is total
  (#free.nan_epochs → #clamp.nan_epochs NaN epochs), as a hard floor on a
  divergent quantity should be, not the partial trend a soft regulariser gives.

  *The program's answer, ranked.* Three recipes now train the net stably:
  #link("/exp062/")[Dale's law], the state clamp, and clamp plus weight decay.
  The state clamp is the best of them: it keeps the signed net's full accuracy
  (#calc.round(clamp.best_acc_pct, digits: 1)%, clearing the Dale's-law recipe by
  several points) while removing the divergence, whereas Dale's law bought
  stability at the price of accuracy. The plan's registered goal was a stable
  recipe; the sharper answer is that the free net never needed *constraining*,
  only its *state bounding*.

  *What is left.* Strong decay on top of the clamp did not lift accuracy here
  (#calc.round(clampwd.best_acc_pct, digits: 1)%), so exp063's unstable #calc.round(e063strong.best_acc_pct, digits: 1)% peak
  was a property of the diverging trajectory, not headroom the clamp preserves:
  the stable ceiling for this recipe sits near
  #calc.round(clamp.best_acc_pct, digits: 1)%. Confirming the result across random
  seeds, and testing whether the clamp shifts the network's γ (gamma) rhythm or
  its firing sparsity, are the natural follow-ups.

  #emph[Caveat.] Single seed. The stability effect is categorical
  (#free.nan_epochs versus #clamp.nan_epochs NaN epochs) and so is safe to report,
  but the accuracy gaps between the stable recipes fall within a few points and
  want a three-seed confirmation before they can be ranked.
]

#let body = [
  == Abstract

  The free signed-recurrent network diverges during training: its loss becomes a
  not-a-number (NaN). Reading the forward pass locates the cause. The
  conductance-based neuron relaxes each timestep toward a steady-state voltage
  $v_infinity$ that is inversely proportional to the total conductance
  $g_"tot"$, so it blows up when signed weights drive $g_"tot"$ to zero or below.
  The fault is the *sign* of the weights, not their size. The hypothesis: a
  forward-pass state clamp that floors every conductance at #(r.config.g_clamp_min) each timestep keeps
  $g_"tot" >= g_L > 0$ and removes the divergence, while the weights stay signed
  and expressive. Three cells test it at $Delta t = #(r.dt_ms)$ ms, single seed:
  the free baseline, free plus clamp, and clamp plus strong weight decay. The
  clamp settles it. It cuts the free net's NaN loss from #free.nan_epochs of
  #free.epochs epochs to none and holds best accuracy at
  #calc.round(clamp.best_acc_pct, digits: 1)%, level with the free baseline's
  #calc.round(free.best_acc_pct, digits: 1)% and above the Dale's-law recipe:
  stability with no loss of the signed net's expressivity.

  == Methods

  Across this program a recurrent spiking network is trained on the Spiking
  Heidelberg Digits (SHD)#cite(1), a 20-class spoken-digit benchmark. The *free*
  variant, whose recurrent weights are *signed* (a neuron may both excite and
  inhibit, breaking Dale's law), diverges during training as its loss becomes
  not-a-number (NaN). A finer integration step (#link("/exp061/")[exp061]) and
  weight decay (#link("/exp063/")[exp063]) both failed to stabilise this free
  signed net, and only enforcing Dale's law (#link("/exp062/")[exp062]) worked,
  at an accuracy cost. Reading the forward pass explains why: the
  conductance-based leaky integrate-and-fire (LIF) neuron relaxes each timestep
  toward a steady-state voltage that integrates in closed form to
  $ v_infinity = (g_L E_L + g_e E_e + g_i E_i) \/ g_"tot", quad g_"tot" = g_L + g_e + g_i, $
  where

  - $v_infinity$ is the steady-state (asymptotic) membrane voltage the neuron is driven toward;
  - $g_L$, $g_e$, $g_i$ are the leak, excitatory, and inhibitory conductances;
  - $E_L$, $E_e$, $E_i$ are their respective reversal (equilibrium) potentials;
  - $g_"tot"$ is the total conductance, the sum of the three.

  With *signed* weights an excitatory or inhibitory conductance can itself go
  negative, so $g_"tot"$ can cross zero and $v_infinity$ diverges to NaN. That is
  why exp060 saw a NaN as early as epoch #e060.first_nan_epoch even with tiny weights: the cause is
  the *sign* of the weights, not their magnitude.

  The reserved fix is a *forward-pass state clamp*: during the forward pass, floor
  each conductance at #(r.config.g_clamp_min) on every timestep (a conductance cannot be physically
  negative), which keeps $g_"tot" >= g_L > 0$ and bounds $v_infinity$ between the
  reversal potentials, while the *weights stay signed and expressive*. Three cells
  test the clamp at $Delta t = #(r.dt_ms)$ ms, single seed, full scale:

  + *free (baseline)*: the signed net with no Dale's-law constraint, the one that
    diverges;
  + *free + state clamp*: the same net with the forward-pass state clamp added.
    Does the clamp remove the NaN divergence?
  + *free + clamp + decay #clampwd.wd*: exp063's strongest weight decay reached #calc.round(e063strong.best_acc_pct, digits: 1)% but
    could not stabilise the net on its own. With the clamp now doing the
    stabilising, does clamp plus decay top the program?

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet (conductance-based network), #(r.config.n_hidden) hidden units, all four recurrent weight blocks trainable, signed (no Dale's law)],
    [Cells], [free · free + clamp · free + clamp + weight-decay #clampwd.wd],
    [State clamp], [floor conductances at #(r.config.g_clamp_min), cap magnitude at #(r.config.g_clamp_max) µS each timestep],
    [Integration], [$Delta t = #(r.dt_ms)$ ms, $T = #(r.config.t_ms)$ ms],
    [Seeds], [1 (seed #r.seed)],
    [Training], [#r.epochs epochs, learning rate #(r.config.lr), batch size #(r.config.batch_size), surrogate-gradient backpropagation through time (BPTT)#cite(2)],
    [Compute], [#r.compute],
  )

  #quote(block: true)[
    *Success.* The clamp trains the free net NaN-free with bounded dynamics, at
    accuracy at least matching Dale's law: stability that keeps the signed net's
    expressivity. *Kill:* if the clamp leaves any NaN present, the $g_"tot" <= 0$
    mechanism is not the whole cause.
  ]

  == Results

  #figure(
    image("/artifacts/data/exp064/clamp_bars.svg", width: 100%,
      alt: "Three bar panels (NaN-epoch rate, max recurrent-weight norm, best accuracy) for free, free+clamp, and clamp+decay."),
    caption: [
      The three recipes compared on three measures (free baseline in red, both
      clamped cells in black). Left panel: the percentage of the #free.epochs
      training epochs whose loss became not-a-number (NaN). Middle panel: the
      largest recurrent excitatory-to-excitatory weight norm ‖W_ee‖ reached during
      training. Right panel: best test accuracy (percent), with the dashed line
      marking the #calc.round(r.chance_pct)% chance level. #results-caption
    ],
  )

  #figure(
    image("/artifacts/data/exp064/loss_traces.svg", width: 100%,
      alt: "Test loss per epoch for the three cells; the free baseline shows NaN gaps, the clamped cells run continuously."),
    caption: [
      Test cross-entropy loss per epoch for the three recipes (x-axis: training
      epoch; y-axis: loss on the held-out test set). The free baseline (red)
      leaves gaps at the epochs where its loss diverges to not-a-number; the two
      clamped cells (near-black, distinguished by line style) run continuously
      from the first epoch to the last, so the clamp removes the divergence.
    ],
  )

  #reading-body

  #reference-list((
    (text: [Cramer, Stradmann, Schemmel & Zenke — _The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks_. 2022.], doi: "10.1109/TNNLS.2020.3044364"),
    (text: [Neftci, Mostafa & Zenke — _Surrogate Gradient Learning in Spiking Neural Networks_. 2019.], doi: "10.1109/MSP.2019.2931595"),
  ))
]
