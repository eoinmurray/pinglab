#let meta = (
  title: "Does weight decay bound the free recurrence?",
  date: "2026-07-11",
  description: "The SHD program's third stability probe: with Dale's law shown (exp062) to stabilise the net at an accuracy cost, sweep AdamW weight decay on the free signed net to ask whether decay alone can bound the runaway recurrent weight and remove the NaN divergence, stabilising the net without the constraint's accuracy tax.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp063/numbers.json")
#let cells = r.cells.sorted(key: c => c.wd)   // weak → strong decay
#let weakest = cells.first()
#let strongest = cells.last()
// exp062's free-vs-Dale's-law accuracies, cross-imported so the comparison stays
// provenance-tracked (the ar009 pattern), not hand-typed.
#let e062free = json("/artifacts/data/exp062/numbers.json").cells.filter(c => c.dales_law == false).at(0)
#let e062dales = json("/artifacts/data/exp062/numbers.json").cells.filter(c => c.dales_law == true).at(0)
// The swept λ values, formatted from the per-cell wd (never hand-typed): 0 stays
// 0, every other decay renders as a power of ten (its base-10 exponent).
#let fmt-wd(w) = if w == 0 { $0$ } else { $10^(#int(calc.round(calc.log(w, base: 10))))$ }
#let wd-set = $lambda in {#(cells.map(c => fmt-wd(c.wd)).join(", "))}$

#let results-caption = [
  *What we see fails the test.* The NaN-epoch rate falls only a little as the
  decay strength λ grows, from #weakest.nan_epochs of #weakest.epochs epochs at
  λ = #weakest.wd to #strongest.nan_epochs at λ = #strongest.wd, and never reaches zero; the
  recurrent weight norm $W_"ee"$ barely moves
  (≈ #calc.round(weakest.max_wee_norm, digits: 1) throughout), so decay does not
  bound the runaway weight. One quantity does improve monotonically: best test
  accuracy climbs from #calc.round(weakest.best_acc_pct, digits: 1)% to
  #calc.round(strongest.best_acc_pct, digits: 1)%. Decay is a good regulariser (it
  improves generalisation), just not a stabiliser (it does not prevent the
  numerical divergence).
]

#let reading-body = [
  *The kill criterion fires.* Even the strongest decay (λ = #strongest.wd) leaves
  the free net diverging (#strongest.nan_epochs of #strongest.epochs epochs go
  NaN) and the recurrent weight norm $W_"ee"$ is essentially flat across the whole
  sweep (#calc.round(strongest.max_wee_norm, digits: 1) at the strongest λ against
  #calc.round(weakest.max_wee_norm, digits: 1) at zero). Weight decay neither
  bounds the recurrent weight nor removes the divergence. It _regularises_:
  accuracy rises steadily with λ, from #calc.round(weakest.best_acc_pct, digits: 1)%
  to #calc.round(strongest.best_acc_pct, digits: 1)%, the free net's best number
  yet, but that is a generalisation effect, not a stability one. Decay is not part
  of the stability recipe.

  *The picture is now complete for the three soft knobs.* A soft knob here is a
  training setting the free net can vary without changing the model equations. Of
  the three tested (the integration timestep Δt in #link("/exp061/")[exp061],
  Dale's law in #link("/exp062/")[exp062], and weight decay here), only Dale's law
  stabilises. Δt makes the gradient explosion _worse_; weight decay regularises
  without bounding $W_"ee"$; only the hard non-negativity constraint of Dale's law
  holds the excitatory-inhibitory-excitatory (E→I→E) feedback loop gain below
  runaway. So the free signed net's divergence is not reachable by any soft knob
  tested. Stabilising it _while keeping its accuracy_ needs the reserved model
  change: a forward-pass state clamp that bounds voltage and conductance so a
  diverging trajectory cannot reach NaN.

  *A hint for that next step.* Decay's accuracy gain (up to
  #calc.round(strongest.best_acc_pct, digits: 1)%, the free net's best here and
  well above the Dale's-law recipe) says the signed net has headroom that the
  constraint gives up. A state clamp that stabilises the free net _at strong
  decay_ could plausibly beat every recipe in the program so far. That is the
  experiment the plan now points to.

  #emph[Caveat.] Single seed; the effect is categorical (NaN present at every λ),
  so it is unlikely to reverse, but a 3-seed confirmation is cheap to run.
]

#let body = [
  == Abstract

  *Weight decay regularises the free net but does not stabilise it.* The Spiking
  Heidelberg Digits (SHD) program already has a stable recipe from
  #link("/exp062/")[exp062]: imposing Dale's law (each neuron's outgoing synapses
  share a fixed sign, all excitatory or all inhibitory) trains the network NaN-free
  where the free signed net, whose weights may take either sign, diverges. But that
  constraint costs a few points of test accuracy
  (#calc.round(e062dales.best_acc_pct, digits: 1)% against the free net's
  #calc.round(e062free.best_acc_pct, digits: 1)%), the price of denying the
  signed recurrence its expressivity. So the live
  question is whether stability can be had *without* that tax: can a plain
  regulariser bound the free net's runaway dynamics instead of a hard constraint?
  To answer it, this sweeps the AdamW weight decay
  #wd-set on the free net and watches the
  recurrent weight norm $W_"ee"$ and the NaN-epoch rate. The verdict is that decay
  fails the test: it improves generalisation but never bounds $W_"ee"$ or removes
  the numerical divergence.

  == Methods

  + Fix the free net (signed recurrence, no Dale's law) to the
    #link("/exp060/")[exp060] recipe at integration timestep Δt = #r.config.dt_ms ms, the
    setting where it diverges, single seed, at full run scale.
  + Sweep the AdamW weight decay $lambda$ (a penalty that shrinks every weight
    toward zero at each step; λ sets its strength) over
    #wd-set. Weight decay is the *only* thing
    that varies, so any change in the stability metrics is attributable to λ alone.
  + For each λ, record the plan's pre-registered contrast: the maximum norm of the
    recurrent excitatory-to-excitatory weight matrix $W_"ee"$ and the NaN-epoch
    rate (the fraction of training epochs whose loss or activity went non-finite),
    together with the best test accuracy.

  A first pass at $lambda = #fmt-wd(cells.at(1).wd)$ tamed the gradient explosion but did not bound
  $W_"ee"$ or remove the NaN, so the sweep reaches up to #fmt-wd(strongest.wd).

  #quote(block: true)[
    *Kill criterion* (the pre-registered condition that would rule decay out): if
    even the strongest decay leaves $W_"ee"$ growing and NaN still present, decay
    is not part of the stability recipe. It regularises but does not stabilise.
  ]

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet (conductance-based spiking network), #(r.config.n_hidden) hidden units, all four recurrent weight blocks ($W_"ee"$, $W_"ei"$, $W_"ie"$, $W_"ii"$) trainable, signed (no Dale's law)],
    [Swept variable], [AdamW weight decay #wd-set],
    [Integration], [$Delta t = #(r.dt_ms)$ ms, $T = #(r.config.t_ms)$ ms (the setting where the free net diverges)],
    [Seeds], [#(r.config.seeds) (seed #r.seed), a quick sweep],
    [Training], [#r.epochs epochs, learning rate #r.config.lr, batch size #(r.config.batch_size), surrogate-gradient backpropagation through time (BPTT)],
    [Regulariser], [upper firing-rate bound (threshold $theta_u = #(r.config.theta_u)$, strength $s_u = #(r.config.s_u)$)],
    [Compute], [#r.compute],
  )

  == Results

  #figure(
    image("/artifacts/data/exp063/decay_sweep.svg", width: 100%,
      alt: "Three panels against weight decay λ: NaN-epoch rate, max recurrent-weight norm, and best test accuracy."),
    caption: [
      Three stability metrics against the AdamW weight decay λ (weak to strong),
      one per panel: (left) the NaN-epoch rate, the percentage of training epochs
      whose loss or activity went non-finite; (middle) the maximum norm of the
      recurrent excitatory-to-excitatory weight matrix $W_"ee"$; (right) the best
      test accuracy in percent, the dashed line marking chance
      (#calc.round(r.chance_pct, digits: 0)%). *What we expect if decay stabilises:*
      the NaN-epoch rate and the runaway weight norm fall as λ grows, ideally
      reaching zero NaN while accuracy holds above the Dale's-law recipe.
      #results-caption
    ],
  )

  #reading-body
]
