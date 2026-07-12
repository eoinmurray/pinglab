#let meta = (
  title: "Plan",
  date: "2026-07-11",
  description: "Pre-registration for the Spiking Heidelberg Digits program. One goal: find the minimal recipe that trains a conductance-based E/I spiking network stably on SHD, and attribute each ingredient — dt, weight decay, Dale's law, firing-rate regulariser — to what it buys.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

#let body = [
  This is the *plan* article — the pre-registration and the live experiment
  queue. Because every demolab page carries the git commit that built it, a plan
  committed before any experiment has run is provably prior to the results. The
  policy the overnight shift obeys lives in the #link("/ar065/")[night-shift]
  article; the run-by-run record lives in the #link("/ar064/")[log].

  == The goal

  We can train the conductance-based E/I network (#link("/ar003/")[COBANet]) on
  #link("/exp059/")[Spiking Heidelberg Digits], and the first smoke run
  (#link("/exp060/")[exp060]) reached 61% on a subset — but it did so *unstably*:
  the free signed-recurrent dynamics tip the forward pass into NaN divergence on
  scattered epochs, gradient norms spike into the millions, and the recurrent
  weights grow without bound. Before any larger claim about what biological
  structure costs or the gamma rhythm does, the network has to *train reliably*.
  So the program has exactly one goal, deliberately modest:

  #quote(block: true)[
    Find the minimal recipe that trains a conductance-based E/I spiking network
    stably on SHD, and attribute each ingredient — integration timestep,
    weight decay, Dale's law, the firing-rate regulariser — to the stability it
    buys.
  ]

  Bigger results (the cost of biological structure, whether gamma helps
  classification) wait until this holds. This is the foundation they stand on.

  *What counts as success.* A recipe that trains to completion with:

  - _no NaN divergence_ — every epoch produces a finite train and test loss;
  - _bounded dynamics_ — pre-clip gradient norm and recurrent weight norm stay
    bounded across training, not runaway;
  - _decisively above chance_ — best test accuracy at or above 25% (chance is
    5% on 20 classes), so "stable" does not mean "stably not learning";

  and, alongside the recipe, a table attributing each ingredient to its effect on
  those stability metrics — what each one is actually for.

  == The queue

  Each entry isolates one candidate ingredient and measures its effect on the
  stability metrics above (NaN-epoch rate, max pre-clip gradient norm, max
  recurrent-weight norm), holding the rest of the recipe fixed. Each has a *kill*
  criterion. Status is read from each run's outputs, never hand-typed here.

  + *exp060 — the failure mode (done).* _(status: done)_
    - _Result:_ the free signed-recurrent ceiling (all four blocks trainable,
      _--no-dales-law_, dt 1.0, firing-rate regulariser on) trains to 61% on a
      1000-sample subset but with intermittent NaN forward-divergence and
      unbounded W_ee growth. This entry defines the problem the rest of the queue
      solves.

  + *exp061 — does finer dt stabilise the integration?* _(1 seed · status: done — killed)_
    - _Hypothesis:_ the NaN divergence is exp-Euler stiffness at coarse dt;
      halving and quartering dt (1.0 → 0.5 → 0.25) drops the NaN-epoch rate
      toward zero.
    - _Kill:_ if NaN persists at dt 0.25, coarse integration is not the cause —
      drop dt from the recipe and look upstream (state clamping).
    - _Measures:_ NaN-epoch rate and max gradient norm vs dt.
    - _Result (#link("/exp061/")[exp061]):_ *killed.* NaN reproduced at dt 1.0
      (13/30 epochs) but _persists_ at dt 0.25 (16/30), and the max pre-clip
      gradient norm gets ≈ 13 orders of magnitude _worse_ as dt shrinks
      (3·10⁴ → 4·10¹⁷) — finer dt quadruples the BPTT unroll and the recurrent
      gradient chain explodes. Not coarse-dt stiffness. dt drops from the recipe;
      the forward-pass state clamp becomes the lead (see Amendments).

  + *exp062 — is Dale's law the implicit stabiliser?* _(1 seed · status: done — supported)_
    - _Hypothesis:_ the Dale's-law constraint (non-negativity clamp) keeps the
      dynamics in a stable regime; the same recipe *with* Dale's law trains
      NaN-free where the free version diverges.
    - _Kill:_ if the Dale's-law net also NaNs at matched settings, the constraint
      is not what stabilises — the free-vs-constrained difference lies elsewhere.
    - _Measures:_ NaN-epoch rate, max weight norm, and best accuracy, free vs
      constrained.
    - _Result (#link("/exp062/")[exp062]):_ *supported.* At matched settings the
      free net NaNs 13/30 epochs (gradient ≈ 3·10⁴, W_ee 9.2); the Dale's-law net
      trains _NaN-free_ (0/30, gradient ≈ 190, W_ee 5.4) at 53.2% vs the free
      net's 56.8%. Dale's law is the stabiliser — the first stable recipe. The
      cost is a few points of accuracy, which reframes exp063 and the state clamp
      (below) as ways to keep the free net's accuracy _without_ its divergence.

  + *exp063 — does weight decay bound the free recurrence?* _(1 seed · status: done — killed)_
    - _Hypothesis:_ decay strong enough (sweep 0 → 1e-3 → 1e-2) bounds W_ee and
      removes the divergence in the free net. A first pass at 1e-3 tamed the
      gradient explosion (7.5M → ~900) but did _not_ bound W_ee or remove the
      NaN, so the sweep must reach higher.
    - _Kill:_ if even 1e-2 leaves W_ee growing and NaN present, decay is not part
      of the recipe (it regularises but does not stabilise).
    - _Measures:_ max weight norm and NaN-epoch rate vs decay strength.
    - _Result (#link("/exp063/")[exp063]):_ *killed* (swept to 1e-1). NaN persists
      at every λ (15/13/12/11 of 30 epochs) and W_ee stays ≈ 9 throughout — decay
      does not bound the recurrence. It _regularises_ (accuracy 56.3% → 63.4% with
      λ) but does not stabilise. Confirms the kill: decay is not the stabiliser.

  + *exp064 — does the forward-pass state clamp stabilise the free net?*
    _(1 seed · status: done — the answer)_ _(origin: promoted from the amendments
    below; a shared-model change, authorised by the scientist)_
    - _Hypothesis:_ the divergence is the exp-Euler $v_infinity = (…)\/g_"tot"$
      blowing up when signed weights drive $g_"tot" = g_L + g_e + g_i <= 0$;
      flooring conductances at 0 each timestep keeps $g_"tot" >= g_L > 0$ and
      removes the NaN while the weights stay signed.
    - _Kill:_ if the clamp leaves NaN present, the $g_"tot" <= 0$ mechanism is not
      the whole cause.
    - _Measures:_ NaN-epoch rate, max weight norm, best accuracy — free vs
      free+clamp vs clamp+decay.
    - _Result (#link("/exp064/")[exp064]):_ *the answer.* Free 13/30 NaN →
      free+clamp *0/30*, gradient 3·10⁴ → ≈ 190, at *56.9%* (vs free 56.8%,
      Dale's law 53.2%). W_ee unchanged (~9) — the clamp bounds the _state_, not
      the weight, which is why decay could not fix it. The best stable recipe:
      the signed net kept its full accuracy, only its state bounded.

  The firing-rate regulariser is already in the recipe (without it the earlier
  run blew up to a 526 Hz inhibitory rate); the queue attributes the _remaining_
  ingredients. If none of dt, Dale's law, or decay yields a NaN-free recipe, a
  forward-pass state clamp (bounding voltage and conductance) enters the queue as
  the fallback — a code change to the shared model, so a last resort.

  == Amendments

  Any change to a hypothesis or kill criterion after an experiment has run
  against it is added here as a dated entry, never edited in place.

  - *2026-07-11 — dt dropped; forward-pass state clamp promoted from fallback to
    lead.* #link("/exp061/")[exp061] killed the dt-stiffness hypothesis: finer dt
    did not remove the NaN and made the pre-clip gradient explosion orders of
    magnitude worse (the longer BPTT unroll dominates). With neither dt (exp061)
    nor weight size (exp060: NaN at epoch 2 with tiny weights) explaining the
    divergence, the reserved forward-pass state clamp — bounding voltage and
    conductance each timestep so a diverging trajectory cannot reach NaN — is now
    the primary lead rather than a last resort. It is a change to the shared
    model, so it enters the queue after the two remaining no-code-change probes
    (exp062 Dale's law, exp063 weight decay) unless those close the case first.

  - *2026-07-11 — the stability goal is met; exp063 + the state clamp are
    reframed.* #link("/exp062/")[exp062] confirmed Dale's law trains the net
    NaN-free at settings where the free net diverges — a working stable recipe,
    which is the program's registered goal. Dale's law costs a few points of
    accuracy against the free net, so the remaining probes are no longer about
    _finding_ stability but about _keeping the free net's accuracy without its
    divergence_: exp063 (weight decay) and the forward-pass state clamp are now
    tested as stabilisers of the _signed_ net, worth running only if that
    accuracy gap is worth chasing.

  - *2026-07-11 — soft-knob queue exhausted; the state clamp is the next
    experiment.* With #link("/exp063/")[exp063] killed, all three no-code-change
    knobs are tested and only Dale's law stabilises. Stabilising the _free_ net —
    the only way to keep its accuracy (exp063 reached 63.4%, above every other
    recipe) — is not reachable by Δt, Dale's law, or weight decay, so it requires
    the reserved forward-pass state clamp: bound voltage and conductance each
    timestep in `tools/snn/models.py` so a diverging trajectory cannot reach NaN.
    That is a shared-model change rather than a knob, so it is left for the human
    gate to authorise as the program's next entry (candidate: a clamped free net
    at strong weight decay, aiming to beat both the free net's accuracy and Dale's
    law's stability).

  - *2026-07-11 — goal exceeded; the state clamp is the answer, arc closed.*
    #link("/exp064/")[exp064] ran the reserved clamp (scientist-authorised) and it
    wins: the free signed net trains NaN-free (13/30 → 0/30) at 56.9%, matching the
    free baseline and beating Dale's law, by bounding the _state_ (conductance ≥ 0)
    not the weights. The registered goal — a minimal stable recipe — is met with
    _no_ accuracy cost, and the mechanism is pinned (g_tot ≤ 0 under signed
    weights), closing the arc Δt → Dale's law → weight decay → state clamp. Natural
    follow-ups, as new entries rather than amendments: the 3-seed confirmation, and
    whether the clamp shifts the network's γ-rhythm or sparsity — the questions the
    program deferred until training was reliable, which it now is.
]
