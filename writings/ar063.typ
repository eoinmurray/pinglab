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

  + *exp061 — does finer dt stabilise the integration?* _(3 seeds · status: queued)_
    - _Hypothesis:_ the NaN divergence is exp-Euler stiffness at coarse dt;
      halving and quartering dt (1.0 → 0.5 → 0.25) drops the NaN-epoch rate
      toward zero.
    - _Kill:_ if NaN persists at dt 0.25, coarse integration is not the cause —
      drop dt from the recipe and look upstream (state clamping).
    - _Measures:_ NaN-epoch rate and max gradient norm vs dt.

  + *exp062 — is Dale's law the implicit stabiliser?* _(3 seeds · status: queued)_
    - _Hypothesis:_ the Dale's-law constraint (non-negativity clamp) keeps the
      dynamics in a stable regime; the same recipe *with* Dale's law trains
      NaN-free where the free version diverges.
    - _Kill:_ if the Dale's-law net also NaNs at matched settings, the constraint
      is not what stabilises — the free-vs-constrained difference lies elsewhere.
    - _Measures:_ NaN-epoch rate, max weight norm, and best accuracy, free vs
      constrained.

  + *exp063 — does weight decay bound the free recurrence?* _(status: queued)_
    - _Hypothesis:_ decay strong enough (sweep 0 → 1e-3 → 1e-2) bounds W_ee and
      removes the divergence in the free net. A first pass at 1e-3 tamed the
      gradient explosion (7.5M → ~900) but did _not_ bound W_ee or remove the
      NaN, so the sweep must reach higher.
    - _Kill:_ if even 1e-2 leaves W_ee growing and NaN present, decay is not part
      of the recipe (it regularises but does not stabilise).
    - _Measures:_ max weight norm and NaN-epoch rate vs decay strength.

  The firing-rate regulariser is already in the recipe (without it the earlier
  run blew up to a 526 Hz inhibitory rate); the queue attributes the _remaining_
  ingredients. If none of dt, Dale's law, or decay yields a NaN-free recipe, a
  forward-pass state clamp (bounding voltage and conductance) enters the queue as
  the fallback — a code change to the shared model, so a last resort.

  == Amendments

  None yet. Any change to a hypothesis or kill criterion after an experiment has
  run against it is added here as a dated entry, never edited in place.
]
