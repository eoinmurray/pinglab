#let meta = (
  title: "Log",
  date: "2026-07-11",
  description: "The lab notebook for the SHD program: decisions, failures, aborts, and anomalies, one section per session, newest first. Digest at the top for the morning read.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

#let body = [
  The program's notebook — the _why_ and _what-happened_ that the run-stamps do
  not capture. Append-only, timestamped, newest first. Relaxed and chronological
  by design;
  still a published page, so run numbers cite the record, never hand-typed. The
  goal and the queue live in the #link("/ar063/")[plan]; the operating policy in
  the #link("/ar065/")[night-shift].

  == Digest

  _Latest, for the 30-second morning read. Updated 2026-07-11 17:36 BST._

  Program opened. Goal set: find the minimal recipe that trains a conductance
  E/I net stably on SHD. #link("/exp060/")[exp060] smoke trains to 61% on a
  subset but *diverges intermittently* (NaN forward-passes, unbounded W_ee).
  Stability queue (dt / Dale's law / weight decay) is registered and waiting.
  *Open anomaly for a human:* NaN appears even at epoch 2 when weights are small
  — so it is not weight-growth runaway; likely integration stiffness at dt = 1.0.

  == Sessions

  === 2026-07-11 17:36 BST — program opened

  Stood the program up end to end in one sitting.

  - *Built the substrate.* SHD training now works through the CLI: an event-based
    loader (official train/test split, lazy binning to the model's dt), all four
    recurrent blocks made trainable (the W_ii CLI path was a latent dead flag,
    now fixed), and an AdamW weight-decay knob. The
    #link("/exp059/")[data-look entry] confirmed the dataset is clean and
    class-separable.

  - *Ran the first training smoke* (#link("/exp060/")[exp060], Rung A: free
    signed-recurrent ceiling, dt 1.0, 1000-sample subset). It trains — loss
    3.18 → 0.56, best test accuracy 61% against a 5% floor — so the pipeline is
    sound. But the free dynamics are unstable: scattered epochs return NaN
    train/test loss, pre-clip gradient norms spike to millions, and the E→E
    recurrent weight grows without bound.

  - *Chased the instability and was wrong once.* First guess was unbounded weight
    growth; added weight decay. It tamed the gradient explosion (7.5M → ~900) and
    lifted final accuracy (50% → 58%), but did _not_ bound W_ee at 1e-3 and made
    the NaN _more_ frequent — and a NaN showed up at epoch 2 with tiny weights.
    So the root cause is not weight size; it is an intermittent forward-pass
    divergence of the signed conductance dynamics, most likely exp-Euler
    stiffness at coarse dt. Parked for exp061.

  - *Reframed the program.* What began as debugging is the actual goal: a stable
    training recipe, with each ingredient attributed. Queued exp061 (dt sweep),
    exp062 (Dale's law as the implicit stabiliser), exp063 (weight-decay sweep),
    with a forward-state clamp held in reserve.
]
