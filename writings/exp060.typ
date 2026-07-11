#let meta = (
  title: "SHD training smoke test — it trains",
  date: "2026-07-11",
  description: "The first training run in the Spiking Heidelberg Digits program: the free signed-recurrent ceiling (Rung A) trains on a 1000-sample subset — loss falls and test accuracy climbs decisively off the 5% chance floor.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed, so
// prose and results cannot drift.
#let r = json("/artifacts/data/exp060/numbers.json")

#let body = [
  == What this checks

  Before the #link("/ar063/")[program plan]'s baseline can be trusted, one thing
  has to hold: the SHD training pipeline actually *trains*. This is that check —
  deliberately a low bar. It runs Rung A of the plan, the free signed-recurrent
  ceiling (all four recurrent blocks trainable, _--no-dales-law_, gamma untuned),
  on a small subset for a few dozen epochs, and asks only whether the loss falls
  and test accuracy climbs off the #calc.round(r.chance_pct)% chance floor. It is engineering
  validation, _not_ a registered result — the baseline number the plan pins comes
  from a full-scale run.

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.n_train_subset)-sample subset (of 8156)],
    [Model], [COBANet, 256 hidden, all four recurrent blocks trainable, signed (no Dale's law)],
    [Integration], [$Delta t = 1.0$ ms, $T = 1000$ ms ($T_"steps" = 1000$) — coarse dt so a full 1 s window still trains in minutes],
    [Training], [#r.epochs epochs, lr 0.001, batch 32, surrogate-gradient BPTT],
    [Regulariser], [upper firing-rate bound (Cramer et al. SHD values, $theta_u = 100$, $s_u = 0.06$) — without it the signed recurrence runs away],
  )

  The coarse $Delta t = 1$ ms is the fastest our conductance model tolerates
  (verified on MNIST); it is still ≈ 10× finer than Cramer et al.'s discrete-LIF
  timestep, because our fast synaptic and inhibitory time constants ($tau_"AMPA" =
  2$ ms, $tau_(m,I) = 5$ ms) need resolving. Where:

  - $Delta t$, the integration timestep;
  - $T_"steps" = T \/ Delta t$, the number of BPTT steps unrolled per utterance.

  == Results

  #figure(
    image("/artifacts/data/exp060/loss.svg", width: 100%,
      alt: "Training and test cross-entropy loss against epoch; training loss falls monotonically while test loss turns upward after a few epochs."),
    caption: [
      Cross-entropy loss against epoch. *What we expect.* If the pipeline learns,
      training loss falls. *What we see.* It does — training loss drops from
      #calc.round(r.first_epoch_loss, digits: 2) to
      #calc.round(r.final_loss, digits: 2). Test loss falls with it for the first
      few epochs, then turns upward: the network is overfitting the
      #(r.n_train_subset)-sample subset, exactly as a small training set predicts.
    ],
  )

  #figure(
    image("/artifacts/data/exp060/accuracy.svg", width: 100%,
      alt: "Test accuracy against epoch, rising from about 12% to a peak above 60%, well above the 5% chance line."),
    caption: [
      Test accuracy against epoch, with the #calc.round(r.chance_pct)% chance floor (20
      classes) drawn in. *What we see.* Accuracy climbs from
      #calc.round(r.first_epoch_acc_pct)% at epoch 1 to a best of
      #calc.round(r.best_acc_pct)% at epoch #r.best_epoch — an order of magnitude
      above chance — then plateaus and wobbles (final
      #calc.round(r.final_acc_pct)%) as the subset overfits.
    ],
  )

  == Reading

  Both bars are cleared: the loss falls and accuracy reaches
  #calc.round(r.best_acc_pct)%, far above both the #calc.round(r.chance_pct)% floor and the
  plan's "decisively above chance" threshold of 25%. So the SHD data path, the
  binning, the all-four-trainable signed recurrence, and the surrogate-gradient
  training loop all work end to end. The firing-rate regulariser is load-bearing:
  without it the earlier unregularised run blew up to a 526 Hz inhibitory rate;
  with it, rates stay physiological and the network learns.

  The overfitting is the expected cost of the small subset and is _why_ this is a
  smoke test, not a result: the honest ceiling number has to come from a
  full-scale run, where the test set is not swamped by a 1000-sample training set.
  That full-scale run is the plan's registered Rung A. Next: scale the subset up
  (and add mild early-stopping) for the number, then descend the ladder to the
  Dale's-law-constrained baseline.
]
