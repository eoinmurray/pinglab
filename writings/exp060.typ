#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "SHD training smoke test — it trains",
  date: "2026-07-11",
  description: "The first training run in the Spiking Heidelberg Digits program: the free signed-recurrent ceiling (Rung A) trains on a 1000-sample subset, with loss falling and test accuracy climbing decisively off the 5% chance floor.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed, so
// prose and results cannot drift.
#let r = json("/artifacts/data/exp060/numbers.json")

#let body = [
  == Abstract

  Before the #link("/ar063/")[program plan]'s baseline can be trusted, the
  Spiking Heidelberg Digits (SHD) training pipeline has to clear one low bar:
  does it actually *train*? SHD is a benchmark of spoken digits recorded as
  cochlea-model spike trains. This run exercises Rung A of the plan, the free
  signed-recurrent ceiling: the least-constrained model in the ladder, expected
  to set the performance upper bound, with all four recurrent weight blocks
  trainable, no Dale's-law sign constraint (weights may take either sign), and
  the gamma-band rhythm left untuned. It trains that model on a
  #(r.n_train_subset)-sample subset for #r.epochs epochs and asks only whether
  the loss falls and the test accuracy climbs off the #calc.round(r.chance_pct)%
  chance floor (one correct label in #(r.n_classes) classes, guessed at random). Both bars
  clear: training loss falls from #calc.round(r.first_epoch_loss, digits: 2) to
  #calc.round(r.final_loss, digits: 2), and test accuracy reaches
  #calc.round(r.best_acc_pct)%, an order of magnitude above chance. This is
  engineering validation, _not_ a registered result: the honest ceiling number
  has to come from a full-scale run, where the test set is not swamped by so
  small a training subset.

  == Methods

  The run, step by step:

  + Load the SHD training set and take a #(r.n_train_subset)-sample subset of the
    #(r.config.n_train_pool)-utterance training pool.
  + Bin each utterance's input spikes onto the $Delta t = #(r.config.dt_ms)$ ms grid over the
    $T = #(r.config.t_ms)$ ms window ($T_"steps" = #(r.config.t_steps)$).
  + Build the COBANet model with #(r.config.n_hidden) hidden units, all four recurrent weight
    blocks trainable and signed (no Dale's law).
  + Train for #r.epochs epochs by backpropagation through time with a surrogate
    gradient for the non-differentiable spike, under an upper firing-rate
    regulariser that keeps the signed recurrence from running away.
  + After each epoch, evaluate cross-entropy loss and classification accuracy on
    the held-out test set, and record the best accuracy.

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.n_train_subset)-sample subset (of the #(r.config.n_train_pool)-utterance training set)],
    [Model], [COBANet (a conductance-based spiking network), #(r.config.n_hidden) hidden units, all four recurrent weight blocks trainable, signed (no Dale's law)],
    [Integration], [$Delta t = #(r.config.dt_ms)$ ms, $T = #(r.config.t_ms)$ ms ($T_"steps" = #(r.config.t_steps)$): a coarse timestep, chosen so a full #(r.config.t_ms / 1000) s window still trains in minutes],
    [Training], [#r.epochs epochs, learning rate #(r.config.lr), batch size #(r.config.batch_size), trained by backpropagation through time (BPTT) with a surrogate gradient for the non-differentiable spike],
    [Regulariser], [upper firing-rate bound (threshold $theta_u = #(r.config.fr_reg_upper_theta)$, strength $s_u = #(r.config.fr_reg_upper_strength)$, values from Cramer et al.#cite(1)): without it the signed recurrence runs away],
  )

  The coarse $Delta t = #(r.config.dt_ms)$ ms is the fastest our conductance model tolerates
  (verified on MNIST); it is still ≈ 10× finer than the discrete leaky
  integrate-and-fire (LIF) timestep used by Cramer et al.#cite(1), because our
  fast synaptic and inhibitory time constants ($tau_"AMPA" = 2$ ms,
  $tau_(m,I) = 5$ ms) need resolving. Where:

  - $Delta t$, the integration timestep (the interval each simulation step advances);
  - $T$, the duration of one input utterance in simulated time;
  - $T_"steps" = T \/ Delta t$, the number of timesteps the network is unrolled over and backpropagated through per utterance;
  - $tau_"AMPA"$, the excitatory (AMPA) synaptic decay time constant;
  - $tau_(m,I)$, the membrane time constant of the inhibitory population;
  - $theta_u$, the firing rate above which the upper-bound regulariser penalises a neuron;
  - $s_u$, the strength (weight) of that regulariser penalty.

  == Results

  #figure(
    image("/artifacts/data/exp060/loss.svg", width: 100%,
      alt: "Training and test cross-entropy loss against epoch; training loss falls while test loss turns upward after the first few epochs."),
    caption: [
      Cross-entropy loss against training epoch, for the training set (solid
      black) and the held-out test set (dashed red). The horizontal axis is the
      training epoch; the vertical axis is mean cross-entropy loss. Training loss
      falls from #calc.round(r.first_epoch_loss, digits: 2) to
      #calc.round(r.final_loss, digits: 2); test loss falls with it for the first
      few epochs, then turns upward as the network overfits the
      #(r.n_train_subset)-sample subset.
    ],
  )

  #figure(
    image("/artifacts/data/exp060/accuracy.svg", width: 100%,
      alt: "Test accuracy against epoch, rising steeply over the first several epochs to a peak well above the " + str(calc.round(r.chance_pct)) + "% chance line, then plateauing."),
    caption: [
      Test accuracy against training epoch. The horizontal axis is the training
      epoch; the vertical axis is the percentage of held-out test utterances
      classified correctly. The dashed grey line marks the
      #calc.round(r.chance_pct)% chance floor (one of #(r.n_classes) classes). Accuracy
      climbs from #calc.round(r.first_epoch_acc_pct)% at epoch #r.first_epoch to a best of
      #calc.round(r.best_acc_pct)% at epoch #r.best_epoch, an order of magnitude
      above chance, then plateaus and wobbles (final
      #calc.round(r.final_acc_pct)%) as the subset overfits.
    ],
  )

  Both bars are cleared: accuracy reaches #calc.round(r.best_acc_pct)%, far above
  both the #calc.round(r.chance_pct)% floor and the plan's "decisively above
  chance" threshold of #calc.round(r.config.plan_threshold_pct)%. So the SHD data path, the spike-to-input binning, the
  all-four-blocks-trainable signed recurrence, and the surrogate-gradient
  training loop all work end to end. The firing-rate regulariser is load-bearing:
  without it the earlier unregularised run blew up to a runaway inhibitory firing
  rate; with it, rates stay physiological and the network learns.

  The overfitting is the expected cost of the small subset and is _why_ this is a
  smoke test, not a result: the honest ceiling number has to come from a
  full-scale run, where the test set is not swamped by a
  #(r.n_train_subset)-sample training set. That full-scale run is the plan's
  registered Rung A. Next: scale the subset up (and add mild early-stopping) for
  the number, then descend the ladder (the plan's sequence of successively more
  constrained models) to the Dale's-law-constrained baseline.

  #reference-list((
    (text: [Cramer, Stradmann, Schemmel & Zenke — _The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks_. IEEE Transactions on Neural Networks and Learning Systems, 2020.], doi: "10.1109/TNNLS.2020.3044364"),
  ))
]
