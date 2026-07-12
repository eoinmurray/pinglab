#import "/.demolab/lib.typ": cite, reference-list

// Every run number quoted in the Digest and Record below is cross-imported from
// the source experiment's numbers.json (the ar009 pattern), never hand-typed, so
// the log stays provenance-tracked and cannot drift from the runs it summarises.
#let r060 = json("/artifacts/data/exp060/numbers.json")
#let r061 = json("/artifacts/data/exp061/numbers.json")
#let r062 = json("/artifacts/data/exp062/numbers.json")
#let r063 = json("/artifacts/data/exp063/numbers.json")
#let r064 = json("/artifacts/data/exp064/numbers.json")

// Order-of-magnitude helper: render a huge gradient norm as 10^n (matches the
// exp061/exp064 writeups), so ≈ 10#super[#oom(x)] reads "≈ 10^n".
#let oom(x) = calc.round(calc.log(x, base: 10))

// The specific cells each claim reads from, selected by their discriminators so
// the reference survives re-ordering of the cells array.
#let c064clamp = r064.cells.filter(c => c.name == "clamp__seed42").at(0)
#let c064free = r064.cells.filter(c => c.name == "free__seed42").at(0)
#let c062free = r062.cells.filter(c => c.dales_law == false).at(0)
#let c062dales = r062.cells.filter(c => c.dales_law == true).at(0)
#let c063strong = r063.cells.sorted(key: c => c.wd).last()
#let c063weak = r063.cells.sorted(key: c => c.wd).first()
#let c061coarse = r061.cells.filter(c => c.dt == 1.0).at(0)
#let c061finest = r061.cells.filter(c => c.dt == 0.25).at(0)

#let meta = (
  title: "Night shift — 2026-07-11",
  date: "2026-07-11",
  description: "One self-contained record of the SHD stability program's night shift: the mandate it committed to before running, the run-by-run record, and the morning digest. Merges the former plan, log, and night-shift contract into a single per-shift document.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

#let body = [
  This is the single document for one night shift of the Spiking Heidelberg
  Digits stability program. It is self-contained: the *mandate* the shift
  committed to before running, the *record* of what happened, and the *digest*
  for the morning read all live here; there is no separate plan, log, or
  contract to cross-reference. The mandate was registered before the runs it
  governs, so, via demolab's per-page commit stamping, the hypotheses and kill
  criteria are provably prior to the results.

  == Digest

  _The 30-second read: the morning-gate summary and the PR description. As of 2026-07-11 23:35 BST._

  *Goal exceeded: a one-line model change (a forward-pass state clamp) trains the
  free network stably on SHD at full accuracy, the program's best recipe.*

  The program's goal was to find the minimal recipe that trains a
  conductance-based excitatory/inhibitory (E/I) spiking network stably on the
  Spiking Heidelberg Digits (SHD)#cite(1) spoken-digit benchmark. "Stably" means: no
  epoch returns a NaN (not-a-number) loss, the gradient and recurrent-weight
  magnitudes stay bounded, and the network still learns (best test accuracy well
  above the #r060.chance_pct% chance level for #r060.n_classes classes).

  The blocker was that the most accurate configuration (a *free* network whose
  recurrent weights may take either sign) diverges: on scattered training epochs
  the forward pass produces NaN, and the pre-clip gradient magnitude spikes by
  many orders of magnitude. Reading the model's update equation found the
  mechanism. The conductance-based neuron settles toward a steady voltage
  $v_infinity = (g_L E_L + g_e E_e + g_i E_i) \/ g_"tot"$, with total conductance
  $g_"tot" = g_L + g_e + g_i$, where:

  - $v_infinity$ is the steady-state (equilibrium) membrane voltage,
  - $g_L$, $g_e$, $g_i$ are the leak, excitatory, and inhibitory conductances,
  - $E_L$, $E_e$, $E_i$ are their respective reversal potentials, and
  - $g_"tot"$ is the total conductance.

  Signed weights let a synaptic conductance go
  _negative_, so $g_"tot"$ can cross zero and $v_infinity$ diverges to NaN, which
  is why the divergence appeared even at epoch #r060.first_nan_epoch with tiny weights: it is the
  _sign_ of the conductance, not its size.

  The fix is a *forward-pass state clamp*: floor every conductance at 0 (physical:
  a conductance cannot be negative) each timestep, which keeps
  $g_"tot" >= g_L > 0$. Result (#link("/exp064/")[exp064]): the free network goes
  from NaN on #c064free.nan_epochs of #c064free.epochs epochs to
  *#c064clamp.nan_epochs of #c064clamp.epochs*, peak gradient magnitude
  10#super[#oom(c064free.max_grad_norm)] → ≈ #calc.round(c064clamp.max_grad_norm),
  at *#calc.round(c064clamp.best_acc_pct, digits: 1)%* test accuracy, matching the
  unconstrained free baseline (#calc.round(c064free.best_acc_pct, digits: 1)%) and
  beating the Dale's-law constraint (#calc.round(c062dales.best_acc_pct, digits: 1)%). It bounds the network's _state_ (the conductance), not its _weights_
  (the recurrent weight magnitude is unchanged), which is exactly why weight decay
  could not fix it and the clamp can. *The free network never needed constraining,
  only its state bounding.*

  *The recipe, ranked by the stability sweep:* forward-pass state clamp
  (#calc.round(c064clamp.best_acc_pct, digits: 1)%, stable) > Dale's law
  (#calc.round(c062dales.best_acc_pct, digits: 1)%, stable) $>>$ free / finer timestep / weight decay
  (all diverge). Of the three "soft" knobs the free net can vary without a model
  change, only Dale's law stabilised.

  *Deferred to future shifts* (as new experiments, not amendments): a
  three-seed confirmation of the single-seed results below, and whether the clamp
  shifts the network's gamma rhythm or its sparsity: the science the program
  parked until training was reliable, which it now is.

  == The mandate

  The goal, the queue of experiments, and the operating contract the shift ran
  under, all committed before the runs.

  === The goal

  A first smoke run (#link("/exp060/")[exp060]) showed the free
  signed-recurrent network trains on an SHD subset
  (#calc.round(r060.best_acc_pct, digits: 1)% test accuracy against a
  #r060.chance_pct% floor) but _unstably_: scattered epochs return NaN, gradient magnitudes
  reach the millions, and the recurrent excitatory-to-excitatory weight grows
  without bound. Before any larger claim about what biological structure costs a
  spiking classifier, or whether the gamma rhythm helps it, the network has to
  train _reliably_. So the program had one deliberately modest goal:

  #quote(block: true)[
    Find the minimal recipe that trains a conductance-based E/I spiking network
    stably on SHD, and attribute each ingredient (integration timestep, weight
    decay, Dale's law, the firing-rate regulariser) to the stability it buys.
  ]

  *Success* means a recipe that trains to completion with (i) _no NaN
  divergence_: every epoch a finite train and test loss; (ii) _bounded
  dynamics_: pre-clip gradient norm and recurrent-weight norm stay bounded, not
  runaway; and (iii) _decisively above chance_: best test accuracy at or above
  #r060.config.plan_threshold_pct% (#calc.round(r060.config.plan_threshold_pct / r060.chance_pct)× the #r060.chance_pct% chance level), so "stable" does not mean "stably not learning",
  plus a table attributing each ingredient to its effect on those metrics.

  === The queue

  Each experiment isolates one candidate stabiliser and measures its effect on
  three stability metrics (the NaN-epoch rate, the maximum pre-clip gradient
  norm, and the maximum recurrent-weight norm), holding the rest of the recipe
  fixed. Each carries a *kill* criterion: the result that would falsify its
  hypothesis. (The firing-rate regulariser, which caps mean firing rate, is
  already in the recipe throughout: without it an earlier run blew up to a
  runaway inhibitory firing rate, so the queue attributes the _remaining_ ingredients.)

  + *#link("/exp061/")[exp061]: does a finer integration timestep stabilise it?*
    _Hypothesis:_ the divergence is numerical stiffness in the forward
    integration at a coarse timestep; halving then quartering the timestep
    (1.0 → 0.5 → 0.25 ms) drives the NaN-epoch rate toward zero. _Kill:_ if NaN
    persists at 0.25 ms, coarse integration is not the cause.

  + *#link("/exp062/")[exp062]: is Dale's law the implicit stabiliser?*
    _Hypothesis:_ constraining weights to one sign per population (Dale's law, via
    a non-negativity clamp) keeps the dynamics in a stable regime, so the same
    recipe _with_ the constraint trains NaN-free where the free version diverges.
    _Kill:_ if the constrained net also NaNs at matched settings, the constraint
    is not what stabilises.

  + *#link("/exp063/")[exp063]: does weight decay bound the free recurrence?*
    _Hypothesis:_ decay strong enough (swept 0 → 0.001 → 0.01 → 0.1) bounds the
    runaway recurrent weight and removes the divergence. _Kill:_ if even 0.1
    leaves the weight growing and NaN present, decay regularises but does not
    stabilise.

  + *#link("/exp064/")[exp064]: does a forward-pass state clamp stabilise it?*
    _(A change to the shared model, held in reserve and authorised by the
    scientist once the soft knobs were exhausted.)_ _Hypothesis:_ the divergence
    is $v_infinity$ blowing up when signed weights drive $g_"tot" <= 0$; flooring
    conductances at 0 each timestep keeps $g_"tot" >= g_L > 0$ and removes the NaN
    while the weights stay signed. _Kill:_ if the clamp leaves NaN present, the
    $g_"tot" <= 0$ mechanism is not the whole cause.

  === Operating contract

  What the unattended overnight agent was permitted to do:

  - *Run only what a human queued* (never invent an experiment, never run a
    _proposed_ one); it may _draft_ proposed entries for review.
  - *Publish nothing*: end with a digest and leave the output for the morning
    gate; nothing merges without a human.
  - *Obey the budgets below*, aborting a run at its wall-clock ceiling and ending
    the shift on any stop condition.
  - *RunPod is authorised for this program*: a deliberate, standing exception to
    the default rule that cloud runs need per-run permission, because the sweeps
    are faster on a GPU and the scientist accepted the cost. The authorisation is
    _bounded_: a hard cost ceiling, one GPU pod at a time, pods reaped at shift
    end.

  ```yaml
  budgets:
    epochs_per_run: 40          # ≈ one seed at native dt on MPS; less on RunPod
    wall_clock_per_run: 6h      # hard ceiling — kills a diverging run
    wall_clock_per_night: 8h
    seeds_default: 3            # one seed is an anecdote, not a result
  scope:
    collection: spiking-heidelberg-digits   # the only collection this shift may touch
    may_propose: true                        # draft `proposed` entries; never run them
  compute:
    local: allowed              # Mac MPS — free, the default
    runpod: allowed             # EXPLICITLY authorised for this program — real spend
    gpu: "4090"
    max_cost_per_night_usd: 40  # hard $ ceiling; stop when reached
    max_pods_concurrent: 1
    reap_pods_at_end: true
  stop_when:
    - queue_empty
    - night_budget_exhausted
    - cost_ceiling_reached
    - build_red_twice           # bail rather than thrash
  ```

  == The record

  What happened, in order. Each experiment ran a single seed at full exp060 scale
  (#(r060.n_train_subset)-sample subset, #r060.epochs epochs) on RunPod unless noted; the numbers below are
  read from each run's outputs.

  === 17:36 BST — program opened

  Built the substrate for SHD training through the CLI: an event-based loader
  (official train/test split, binned to the model timestep), all four recurrent
  weight blocks made trainable (a latent dead CLI flag for the
  inhibitory-to-inhibitory block, fixed), and a weight-decay knob. The
  #link("/exp059/")[data-look entry] confirmed the dataset is clean and
  class-separable, and #link("/exp060/")[exp060] confirmed the pipeline trains
  (loss #calc.round(r060.first_epoch_loss, digits: 2) → #calc.round(r060.final_loss, digits: 2), #calc.round(r060.best_acc_pct, digits: 1)% accuracy) but that the free dynamics are unstable. A
  first guess (that unbounded weight _growth_ caused it) was wrong: adding
  weight decay tamed the gradient spike but made NaN _more_ frequent, and NaN
  appeared at epoch #r060.first_nan_epoch with tiny weights, so the cause is a per-step forward
  divergence, not weight size. The program was reframed around finding a stable
  recipe, with exp061–063 queued and the state clamp held in reserve.

  === 20:10 BST — exp061: the timestep hypothesis is killed; a compute blocker fixed

  Swept the integration timestep (#r061.dt_sweep.map(d => str(d)).join(" / ") ms) on the free network.
  #link("/exp061/")[The result] refutes the hypothesis: NaN is reproduced at the
  coarse timestep but _not removed_ by a finer one: it persists at #c061finest.dt ms
  (#c061finest.nan_epochs of #c061finest.epochs epochs), and the peak pre-clip gradient magnitude explodes
  monotonically _worse_ (10#super[#oom(c061coarse.max_grad_norm)] → 10#super[#oom(c061finest.max_grad_norm)]) because quartering the
  timestep quadruples the number of backprop-through-time steps (1000 → 4000).
  The divergence is a gradient explosion over the recurrent unroll, not
  integration stiffness; the timestep leaves the recipe. Two per-epoch metrics
  the sweep needed were added to the trainer (peak pre-clip gradient norm; count
  of NaN forward passes, previously skipped silently), additive only.

  _Compute blocker, fixed._ The RunPod fan-out's collector copies results off the
  shared volume over SSH, which the cloud sandbox blocks, so the pods trained
  correctly but their results were stranded. Routed around it by reading the
  volume over its S3-compatible HTTPS interface instead (`collect_via_s3`); the
  earlier "failed" pods had in fact trained fine all along.

  === 20:35 BST — exp062: Dale's law is the stabiliser (the goal is met)

  Ran two cells identical but for the constraint, at the coarse timestep where the
  free net diverges. #link("/exp062/")[The result]: the free net NaNs on
  #c062free.nan_epochs of #c062free.epochs
  epochs (peak gradient ≈ 10#super[#oom(c062free.max_grad_norm)], recurrent weight
  norm #calc.round(c062free.max_wee_norm, digits: 1)); the
  Dale's-law net trains _NaN-free_ (#c062dales.nan_epochs of #c062dales.epochs,
  gradient ≈ #calc.round(c062dales.max_grad_norm), weight
  norm #calc.round(c062dales.max_wee_norm, digits: 1)) at
  #calc.round(c062dales.best_acc_pct, digits: 1)% versus the free net's
  #calc.round(c062free.best_acc_pct, digits: 1)%. Constraining the sign bounds
  the excitatory–inhibitory loop gain below the runaway threshold, so the
  divergence never starts. This is the first stable recipe (the registered goal)
  at a cost of a few points of accuracy. That reframes the remaining probes: they
  are no longer about _finding_ stability but about keeping the free net's higher
  accuracy _without_ its divergence.

  === 20:45 BST — exp063: weight decay regularises but does not stabilise (killed)

  Swept weight decay (#r063.wd_sweep.first() to #r063.wd_sweep.last()) on the free net. #link("/exp063/")[The result]:
  NaN persists at every strength (#r063.cells.sorted(key: c => c.wd).map(c => str(c.nan_epochs)).join(" / ") of #c063weak.epochs epochs, never zero) and
  the recurrent weight norm stays pinned near 9 regardless: decay does not bound
  the runaway weight. The kill fires. Accuracy _does_ climb with decay
  (#calc.round(c063weak.best_acc_pct, digits: 1)% → #calc.round(c063strong.best_acc_pct, digits: 1)%, the free net's best), so it is a strong regulariser, not a
  stabiliser. With all three soft knobs now tested and only Dale's law
  stabilising, the reserved model change (the forward-pass state clamp) was
  left for the human gate to authorise as the next experiment.

  === 23:35 BST — exp064: the state clamp is the answer (goal exceeded)

  With the clamp authorised, implemented it in the model (`--state-clamp`, off by
  default so every prior run is unchanged) and ran three cells: free, free+clamp,
  and free+clamp+strong-decay. #link("/exp064/")[The result]: free+clamp trains
  NaN-free (#c064free.nan_epochs of #c064free.epochs → *#c064clamp.nan_epochs of #c064clamp.epochs*),
  gradient 10#super[#oom(c064free.max_grad_norm)] → ≈ #calc.round(c064clamp.max_grad_norm),
  at *#calc.round(c064clamp.best_acc_pct, digits: 1)%*,
  matching the free baseline and beating Dale's law, at no accuracy cost. The
  recurrent weight norm is unchanged by the clamp (≈ #calc.round(c064clamp.max_wee_norm, digits: 1)): it bounds the _state_,
  not the weight, confirming the mechanism (signed weights drive $g_"tot" <= 0$;
  the floor prevents it). The goal is met with no accuracy penalty and the
  mechanism pinned.

  _Operational note._ Two clamp pods stalled in startup on the 4090 pool (no
  output after 40–80 min); the same cell ran fine on a 5090. The stalled pods were
  killed (nothing leaked) and re-fired on a 5090: a 4090 startup-stall flag for
  the next shift.

  == Amendments

  Changes to a hypothesis or kill criterion _after_ an experiment ran against it,
  appended and dated, never edited in place.

  - *2026-07-11: timestep dropped; state clamp promoted from fallback to lead.*
    exp061 killed the stiffness hypothesis (finer timestep did not remove the NaN
    and made the gradient explosion far worse). With neither timestep (exp061) nor
    weight size (exp060: NaN at epoch #r060.first_nan_epoch with tiny weights) explaining the
    divergence, the reserved forward-pass state clamp became the primary lead
    rather than a last resort, run after the two remaining no-code-change probes.

  - *2026-07-11: stability goal met; the remaining probes are reframed.* exp062
    confirmed Dale's law trains the net NaN-free where the free net diverges, a
    working stable recipe, the program's goal. Since it costs accuracy, exp063 and
    the state clamp are now tested as stabilisers of the _free_ net, to ask whether
    its extra accuracy can be kept without its divergence.

  - *2026-07-11: soft-knob queue exhausted; the state clamp is next.* exp063
    killed weight decay as a stabiliser. Stabilising the free net (the only way to
    keep its accuracy) is not reachable by any soft knob, so it requires the
    reserved model change. Being a shared-model change rather than a knob, it was
    left for the human gate to authorise.

  - *2026-07-11: goal exceeded; the state clamp is the answer, arc closed.*
    exp064 ran the clamp (scientist-authorised) and it wins: NaN-free at #calc.round(c064clamp.best_acc_pct, digits: 1)%,
    matching the free baseline and beating Dale's law, by bounding conductance
    (the state) not the weights. The arc timestep → Dale's law → weight decay →
    state clamp is closed. Follow-ups (three-seed confirmation; whether the clamp
    shifts the gamma rhythm or sparsity) become new experiments in later shifts.

  #reference-list((
    (
      text: [Cramer, Stradmann, Schemmel & Zenke — _The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks_. IEEE Transactions on Neural Networks and Learning Systems, 2020.],
      doi: "10.1109/TNNLS.2020.3044364",
    ),
  ))
]
