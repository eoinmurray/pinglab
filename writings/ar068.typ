#let meta = (
  title: "Night shift — full-development SHD comparison",
  date: "2026-07-18",
  description: "A one-seed exploratory comparison of matched Dale-constrained COBA and PING networks using a speaker-and-class-stratified SHD development split and a sealed official test evaluation.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 4,
  activity_trace: "ar069",
  queue: (
    (
      id: "exp068",
      hypothesis: "With matched training and readout settings on substantially more SHD development data, PING exceeds COBA by at least two percentage points in overall official-test accuracy while using less excitatory spike activity.",
      kill: "The accuracy hypothesis is falsified if PING does not exceed COBA on overall official-test accuracy. Kill either cell for non-finite loss or logits, persistent skipped updates, silence, clear saturation, or failure to complete the fixed protocol; kill the comparison for different partitions, mismatched registered settings, or any use of official-test data before both checkpoints are frozen.",
      baseline: "exp066 established one-seed feasibility on 1,000 training utterances: COBA reached 31.0% best held-out accuracy and PING 34.7%. The present contemporaneous matched COBA cell is the architectural baseline; the 5% SHD chance floor remains a diagnostic reference.",
      seeds: 1,
      budget: "One local plumbing smoke plus one 40-epoch full run per architecture; two concurrent pods maximum, 3 h target per pod, and 10 USD hard total RunPod spend.",
      status: "queued",
      origin: "human",
    ),
  ),
)

#let body = [
  == Digest

  _Empty until the registered experiment finishes._

  == Mandate

  === Purpose and scope

  This one-seed exploratory shift asks whether PING's modest point advantage in
  `exp066` becomes clearer when both architectures train on substantially more
  Spiking Heidelberg Digits (SHD) data under a leakage-resistant
  train/validation/test protocol. It is intended to identify a claim worth
  defending later, not to provide multi-seed confirmatory evidence.

  The primary question is whether PING outperforms matched COBA in overall
  official-test accuracy. Accuracy relative to excitatory activity is reported
  transparently as paired accuracy and firing-rate measurements; no post-hoc
  composite score will be invented.

  === Registered experiment

  #table(
    columns: (auto, 1fr),
    table.header([*Field*], [*exp068*]),
    [Hypothesis], [#meta.queue.first().hypothesis],
    [Baseline], [#meta.queue.first().baseline],
    [Kill criterion], [#meta.queue.first().kill],
    [Seed count], [#str(meta.queue.first().seeds)],
    [Budget], [#meta.queue.first().budget],
    [Status], [#meta.queue.first().status],
  )

  === Data partition and sealed evaluation

  Use SHD's official 8,156-utterance training split and 2,264-utterance test
  split. At seed 42, partition the official training split deterministically
  into approximately 90% development training and 10% validation, stratified
  jointly by class and speaker. COBA and PING receive byte-identical index
  sets, preprocessing, and sample-ordering policy.

  The complete official test split remains sealed during implementation,
  debugging, training, hyperparameter selection, stopping, and checkpoint
  selection. After both training runs have finished and both checkpoint choices
  are frozen, evaluate each selected checkpoint exactly once on the full
  official test set. Neither result may be revealed or acted upon before both
  checkpoints are frozen. Overall official-test accuracy is primary. Macro
  class accuracy and accuracy for seen versus unseen speakers are registered
  diagnostics; SHD test speakers 4 and 5 form the unseen-speaker group.

  === Matched architectures and optimisation

  Both cells use 256 excitatory and 64 inhibitory cells, Dale-constrained fixed
  recurrent weights, input-weight mean 0.9, 95% input sparsity, membrane-mean
  readout with scale 225, Adam learning rate 0.0004, batch size 32, 1 ms
  integration timestep, 1,000 ms simulation duration, and no firing-rate
  regulariser. Initialisation, data, preprocessing, ordering, input weights,
  readout, and every other applicable setting remain matched.

  COBA disables the inhibitory loop and uses no voltage-gradient dampening.
  PING enables the fixed inhibitory loop and uses voltage-gradient dampening
  1000. No architecture-specific tuning or rescue is permitted after either
  result is seen.

  === Training and checkpoint selection

  First run a small local staged train/validation smoke test to verify the
  partition, data plumbing, tensor shapes, finite optimisation, checkpointing,
  and active populations. Smoke measurements are not scientific evidence.

  Each full cell then trains for exactly 40 epochs. Select the checkpoint with
  highest validation accuracy; break equal-accuracy ties with lower validation
  cross-entropy and then the earlier epoch. The experiment runner must preserve
  this rule explicitly rather than relying on an incompatible default. Only
  after both checkpoint identities are recorded may the sealed official-test
  evaluation begin.

  === Outcomes and interpretation

  The primary exploratory result is
  $delta_("accuracy") = "PING accuracy" - "COBA accuracy"$ on the complete
  official test split.

  + If the difference is at least +2 percentage points, the result identifies
    a promising PING accuracy claim for a later confirmatory study.
  + A positive difference below +2 points is a weak directional signal, not a
    useful accuracy claim.
  + A zero or negative difference provides no PING accuracy advantage under
    this recipe.

  Registered secondary diagnostics are official-test cross-entropy,
  validation accuracy and loss curves, macro class accuracy, seen- and
  unseen-speaker accuracy, excitatory and inhibitory firing rates, accuracy
  alongside excitatory spike rate, runtime, peak GPU memory, non-finite
  batches, skipped updates, and directly matched input/E/I rasters for
  preselected official-test utterances.

  === Integrity and stopping rules

  Kill and record a cell if loss or logits become non-finite, skipped updates
  persist, activity is silent or clearly saturated, or the fixed protocol
  cannot complete. Kill the comparison if the cells receive different
  partitions or any registered setting differs. Failed and killed attempts are
  retained rather than overwritten. The question, baseline, primary outcome,
  interpretation bands, and kill criteria cannot change after results are
  observed.

  === Operating contract

  ```yaml
  budgets:
    wall_clock_target_per_pod: 3h
    wall_clock_per_shift: 8h
    runpod_total_usd: 10
    max_pods_concurrent: 2
    seeds_default: 1
  scope:
    collection: spiking-heidelberg-digits
    may_propose: true
    may_edit_snn_cli: false
    experiment_id: exp068
    activity_trace: ar069
  stop_when:
    - queue_empty
    - shift_budget_exhausted
    - runpod_budget_exhausted
    - protocol_integrity_failure
    - build_red_twice
  ```

  Two concurrent RunPod pods are authorized: one for COBA and one for PING,
  preferably using RTX 5090 or an equivalently suitable GPU. Total experiment
  spend may not exceed 10 USD. Both pods must be reaped immediately after
  collection or failure. Stop for additional authority before exceeding the
  budget or pod limit, editing `tools/snn`, changing this design, or taking any
  other action outside the approved scope.

  Implementation should use `experiments/exp068.py` and existing CLI
  capabilities. Runner-controlled staging of deterministic SHD
  train/validation files is permitted when provenance is retained and the
  untouched official test split is used only for final evaluation. Use focused
  lightweight checks and `demolab build`; do not run the full test suite on the
  4 GB host.

  == Record

  _Empty until the mandate is approved, committed on `main`, and execution
  begins._
]
