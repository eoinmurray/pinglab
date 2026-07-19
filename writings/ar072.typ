#let meta = (
  title: "Night shift — accelerating matched SHD learning",
  date: "2026-07-19",
  description: "A locked one-seed exploratory ladder that tests whether temporal discretization, shared input drive, or optimization can improve both matched Dale-constrained COBA and PING cells before a forty-epoch validation-only comparison.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 8,
  activity_trace: "ar073",
  queue: (
    (
      id: "exp070",
      hypothesis: "The first registered short-run intervention that clears the promotion gate will improve both COBA and PING selected held-out validation accuracy by at least three percentage points at forty epochs relative to each cell's exp069 trajectory through epoch forty.",
      kill: "Kill an attempt for non-finite loss or logits, any skipped update, silence, saturation, partition mismatch, official-test access, or failure of either cell to clear the registered short-run promotion gate. If all three short candidates are killed, stop without a forty-epoch run. Kill the final hypothesis if either promoted cell improves by less than three percentage points over its own exp069 through-epoch-forty selected validation baseline.",
      baseline: "exp069, seed 42, on the identical 7,340-utterance development-training and 816-utterance validation partition. Comparisons use exp069 at epoch five, its selected checkpoint through epoch forty, and its final eighty-epoch selected checkpoint; the official SHD test is unavailable.",
      seeds: 1,
      budget: "One local 128-training/128-validation two-epoch smoke per implemented candidate; at most three matched five-epoch full-data candidate pairs; then at most one matched forty-epoch promoted pair. Paid compute is forbidden until a runtime and cost estimate receives fresh authorization.",
      status: "running",
      origin: "human",
    ),
  ),
)

#let prior = json("/artifacts/data/exp069/numbers.json")
#let prior-c = json("/artifacts/data/exp069/raw/coba/metrics.json")
#let prior-p = json("/artifacts/data/exp069/raw/ping/metrics.json")
#let smoke = json("/artifacts/data/exp070/smoke_summary.json")

#let first-five(run) = run.epochs.at(4)
#let best-through-forty(run) = run.epochs.slice(0, 40).fold(
  (acc: -1.0, loss: 1e9, ep: 0),
  (best, row) => if row.acc > best.acc or
    (row.acc == best.acc and row.test_loss < best.loss) {
      (acc: row.acc, loss: row.test_loss, ep: row.ep)
    } else { best },
)

#let c5 = first-five(prior-c)
#let p5 = first-five(prior-p)
#let c40 = best-through-forty(prior-c)
#let p40 = best-through-forty(prior-p)

#let body = [
  == Digest

  *The experiment is registered but has not started.* The exp069 evidence
  indicates that adding epochs is too slow for iteration and does not identify
  the limiting mechanism. This night therefore uses a short, ordered
  intervention ladder. It promotes the first candidate that improves both
  cells, freezes that configuration, and tests it for forty epochs. No official
  SHD test data may be loaded or inspected.

  == Mandate

  === Question and fixed comparison

  Can one mechanistically motivated, matched change make both Dale-constrained
  conductance-based (COBA) and pyramidal-interneuron gamma (PING) networks learn
  SHD materially faster, while preserving a direct comparison between their
  recurrent cells?

  Both cells retain 256 excitatory and 64 inhibitory slots, identical input and
  readout dimensions, fixed Dale-constrained recurrence, batch size 32, seed
  42, the exp069 development split, membrane-mean classification objective,
  and the registered validation checkpoint rule. COBA disables its inhibitory
  loop and uses no voltage-gradient dampening. PING enables the loop and uses
  dampening 1000. Capacity, connectivity, recurrence structure, data ordering,
  and all settings not named by a candidate remain unchanged and matched.

  === Baseline trajectory

  #table(
    columns: (1fr, auto, auto, auto),
    table.header([*Cell*], [*Epoch-five accuracy*], [*Epoch-five loss*], [*Best accuracy through forty*]),
    [COBA], [#calc.round(c5.acc, digits: 2)%], [#calc.round(c5.test_loss, digits: 3)], [#calc.round(c40.acc, digits: 2)% at epoch #c40.ep],
    [PING], [#calc.round(p5.acc, digits: 2)%], [#calc.round(p5.test_loss, digits: 3)], [#calc.round(p40.acc, digits: 2)% at epoch #p40.ep],
  )

  The final exp069 selected checkpoints remain a required contextual
  comparison: COBA reached
  #calc.round(prior.cells.coba.selected_validation_accuracy_pct, digits: 2)%
  and PING reached
  #calc.round(prior.cells.ping.selected_validation_accuracy_pct, digits: 2)%
  after eighty epochs.

  === Registered intervention ladder

  1. *Temporal resolution.* Change the integration and input-bin width from
     1 ms to 2 ms while retaining the 1,000 ms observation window and every
     other exp069 setting. This halves backpropagation depth while remaining
     commensurate with the fastest registered synaptic time constant.
  2. *Shared input drive.* Only if candidate 1 is killed, restore 1 ms and
     increase the shared input-weight mean from 0.9 to 1.2. This tests whether
     weak PING excitatory recruitment limits early learning without tuning one
     architecture separately.
  3. *Optimization.* Only if candidates 1 and 2 are killed, restore the exp069
     input settings and increase the matched Adam learning rate from 0.0004 to
     0.001. No scheduler, weight decay, loss change, or architecture-specific
     optimizer is allowed.

  Stop the ladder at the first promoted candidate. Do not combine candidates,
  search intermediate values, or run later candidates after a promotion.
  Inhibitory balance, gradient flow, and existing time constants remain
  diagnostics. A new time-constant mechanism or readout would require a
  separate exact preregistration before implementation or results; this mandate
  does not authorize an unspecified rescue.

  === Promotion, final test, and interpretation

  Each candidate first passes a local 128-training/128-validation, two-epoch
  plumbing smoke, then runs both cells for exactly five epochs on the full
  development-training partition. Promote the first candidate only when, at
  epoch five, both cells exceed their own exp069 epoch-five validation accuracy
  by at least three percentage points, neither validation cross-entropy is
  worse, all losses and gradients are finite, no update is skipped, and the E
  population in both cells plus the PING I population remain active and below
  saturation.

  Freeze the promoted configuration before extending it. Train both frozen
  cells for exactly forty epochs from the registered seed and initialisation.
  Select checkpoints by highest validation accuracy, then lower validation
  cross-entropy, then earlier epoch. The primary criterion is an improvement of
  at least three percentage points for *each* cell over that cell's exp069
  selected checkpoint through epoch forty. Report the contemporaneous
  PING-minus-COBA difference as a secondary diagnostic, not as a seeded
  architecture claim. Also compare both cells with exp069's eighty-epoch
  selected checkpoints without selective checkpoint reporting.

  === Queue and operating contract

  #table(
    columns: (auto, 1fr),
    table.header([*Field*], [*exp070*]),
    [Hypothesis], [#meta.queue.first().hypothesis],
    [Baseline], [#meta.queue.first().baseline],
    [Kill criterion], [#meta.queue.first().kill],
    [Seed count], [#str(meta.queue.first().seeds)],
    [Budget], [#meta.queue.first().budget],
    [Status], [#meta.queue.first().status],
  )

  ```yaml
  budgets:
    local_smoke_samples: 128
    short_epochs: 5
    short_candidate_pairs_max: 3
    final_epochs: 40
    seeds_default: 1
    runpod_total_usd: pending_fresh_authorization
  scope:
    collection: spiking-heidelberg-digits
    experiment_id: exp070
    activity_trace: ar073
    tools_snn_edits: narrow_and_backward_compatible_only
    official_test_access: forbidden
    pr: 52
    main_edits: forbidden
  stop_when:
    - first_candidate_promoted_and_final_pair_complete
    - all_candidates_killed
    - paid_compute_authority_required
    - protocol_integrity_failure
    - invasive_engine_change_required
  ```

  Paid compute begins only after the local checks pass and the scientist
  approves an explicit estimate and hard ceiling. At most one pod per cell may
  be used under that approval, and every pod must be reaped immediately. Do not
  run the full test suite on the local 4 GB host. Use focused checks and the
  complete Demolab build.

  == Record

  === 2026-07-19 06:44–06:49 UTC: diagnosis and registration

  The scientist approved the exploratory objective, one-seed policy, locked
  comparison, intervention order, official-test prohibition, and fresh
  paid-compute gate in the attached goal. The existing pull request remains the
  requested programme branch; `main` is not modified. This branch checkpoint
  therefore serves as the pre-result anchor, a documented workflow deviation
  from placing a new-night mandate on `main`.

  Exp069's complete local histories show finite but slow learning in both
  cells, active populations, and no skipped updates. Its COBA input gradient is
  repeatedly clipped while PING's dampened gradient is much smaller, yet both
  validation losses continue falling. A development-data-only input audit
  found that extending the one-second window would recover too few events to
  explain the accuracy ceiling. Coarsening to 2 ms loses little same-channel
  event multiplicity and halves the recurrent unroll; 4 ms is already coarse
  relative to the AMPA decay. Candidate 1 was therefore fixed at 2 ms before
  any exp070 training result existed. No new runner, smoke, cloud job,
  official-test access, or spend exists at this checkpoint.

  === 2026-07-19 07:00–07:04 UTC: candidate-1 smoke gate

  The 2 ms candidate passed its registered local plumbing smoke. Both cells
  completed two finite epochs on 128 development-training and 128 validation
  utterances with the exp069 partition hashes and no skipped or non-finite
  updates. COBA's final excitatory rate was
  #calc.round(smoke.cells.coba.final_e_rate_hz, digits: 2) Hz. PING's final
  excitatory and inhibitory rates were
  #calc.round(smoke.cells.ping.final_e_rate_hz, digits: 2) Hz and
  #calc.round(smoke.cells.ping.final_i_rate_hz, digits: 2) Hz. The trainer used
  an experiment-specific development-data alias, so the official test cache
  was neither copied nor opened. Cloud spend remained 0 USD and no pod was
  created.

  === 2026-07-19 07:21 UTC: paid-compute authority

  The scientist authorized the proposed 10 USD hard total RunPod ceiling for
  exp070. The approved fleet permits at most two concurrent pods, exactly one
  per cell, with one-hour backstops for short candidates and two-hour
  backstops for a promoted final pair. Every pod must be reaped immediately.
  The fleet was empty, spend remained 0 USD, and the branch was clean at
  `2bc22d3` before the authorization checkpoint. Candidate 1 is the only
  authorized dispatch at this stage.

  === 2026-07-19 07:23 UTC: candidate-1 dispatch

  A clean dry run resolved to exactly two five-epoch jobs. One COBA and one
  PING 5090 pod were dispatched from `cd82023` at 0.99 USD per pod-hour. The
  verified fleet contained exactly those two pods, giving a 1.98 USD maximum
  exposure under the one-hour backstops. Both jobs use the same development
  partition and candidate settings; no official-test route exists.

  === 2026-07-19 07:31–07:35 UTC: candidate 1 killed

  Both pods self-terminated about eight minutes after dispatch, the active
  fleet returned to zero, and both complete five-epoch records were collected.
  The candidate was finite and active with no skipped steps or non-finite
  forward batches, but it failed the registered efficacy gate in both cells.

  #table(
    columns: (1fr, auto, auto, auto, auto),
    table.header([*Cell*], [*Accuracy*], [*vs exp069*], [*Cross-entropy*], [*vs exp069*]),
    [COBA], [23.53%], [-4.41 pp], [2.678], [+0.206],
    [PING], [27.70%], [-4.90 pp], [2.592], [+0.148],
  )

  Doubling the temporal bin width therefore made early validation learning
  worse, not faster. This is a scientific kill rather than a plumbing failure:
  COBA remained active at 14.22 Hz and PING remained active at 6.61 Hz E and
  29.11 Hz I. Candidate 2 remains next by registration, but has not been
  implemented or dispatched. Exact provider billing is still pending in the
  provider history; publication remains fail-closed rather than substituting
  an estimated charge.

  === 2026-07-19 07:41–07:47 UTC: candidate-2 smoke gate

  Candidate 2 was implemented only after the candidate-1 kill checkpoint. It
  restores the 1 ms baseline and changes only the shared input-weight mean from
  0.9 to 1.2. The runner rejects unregistered attempt names and keeps each
  attempt's cells, smoke, decision, and compute ledger separate. `tools/snn`
  remains unchanged.

  Its required 128-training/128-validation, two-epoch smoke passed for both
  cells with finite losses and gradients, no skipped updates, and the exact
  registered split hashes. COBA ended at 26.79 Hz E; PING ended at 3.24 Hz E
  and 17.00 Hz I. The development-only alias contained no official-test file.
  A cloud dry run found an empty fleet and exactly two candidate-2 jobs, one
  per cell; it created no pod. Paid candidate-2 dispatch remains pending this
  checkpoint and candidate-1 billing reconciliation.

  === 2026-07-19 07:49 UTC: candidate-2 dispatch

  From clean commit `04d5a28`, a second dry run again resolved to exactly two
  jobs and zero existing pods. One COBA and one PING 5090 pod were then
  dispatched at 0.99 USD per pod-hour with one-hour backstops. The verified
  fleet contained exactly those two pods. Candidate 1's exact provider rows
  remain delayed, but its hard exposure is at most 1.98 USD; adding candidate
  2 leaves a cumulative hard bound of 3.96 USD under the authorized 10 USD
  ceiling. Result and exact billing remain pending.
]
