#let meta = (
  title: "Night shift — extending matched SHD training",
  date: "2026-07-19",
  description: "A locked one-seed exploratory test of whether extending matched COBA and PING training from forty to eighty epochs improves held-out SHD validation accuracy without reusing the official test.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 6,
  activity_trace: "ar071",
  queue: (
    (
      id: "exp069",
      hypothesis: "Under the unchanged exp068 recipe, extending matched training from 40 to 80 epochs raises PING's selected held-out validation accuracy by at least 3 percentage points above its 43.50% exp068 baseline.",
      kill: "Kill either cell for non-finite loss or logits, persistent skipped updates, silence, clear saturation, or failure to complete the fixed protocol; kill the comparison for different partitions, mismatched registered settings, architecture-specific tuning, or any access to the official SHD test split.",
      baseline: "exp068 selected COBA epoch 25 at 40.81% and PING epoch 40 at 43.50% on the identical 816-utterance validation split. Its official-test results are context only and are not outcomes for exp069.",
      seeds: 1,
      budget: "One local plumbing smoke plus one 80-epoch full run per architecture; two concurrent pods maximum, 4 h backstop per pod, and 10 USD hard total RunPod spend.",
      status: "done",
      origin: "human",
    ),
  ),
)

#let r = json("/artifacts/data/exp069/numbers.json")
#let c = r.cells.coba
#let p = r.cells.ping

#let body = [
  == Digest

  *One registered experiment completed; none was killed or aborted.* Extending
  the unchanged matched recipe from forty to eighty epochs raised PING's
  selected validation accuracy to
  #calc.round(p.selected_validation_accuracy_pct, digits: 2)%, a
  #calc.round(r.primary.ping_change_from_exp068_pp, digits: 2)-point gain over
  its registered 43.50% baseline. This exceeds the preregistered +3-point
  criterion. COBA reached
  #calc.round(c.selected_validation_accuracy_pct, digits: 2)%, leaving PING
  #calc.round(r.primary.ping_minus_coba_validation_pp, digits: 2) points ahead
  contemporaneously.

  Both cells completed all eighty epochs with finite records, active
  populations, no skipped/non-finite updates, identical exp068 split hashes,
  and matched validation rasters. The official SHD test was not accessed. The
  only operational anomaly was one transient capacity miss before PING pod
  creation; the bounded retry succeeded without duplicate compute. Exact
  provider-billed spend was
  #calc.round(r.runpod.total_spend_usd, digits: 3) USD and zero pods remained.

  The result supports longer training as a useful exploratory improvement for
  this one recipe and seed. It does not establish a population-level PING
  advantage. The late selected epochs—75 for COBA and 79 for PING—suggest
  remaining headroom, but another duration change requires a new mandate rather
  than a post-result extension here.

  == Mandate

  === Purpose

  `exp068` produced a promising one-seed PING advantage, while PING's best
  validation checkpoint occurred at the final allowed epoch. This night tests
  the narrowest explanation: the matched recipe may simply be under-trained at
  forty epochs. It changes training duration only. It is exploratory and does
  not estimate a population-level architecture effect.

  === Registered experiment

  #table(
    columns: (auto, 1fr),
    table.header([*Field*], [*exp069*]),
    [Hypothesis], [#meta.queue.first().hypothesis],
    [Baseline], [#meta.queue.first().baseline],
    [Kill criterion], [#meta.queue.first().kill],
    [Seed count], [#str(meta.queue.first().seeds)],
    [Budget], [#meta.queue.first().budget],
    [Status], [#meta.queue.first().status],
  )

  === Fixed data and settings

  Reuse exp068's deterministic seed-42 joint speaker/class split: 7,340
  development-training and 816 validation utterances, with identical index
  hashes in COBA and PING. The official SHD test split must not be staged,
  loaded, evaluated, or inspected. Repeated official-test use during exploratory
  optimization would turn it into a development target.

  Preserve every exp068 setting except the registered duration change: 256
  excitatory and 64 inhibitory cells, fixed Dale-constrained recurrence,
  input-weight mean 0.9, 95% input sparsity, membrane-mean readout scaled by 225,
  Adam learning rate 0.0004, batch size 32, 1 ms timestep, 1,000 ms duration,
  and no firing-rate regularizer. COBA disables the inhibitory loop and uses no
  voltage-gradient dampening; PING enables the loop and uses dampening 1000.
  Initialisation, ordering, inputs, readout, and all other settings remain
  matched. No architecture-specific tuning or rescue is permitted.

  === Training, outcomes, and interpretation

  Run a local 128-training/128-validation, two-epoch plumbing smoke first. If
  both cells remain finite and active, train each full cell for exactly 80
  epochs. Select the checkpoint with highest validation accuracy; break ties by
  lower validation cross-entropy and then earlier epoch.

  The primary outcome is PING's selected validation-accuracy change relative to
  exp068's 43.50% baseline. At least +3 percentage points is a useful longer-
  training signal; a positive gain below +3 points is weak; zero or negative
  gain provides no evidence that forty epochs caused the prior ceiling.

  Registered diagnostics are COBA's change from 40.81%, the contemporaneous
  PING-minus-COBA validation difference, full learning and activity curves,
  selected epoch and loss, E/I firing rates, skipped and non-finite batches,
  runtime, peak GPU memory, and matched validation input/E/I rasters. None is a
  substitute primary outcome.

  === Operating contract

  ```yaml
  budgets:
    runpod_total_usd: 10
    max_pods_concurrent: 2
    wall_clock_backstop_per_pod: 4h
    seeds_default: 1
  scope:
    collection: spiking-heidelberg-digits
    experiment_id: exp069
    activity_trace: ar071
    may_edit_snn_cli: false
    official_test_access: forbidden
  stop_when:
    - queue_empty
    - runpod_budget_exhausted
    - protocol_integrity_failure
    - build_red_twice
  ```

  Use `experiments/exp069.py` and existing CLI capabilities. Do not edit
  `tools/snn`. Use focused lightweight checks and `demolab build`; do not run
  the full test suite on the local 4 GB host. Two concurrent pods, one per
  architecture, are authorized within the 10 USD hard ceiling. Reap them after
  completion or failure. Stop for authority before changing this design,
  exceeding the budget, or taking an action outside this scope.

  == Record

  === 2026-07-19 00:45–00:49 UTC: publication accepted and mandate approved

  The scientist accepted exp068 by merging pull request 51. Its merge commit is
  `569926f`. The next experiment was then narrowed to training duration alone,
  with validation-only outcomes so the already observed official test does not
  become an optimization target. The scientist approved this complete contract
  at 00:48:51 UTC. No exp069 runner, compute, result, or cost existed at approval.

  === 2026-07-19 00:51 UTC: mandate anchor and night branch

  The approved mandate and both reserved article identifiers were committed to
  `main` at `3c440aa`. The dedicated night branch was created from that exact
  anchor. At branch creation, exp069 spend remained 0 USD, no pod was active,
  and the official SHD test remained outside the registered experiment scope.

  === 2026-07-19 01:35–01:40 UTC: implementation and smoke gate

  The validation-only runner was implemented without changing `tools/snn`.
  Unlike exp068, it contains no official-test download, staging, loading, or
  evaluation route; its matched rasters use fixed validation examples. Focused
  Ruff, type, and compilation checks passed. The registered 128-training /
  128-validation, two-epoch smoke passed for both cells with finite losses,
  zero skipped or non-finite updates, active COBA E firing at 23.93 Hz, and
  active PING E/I firing at 2.95 / 14.56 Hz. The derived development split
  hashes exactly matched exp068. No pod had been created and spend remained
  0 USD at the smoke gate.

  === 2026-07-19 01:42–01:43 UTC: full-cell dispatch

  Commit `b944ebf` pinned the passing smoke implementation before compute.
  The dispatch dry-run resolved to exactly one COBA and one PING cell. Two
  single-GPU pods were then created concurrently at 0.99 USD/hour each with
  four-hour backstops. One transient capacity miss on the PING request was
  resolved by the registered provisioning retry; no extra pod was created.
  The verified fleet count was two, and worst-case exposure was 7.92 USD.

  === 2026-07-19 04:01–05:01 UTC: completion, collection, and billing

  Both registered cells completed all eighty epochs and self-terminated before
  their four-hour backstops. PING selected epoch #p.selected_epoch at
  #calc.round(p.selected_validation_accuracy_pct, digits: 2)% validation
  accuracy, improving by
  #calc.round(r.primary.ping_change_from_exp068_pp, digits: 2) points over its
  baseline and meeting the +3-point primary criterion. COBA selected epoch
  #c.selected_epoch at
  #calc.round(c.selected_validation_accuracy_pct, digits: 2)%, so the matched
  contemporaneous gap was
  #calc.round(r.primary.ping_minus_coba_validation_pp, digits: 2) points.

  Artifact integrity checks found exactly eighty finite epoch records per cell,
  zero skipped or non-finite updates, active selected populations, exp068's
  exact train/validation index hashes, byte-identical raster inputs, and no
  failure marker. The official SHD test remained absent from the runner. The
  provider billing history settled at
  #calc.round(r.runpod.total_spend_usd, digits: 3) USD, below the 10 USD ceiling;
  collection and a final provider query both confirmed zero active pods.
]
