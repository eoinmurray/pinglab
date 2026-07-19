#let meta = (
  title: "Trying a cumulative-potential readout on matched SHD networks",
  date: "2026-07-19",
  description: "A one-seed validation-only exploratory ladder tests whether the newly added signed cumulative-potential decoder and modest capacity changes raise matched COBA/PING SHD learning.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 10,
)

#let r = json("/artifacts/data/exp071/numbers.json")
#let trace = json("/artifacts/data/exp071/activity/messages_cp001.json")
#let trace-two = json("/artifacts/data/exp071/activity/messages_cp002.json")

#let maybe-pct(x) = {
  if type(x) == float or type(x) == int {
    [#calc.round(x, digits: 2)%]
  } else {
    [pending]
  }
}

#let verbatim-prose(value) = {
  for (index, line) in value.split("\n").enumerate() {
    if index > 0 { linebreak() }
    text(size: 8.6pt, line)
  }
}

#let message-card(message, checkpoint) = {
  let speaker = if message.role == "user" { "User" } else { "Assistant" }
  block(width: 100%, breakable: true, fill: luma(96%), stroke: 0.6pt + luma(82%),
    radius: 4pt, inset: 9pt)[
    #grid(columns: (1fr, auto), [#strong(speaker)],
      [#text(size: 8pt, fill: luma(42%), message.timestamp)])
    #v(3pt)
    #text(size: 7.5pt, fill: luma(48%))[
      Role: #message.role · Checkpoint: #checkpoint.checkpoint_id · Session: #checkpoint.session_id
    ]
    #v(6pt)
    #verbatim-prose(message.content)
  ]
  v(8pt)
}

#let body = [
  == Abstract

  This one-seed exploratory experiment tests whether the newly merged
  `tools/snn` cumulative-potential decoder can move matched Dale-constrained
  COBA and PING networks above the exp068–exp070 validation plateau on SHD.
  The screening ladder uses the same deterministic development-training /
  held-out-validation split as exp069 and exp070; the official SHD test remains
  sealed. Candidate one keeps the baseline 256-cell architecture and changes
  the classifier to `--readout cumulative-potential --signed-readout
  --readout-bias`. If that is finite, active, and promising, candidate two keeps
  the same decoder and tests two 256-cell hidden layers. Only the best short
  candidate may be promoted to forty epochs.

  == Goal prompt

  #raw(read("/artifacts/data/exp071/goal.txt"), block: true)

  == Methods

  === Locked comparison

  Both cells use seed #r.seed, the exp069/exp070 development split
  (#r.split.development_train_count training utterances and
  #r.split.validation_count validation utterances), batch size
  #r.config.batch_size, Adam learning rate #r.config.learning_rate,
  #r.config.dt_ms ms simulation steps, #r.config.t_ms ms utterance windows,
  fixed Dale-constrained feed-forward/recurrent conductances, and identical
  input preprocessing. COBA disables the inhibitory loop and uses the engine's
  no-dampening value; PING enables the inhibitory loop and uses voltage-gradient
  dampening 1000. These cell-specific settings are the only intended
  COBA/PING difference inside each candidate.

  The readout is the new cumulative-potential decoder with signed abstract
  classifier weights and a trainable readout bias. The readout initialisation
  scale is one, not the old membrane-readout scale of 225, because that scale
  belonged to the prior output-LIF membrane recipe. COBA and PING are still
  matched exactly within each candidate.

  === Ordered exploratory ladder

  #table(
    columns: (auto, 1.4fr, auto, auto),
    table.header([*Order*], [*Candidate*], [*Short epochs*], [*Status*]),
    [1], [Baseline 256-cell architecture plus cumulative-potential readout],
    [#r.screening_epochs], [registered],
    [2], [Two hidden layers, 256 cells each, same readout],
    [#r.screening_epochs], [conditional],
  )

  Candidate selection uses held-out validation accuracy and cross-entropy only.
  Training must remain finite, active, non-saturated, and free of skipped or
  non-finite updates. The official test file is not staged or loaded by the
  runner.

  == Current evidence

  #if r.result_status == "preregistered" [
    No paid compute or full screening run has started in this article state.
    The artifact records spend as #calc.round(r.runpod.total_spend_usd,
    digits: 3) USD and active pods as #r.runpod.active_pods_after_collection.
    #if r.stage == "local_smoke_passed" [
      The local 128/128 two-epoch smoke passed: COBA selected
      #calc.round(r.smoke.cells.coba.selected_validation_accuracy_pct,
      digits: 2)% with final E rate
      #calc.round(r.smoke.cells.coba.final_e_rate_hz, digits: 2) Hz, and
      PING selected #calc.round(r.smoke.cells.ping.selected_validation_accuracy_pct,
      digits: 2)% with final E/I rates
      #calc.round(r.smoke.cells.ping.final_e_rate_hz, digits: 2) /
      #calc.round(r.smoke.cells.ping.final_i_rate_hz, digits: 2) Hz. These
      smoke accuracies are plumbing diagnostics, not screening evidence.
    ] else [
      The local smoke is still pending.
    ]
    Pending work is the first commit/PR checkpoint, explicit RunPod spending
    gate, short candidate pair, and promotion decision.
  ] else [
    The current published attempt is `#r.attempt` at stage `#r.stage`.
    COBA selected #maybe-pct(r.cells.coba.selected_validation_accuracy_pct)
    validation accuracy; PING selected
    #maybe-pct(r.cells.ping.selected_validation_accuracy_pct). Exact spend was
    #calc.round(r.runpod.total_spend_usd, digits: 3) USD, with
    #r.runpod.active_pods_after_collection active pods after collection.
  ]

  == Activity appendix

  === Checkpoint #trace.checkpoint_id

  Timestamp: `#trace.checkpoint_time_utc`. Sanitized source hash prefix:
  `#trace.sanitized_sha256_prefix`. The log excludes hidden reasoning, tool
  payloads, credentials, private paths, environment values, addresses, and
  sensitive infrastructure details.

  ==== Decisions, actions, and pending work

  #for item in trace.ledger { [+ #item] }

  ==== Visible messages

  #for message in trace.messages { message-card(message, trace) }

  === Checkpoint #trace-two.checkpoint_id

  Timestamp: `#trace-two.checkpoint_time_utc`. Sanitized source hash prefix:
  `#trace-two.sanitized_sha256_prefix`.

  ==== Decisions, actions, and pending work

  #for item in trace-two.ledger { [+ #item] }

  ==== Visible messages added

  #for message in trace-two.messages { message-card(message, trace-two) }
]
