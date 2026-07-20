#let meta = (
  title: "Adaptive conductance controls for matched SHD networks",
  date: "2026-07-19",
  description: "A one-seed validation-only exploratory ladder tests whether trainable conductance leak and adaptive E-cell thresholds move matched COBA/PING SHD accuracy beyond the exp071 plateau.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 11,
)

#let r = json("/artifacts/data/exp072/numbers.json")
#let trace = json("/artifacts/data/exp072/activity/messages_cp001.json")
#let trace-two = json("/artifacts/data/exp072/activity/messages_cp002.json")
#let trace-three = json("/artifacts/data/exp072/activity/messages_cp003.json")
#let trace-four = json("/artifacts/data/exp072/activity/messages_cp004.json")
#let trace-five = json("/artifacts/data/exp072/activity/messages_cp005.json")
#let trace-six = json("/artifacts/data/exp072/activity/messages_cp006.json")

#let maybe-pct(x) = {
  if type(x) == float or type(x) == int {
    [#calc.round(x, digits: 2)%]
  } else {
    [pending]
  }
}

#let attempt-figure(suffix) = "/artifacts/data/exp072/" + r.stage + "_" + r.attempt + "_" + suffix

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

  This one-seed exploratory experiment tests whether two newly added
  conductance-neuron controls move matched Dale-constrained COBA and PING
  networks beyond the exp071 roughly 70% validation plateau on SHD. The control
  is the exp071 cumulative-potential readout recipe:
  `--readout cumulative-potential --signed-readout --readout-bias`. The ladder
  then tests bounded trainable leak/membrane time constants, bounded adaptive
  E-cell thresholds, and their combination. All selection decisions use the
  held-out validation split from the official SHD training set; the official
  test set remains sealed.

  == Goal prompt

  #raw(read("/artifacts/data/exp072/goal.txt"), block: true)

  == Methods

  === Locked comparison

  Both cells use seed #r.seed, the established SHD development split
  (#r.split.development_train_count training utterances and
  #r.split.validation_count validation utterances), batch size
  #r.config.batch_size, Adam learning rate #r.config.learning_rate,
  #r.config.dt_ms ms simulation steps, #r.config.t_ms ms utterance windows,
  matched input preprocessing, and the same cumulative-potential readout.
  COBA uses the no-inhibitory-loop setting and no voltage-gradient dampening;
  PING uses the registered inhibitory loop and dampening 1000. Those
  cell-specific settings are the only intended COBA/PING differences within a
  candidate.

  === Candidate ladder

  #table(
    columns: (auto, 1.4fr, 1.1fr, auto),
    table.header([*Order*], [*Candidate*], [*Extra flags*], [*Short epochs*]),
    [0], [Control: exp071 architecture/readout], [none], [#r.screening_epochs],
    [1], [Trainable conductance leak / membrane time constants], [`--train-leak`], [#r.screening_epochs],
    [2], [Adaptive E-cell thresholds], [`--adaptive-threshold`], [#r.screening_epochs],
    [3], [Combined leak plus adaptation], [`--train-leak --adaptive-threshold`], [#r.screening_epochs],
  )

  The default bounds are E membrane tau [5, 50] ms, I membrane tau [2, 20] ms,
  adaptive-threshold tau [50, 500] ms, initial adaptive strength 1.0 mV, and
  maximum adaptive strength 20.0 mV. Candidate selection uses held-out
  validation accuracy and cross-entropy only. Training must remain finite,
  active, non-saturated, and free of skipped or non-finite updates.

  == Results

  #if r.result_status == "preregistered" [
    No paid compute has started in this article state. The artifact records
    spend as #calc.round(r.runpod.total_spend_usd, digits: 3) USD and active
    pods as #r.runpod.active_pods_after_collection.
    #if r.stage == "local_smoke_ladder_passed" [
      The local 128/128 two-epoch smoke passed for all four candidates. These
      smoke numbers validate plumbing only; they are not screening evidence.
      The selected validation accuracies were:

      #table(
        columns: (1.25fr, auto, auto),
        table.header([*Candidate*], [*COBA smoke*], [*PING smoke*]),
        [control],
        [#maybe-pct(r.smoke_ladder.control.cells.coba.selected_validation_accuracy_pct)],
        [#maybe-pct(r.smoke_ladder.control.cells.ping.selected_validation_accuracy_pct)],
        [train_leak],
        [#maybe-pct(r.smoke_ladder.train_leak.cells.coba.selected_validation_accuracy_pct)],
        [#maybe-pct(r.smoke_ladder.train_leak.cells.ping.selected_validation_accuracy_pct)],
        [adaptive_threshold],
        [#maybe-pct(r.smoke_ladder.adaptive_threshold.cells.coba.selected_validation_accuracy_pct)],
        [#maybe-pct(r.smoke_ladder.adaptive_threshold.cells.ping.selected_validation_accuracy_pct)],
        [combined],
        [#maybe-pct(r.smoke_ladder.combined.cells.coba.selected_validation_accuracy_pct)],
        [#maybe-pct(r.smoke_ladder.combined.cells.ping.selected_validation_accuracy_pct)],
      )
    ] else if r.stage == "local_smoke_passed" [
      The local 128/128 two-epoch smoke passed for the currently selected
      candidate. COBA selected
      #calc.round(r.smoke.cells.coba.selected_validation_accuracy_pct, digits: 2)%
      and PING selected
      #calc.round(r.smoke.cells.ping.selected_validation_accuracy_pct, digits: 2)%.
      These smoke numbers validate plumbing only; they are not screening
      evidence.
    ] else [
      The local smoke is still pending.
    ]
    Pending work is short validation screening of the four candidates, promotion
    of one candidate to forty epochs, final artifacts, and human review.
  ] else [
    The eight-epoch ladder completed for all four candidates. Control had the
    best paired average, but trainable leak was the only modification that
    improved the target PING cell while keeping COBA close, so `train_leak` was
    promoted to forty epochs.

    #table(
      columns: (1.25fr, auto, auto, auto, auto),
      table.header([*Candidate*], [*COBA acc.*], [*COBA CE*], [*PING acc.*], [*PING CE*]),
      [control],
      [#maybe-pct(r.ladder.short.control.cells.coba.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.control.cells.coba.selected_validation_cross_entropy, digits: 4)],
      [#maybe-pct(r.ladder.short.control.cells.ping.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.control.cells.ping.selected_validation_cross_entropy, digits: 4)],
      [train_leak],
      [#maybe-pct(r.ladder.short.train_leak.cells.coba.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.train_leak.cells.coba.selected_validation_cross_entropy, digits: 4)],
      [#maybe-pct(r.ladder.short.train_leak.cells.ping.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.train_leak.cells.ping.selected_validation_cross_entropy, digits: 4)],
      [adaptive_threshold],
      [#maybe-pct(r.ladder.short.adaptive_threshold.cells.coba.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.adaptive_threshold.cells.coba.selected_validation_cross_entropy, digits: 4)],
      [#maybe-pct(r.ladder.short.adaptive_threshold.cells.ping.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.adaptive_threshold.cells.ping.selected_validation_cross_entropy, digits: 4)],
      [combined],
      [#maybe-pct(r.ladder.short.combined.cells.coba.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.combined.cells.coba.selected_validation_cross_entropy, digits: 4)],
      [#maybe-pct(r.ladder.short.combined.cells.ping.selected_validation_accuracy_pct)],
      [#calc.round(r.ladder.short.combined.cells.ping.selected_validation_cross_entropy, digits: 4)],
    )

    The promoted forty-epoch `train_leak` run selected
    #maybe-pct(r.ladder.final40.train_leak.cells.coba.selected_validation_accuracy_pct)
    for COBA and
    #maybe-pct(r.ladder.final40.train_leak.cells.ping.selected_validation_accuracy_pct)
    for PING. As a later sensitivity check, the initially weak `combined`
    candidate was also run for eighty epochs.

    #table(
      columns: (1.2fr, auto, auto, auto, auto),
      table.header([*Long run*], [*COBA acc.*], [*COBA epoch*], [*PING acc.*], [*PING epoch*]),
      [train_leak, 40 epochs],
      [#maybe-pct(r.ladder.final40.train_leak.cells.coba.selected_validation_accuracy_pct)],
      [#r.ladder.final40.train_leak.cells.coba.selected_epoch],
      [#maybe-pct(r.ladder.final40.train_leak.cells.ping.selected_validation_accuracy_pct)],
      [#r.ladder.final40.train_leak.cells.ping.selected_epoch],
      [combined, 80 epochs],
      [#maybe-pct(r.ladder.final80.combined.cells.coba.selected_validation_accuracy_pct)],
      [#r.ladder.final80.combined.cells.coba.selected_epoch],
      [#maybe-pct(r.ladder.final80.combined.cells.ping.selected_validation_accuracy_pct)],
      [#r.ladder.final80.combined.cells.ping.selected_epoch],
    )

    The eighty-epoch combined follow-up nudged PING to
    #maybe-pct(r.ladder.final80.combined.cells.ping.selected_validation_accuracy_pct),
    but only after twice as many epochs and with stuck-dynamics warnings; COBA
    remained lower at
    #maybe-pct(r.ladder.final80.combined.cells.coba.selected_validation_accuracy_pct)
    and showed saturated/stuck dynamics. This supports the interpretation that
    adaptive thresholds did not provide a useful path to the canonical
    high-accuracy SHD regime in this setup.

    Cumulative RunPod spend recorded by the experiment ledgers is
    #calc.round(r.spend.total_spend_usd, digits: 3) USD. The provider billing
    rows had not posted at publication time, so the spend is marked
    `#r.spend.status`; active pods after the final collection were
    #r.runpod.active_pods_after_collection.

    #v(8pt)
    #image(attempt-figure("validation_curves.svg"), width: 100%)
    #v(4pt)
    #image(attempt-figure("activity_curves.svg"), width: 100%)
    #v(4pt)
    #image(attempt-figure("matched_rasters.png"), width: 100%)

    The current figure set below shows the latest published attempt,
    `#r.attempt` at `#r.stage`; previous long-run and short-ladder numeric
    summaries remain archived under `raw/collected_ladder_summary.json`.

    The learned parameter diagnostics for the latest attempt show COBA E-cell membrane time constants with
    mean #calc.round(r.parameters.coba.layers.at("1").tau_m_e_ms.mean, digits: 2)
    ms and PING E/I means of
    #calc.round(r.parameters.ping.layers.at("1").tau_m_e_ms.mean, digits: 2) ms /
    #calc.round(r.parameters.ping.layers.at("1").tau_m_i_ms.mean, digits: 2) ms.
    The run also writes per-cell parameter diagnostics under
    `raw/#r.stage/#r.attempt/<cell>/parameter_diagnostics.json` and archives all
    collected short-ladder summaries under `raw/collected_ladder_summary.json`.
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

  === Checkpoint #trace-three.checkpoint_id

  Timestamp: `#trace-three.checkpoint_time_utc`. Sanitized source hash prefix:
  `#trace-three.sanitized_sha256_prefix`.

  ==== Decisions, actions, and pending work

  #for item in trace-three.ledger { [+ #item] }

  ==== Visible messages added

  #for message in trace-three.messages { message-card(message, trace-three) }

  === Checkpoint #trace-four.checkpoint_id

  Timestamp: `#trace-four.checkpoint_time_utc`. Sanitized source hash prefix:
  `#trace-four.sanitized_sha256_prefix`.

  ==== Decisions, actions, and pending work

  #for item in trace-four.ledger { [+ #item] }

  ==== Visible messages added

  #for message in trace-four.messages { message-card(message, trace-four) }

  === Checkpoint #trace-five.checkpoint_id

  Timestamp: `#trace-five.checkpoint_time_utc`. Sanitized source hash prefix:
  `#trace-five.sanitized_sha256_prefix`.

  #if trace-five.sanitization.status == "partial_fail_closed" [
    Note: this checkpoint records the scientific ledger for the long RunPod
    monitoring turn, but does not claim a complete verbatim transcript because
    no raw transcript export was available in the workspace.
  ]

  ==== Decisions, actions, and pending work

  #for item in trace-five.ledger { [+ #item] }

  ==== Visible milestone messages added

  #for message in trace-five.messages { message-card(message, trace-five) }

  === Checkpoint #trace-six.checkpoint_id

  Timestamp: `#trace-six.checkpoint_time_utc`. Sanitized source hash prefix:
  `#trace-six.sanitized_sha256_prefix`.

  ==== Decisions, actions, and pending work

  #for item in trace-six.ledger { [+ #item] }

  ==== Visible milestone messages added

  #for message in trace-six.messages { message-card(message, trace-six) }
]
