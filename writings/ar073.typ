#let meta = (
  title: "Activity trace — accelerating matched SHD learning",
  date: "2026-07-19",
  description: "A privacy-sanitized, message-level activity record for the exp070 exploratory ladder.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 9,
)

#let trace = json("/artifacts/data/ar073/messages.json")
#let trace-two = json("/artifacts/data/ar073/messages_cp002.json")
#let trace-three = json("/artifacts/data/ar073/messages_cp003.json")
#let trace-four = json("/artifacts/data/ar073/messages_cp004.json")

#let verbatim-prose(value) = {
  for (index, line) in value.split("\n").enumerate() {
    if index > 0 { linebreak() }
    text(size: 9.3pt, line)
  }
}

#let message-card(message, checkpoint) = {
  let speaker = if message.role == "user" { "User" } else { "Assistant" }
  block(width: 100%, breakable: true, fill: luma(96%), stroke: 0.6pt + luma(82%),
    radius: 4pt, inset: 10pt)[
    #grid(columns: (1fr, auto), [#strong(speaker)],
      [#text(size: 8pt, fill: luma(42%), message.timestamp)])
    #v(3pt)
    #text(size: 7.5pt, fill: luma(48%))[
      Role: #message.role · Checkpoint: #checkpoint.checkpoint_id · Session: #checkpoint.session_id
    ]
    #v(7pt)
    #verbatim-prose(message.content)
    #if "attachment" in message {
      v(8pt)
      text(size: 8pt, weight: "bold", fill: luma(42%))[Attached request]
      v(4pt)
      verbatim-prose(message.attachment)
    }
  ]
  v(8pt)
}

#let body = [
  == Checkpoint CP-001

  *The next exploratory night is registered before any exp070 training result.*
  This checkpoint has immutable private source SHA-256 prefix
  `5189910c09c9`. Its checkpoint time is `2026-07-19 06:49:09.368 UTC`, and
  it has no predecessor within this night. The raw snapshot is held read-only
  outside the repository. Hidden reasoning, tool payloads, injected environment
  material, credentials, private paths, addresses, and infrastructure
  identifiers are excluded. The visible attachment path is explicitly redacted;
  the request itself is reproduced verbatim.

  === Decision and action ledger

  + Pull request 52 is open and draft on the requested existing branch. Main is
    unchanged.
  + Exp069's complete epoch histories, firing diagnostics, gradients, rasters,
    split provenance, billing, and official-test prohibition were inspected.
  + A development-data-only audit rejected observation-window truncation as the
    likely bottleneck. The first registered candidate changes only the matched
    temporal resolution from 1 ms to 2 ms.
  + The ordered ladder, first-passing selection rule, promotion threshold,
    forty-epoch success criterion, kill criteria, one-seed policy, locked model
    comparison, and fresh paid-compute gate are fixed in ar072.
  + No exp070 runner, training result, cloud job, official-test access, or spend
    exists at this checkpoint. Pending work is the candidate-1 implementation,
    focused checks, local smoke, cost estimate, and paid-compute authority gate.

  == Privacy-sanitized visible transcript

  #for message in trace.messages { message-card(message, trace) }

  == Checkpoint CP-002

  *Candidate 1 is implemented and its local smoke gate passes.* This checkpoint
  follows mandate anchor `cac3d07`. Its immutable private source has SHA-256
  prefix `8b7c5612397e`, and its checkpoint time is
  `2026-07-19 07:04:04.503 UTC`.

  === Decision and action ledger

  + The exp070 runner changes only temporal resolution and short-run duration
    while reusing exp069's split, checkpoint selection, diagnostics, rasters,
    and cloud orchestration. `tools/snn` is unchanged.
  + A pre-smoke audit found that inherited backup/restore plumbing could copy an
    existing official-test cache even though it never evaluated it. Exp070 now
    redirects the engine to a development-only alias, closing that protocol
    ambiguity before training.
  + Focused Ruff, `ty`, compilation, and two-pod dry-run checks pass. The dry
    run created no pod.
  + Both 2 ms smoke cells are finite and active, use the exact exp069 split
    hashes, and report no skipped or non-finite updates. Spend remains 0 USD.
  + Pending work is a commit/push checkpoint and fresh authority for the matched
    five-epoch cloud pair. No paid compute may start before that approval.

  === Visible messages added in CP-002

  #for message in trace-two.messages { message-card(message, trace-two) }

  == Checkpoint CP-003

  *The cloud boundary now fails closed on scientific and billing integrity.*
  This checkpoint follows smoke commit `135b82e`. Its immutable private source
  has SHA-256 prefix `20a7b2d23008`, and its checkpoint time is
  `2026-07-19 07:11:32.044 UTC`.

  === Decision and action ledger

  + The expected first-candidate plus promoted-final path is approximately 2
    USD. The requested hard ceiling is 10 USD across the entire registered
    ladder, with one pod per cell and two concurrent pods maximum. Authority
    has not yet been granted, so no dispatch occurred.
  + GitHub Actions reports the same ten repository-wide `ty` diagnostics in
    unchanged older experiments and tests. Exp070 produces no CI diagnostic,
    and those unrelated files remain out of scope.
  + Two focused tests pin the first-passing promotion gate: both cells must
    clear the accuracy and loss thresholds, and non-finite or saturated
    activity kills the candidate.
  + Artifact publication now refuses to proceed without finite authoritative
    provider spend and confirmation that zero pods remain. Ruff, focused `ty`,
    compilation, both tests, and the missing-ledger failure check pass.
  + Pending work remains fresh paid-compute authorization, the matched
    five-epoch pair, exact billing reconciliation, and the registered adaptive
    decision. Spend remains 0 USD.

  === Visible messages added in CP-003

  #for message in trace-three.messages { message-card(message, trace-three) }

  == Checkpoint CP-004

  *Fresh paid-compute authority is explicit and bounded.* This checkpoint
  follows hardening commit `2bc22d3`. Its immutable private source has SHA-256
  prefix `c20363a95684`, and its checkpoint time is
  `2026-07-19 07:21:26.366 UTC`.

  === Decision and action ledger

  + The scientist authorized a 10 USD hard total ceiling across exp070, with at
    most two concurrent pods and exactly one cell per pod.
  + Short candidates carry one-hour pod backstops. A promoted forty-epoch pair
    may use two-hour backstops. Every pod must be reaped immediately.
  + Before this checkpoint the fleet was empty, spend was 0 USD, the branch was
    clean at `2bc22d3`, and the pull request remained draft and unmerged.
  + The next action is limited to the registered five-epoch 2 ms pair. No later
    candidate or final pair is dispatched before applying the registered
    result-dependent decision.

  === Visible messages added in CP-004

  #for message in trace-four.messages { message-card(message, trace-four) }
]
