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
#let trace-five = json("/artifacts/data/ar073/messages_cp005.json")
#let trace-six = json("/artifacts/data/ar073/messages_cp006.json")
#let trace-seven = json("/artifacts/data/ar073/messages_cp007.json")
#let trace-eight = json("/artifacts/data/ar073/messages_cp008.json")
#let trace-nine = json("/artifacts/data/ar073/messages_cp009.json")
#let trace-ten = json("/artifacts/data/ar073/messages_cp010.json")
#let trace-eleven = json("/artifacts/data/ar073/messages_cp011.json")

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

  == Checkpoint CP-005

  *Only the registered candidate-1 pair is dispatched.* This checkpoint follows
  authority commit `cd82023`. Its immutable private source has SHA-256 prefix
  `3b4b22492a5c`, and its checkpoint time is
  `2026-07-19 07:23:43.025 UTC`.

  === Decision and action ledger

  + The dry run found zero existing pods and exactly two jobs: one COBA and one
    PING cell at 2 ms for five epochs.
  + Two 5090 pods were created at 0.99 USD/hour each, pinned to `cd82023`, with
    one-hour self-removal backstops. The post-dispatch fleet count was exactly
    two, so worst-case exposure for this attempt is 1.98 USD.
  + Pod identifiers and other infrastructure details remain only in the private
    trace. The public record retains names, counts, rates, commit, and bounds.
  + Pending work is self-termination, collection, exact provider billing,
    artifact publication, and the preregistered promotion decision.

  === Visible messages added in CP-005

  #for message in trace-five.messages { message-card(message, trace-five) }

  == Checkpoint CP-006

  *Candidate 1 is killed after the registered five-epoch comparison.* This
  checkpoint follows dispatch commit `1bba764`. Its immutable private source
  has SHA-256 prefix `cf6b7ebf703b`, and its checkpoint time is
  `2026-07-19 07:34:41.277 UTC`.

  === Decision and action ledger

  + Both pods self-terminated about eight minutes after dispatch; the active
    fleet is zero and the complete cell outputs were collected.
  + COBA reached 23.53% validation accuracy at epoch five, 4.41 percentage
    points below exp069, while cross-entropy worsened by 0.206 to 2.678.
  + PING reached 27.70%, 4.90 points below exp069, while cross-entropy worsened
    by 0.148 to 2.592.
  + Both cells were finite and active with no skipped or non-finite updates, so
    this is a scientific kill rather than a plumbing failure. The 2 ms change
    fails every registered promotion requirement.
  + Exact provider billing has not yet appeared in the billing history. The
    publication gate remains closed until the authoritative spend is recorded;
    no estimate is substituted. Candidate 2 has not been implemented or run.

  === Visible messages added in CP-006

  #for message in trace-six.messages { message-card(message, trace-six) }

  == Checkpoint CP-007

  *Candidate 2 is implemented and passes its local smoke gate.* This checkpoint
  follows the killed-attempt commit `9e1385a`. Its immutable private source has
  SHA-256 prefix `b12370990974`, and its checkpoint time is
  `2026-07-19 07:47:15.296 UTC`.

  === Decision and action ledger

  + Candidate 2 restores 1 ms and changes only the matched shared input-weight
    mean from 0.9 to 1.2. Candidate 1's outputs remain separately addressable.
  + Focused Ruff, type, compile, and promotion-gate checks pass. `tools/snn` is
    unchanged.
  + The required 128-training/128-validation two-epoch smoke passes in both
    cells. COBA ends at 26.79 Hz E; PING ends at 3.24 Hz E and 17.00 Hz I.
    Losses and gradients are finite and no update is skipped.
  + The exact registered split hashes are retained and the development-only
    alias contains no official-test file. The two-job cloud dry run creates no
    pod and resolves to exactly one job per cell.
  + Candidate-1 provider billing is still delayed. The fleet is empty and
    candidate 2 has not been dispatched at this checkpoint.

  === Visible messages added in CP-007

  #for message in trace-seven.messages { message-card(message, trace-seven) }

  == Checkpoint CP-008

  *Only the registered candidate-2 pair is dispatched.* This checkpoint follows
  smoke commit `04d5a28`. Its immutable private source has SHA-256 prefix
  `a0717f4a44f2`, and its checkpoint time is
  `2026-07-19 07:50:04.550 UTC`.

  === Decision and action ledger

  + The clean dry run found zero existing pods and exactly two candidate-2
    jobs: one COBA and one PING cell at 1 ms with shared input mean 1.2.
  + Two 5090 pods were created at 0.99 USD/hour each, pinned to `04d5a28`, with
    one-hour self-removal backstops. The verified post-dispatch fleet count is
    exactly two.
  + Candidate 1's exact billing is still delayed, but its hard bound is 1.98
    USD. The two short attempts therefore have a cumulative hard bound of 3.96
    USD, below the authorized 10 USD ceiling.
  + Infrastructure identifiers remain only in the immutable private trace.
    Pending work is self-termination, collection, provider reconciliation, and
    the locked promotion decision.

  === Visible messages added in CP-008

  #for message in trace-eight.messages { message-card(message, trace-eight) }

  == Checkpoint CP-009

  *Candidate 2 is killed after the locked epoch-five comparison.* This
  checkpoint follows dispatch commit `90575b9`. Its immutable private source
  has SHA-256 prefix `c55f4048b3c7`, and its checkpoint time is
  `2026-07-19 08:06:39.010 UTC`.

  === Decision and action ledger

  + PING self-terminated first and COBA followed by 08:05:03 UTC. The fleet is
    zero and both complete records were collected.
  + COBA reaches 27.94% at epoch five, exactly matching exp069, while
    cross-entropy is 0.0168 worse. PING reaches 33.70%, only 1.10 percentage
    points above exp069, while cross-entropy is 0.00153 worse.
  + Both cells are finite and active with no skipped or non-finite updates.
    Candidate 2 therefore fails the efficacy gate cleanly; it is not promoted.
  + Exact provider rows for both completed attempts remain delayed. No estimate
    is published as exact. Candidate 3 is the sole remaining registered short
    attempt and has not yet been implemented.

  === Visible messages added in CP-009

  #for message in trace-nine.messages { message-card(message, trace-nine) }

  == Checkpoint CP-010

  *The final registered short candidate passes its local smoke gate.* This
  checkpoint follows the candidate-2 kill commit `3ac37b4`. Its immutable
  private source has SHA-256 prefix `1035934cfc77`, and its checkpoint time is
  `2026-07-19 08:14:25.022 UTC`.

  === Decision and action ledger

  + Candidate 3 restores the baseline 1 ms temporal resolution and shared input
    mean 0.9, changing only the matched Adam learning rate to 0.001.
  + Focused Ruff, type, compile, and promotion-gate tests pass. `tools/snn` is
    unchanged and no candidates are combined.
  + Both 128/128 two-epoch smoke cells pass with finite activity and no skipped
    updates. COBA ends at 26.13 Hz E; PING at 3.36 Hz E and 19.00 Hz I.
  + Exact split hashes are retained, the staged alias has no official-test
    file, and the cloud dry run creates no pod while resolving to one job per
    cell. Paid dispatch remains pending this checkpoint.

  === Visible messages added in CP-010

  #for message in trace-ten.messages { message-card(message, trace-ten) }

  == Checkpoint CP-011

  *Only the final registered short pair is dispatched.* This checkpoint follows
  smoke commit `e9d0332`. Its immutable private source has SHA-256 prefix
  `c49ca5689e35`, and its checkpoint time is
  `2026-07-19 08:16:37.476 UTC`.

  === Decision and action ledger

  + The clean dry run found zero pods and exactly two candidate-3 jobs. The live
    dispatch created one COBA and one PING 5090 pod at 0.99 USD/hour each.
  + Both jobs are pinned to `e9d0332` and carry one-hour self-removal backstops.
    The verified fleet count is exactly two.
  + Using unreconciled one-hour maxima for all three short pairs gives a 5.94
    USD cumulative hard bound, still below the authorized 10 USD ceiling.
  + Infrastructure identifiers remain private. Pending work is termination,
    collection, billing reconciliation, and the locked promotion decision.

  === Visible messages added in CP-011

  #for message in trace-eleven.messages { message-card(message, trace-eleven) }
]
