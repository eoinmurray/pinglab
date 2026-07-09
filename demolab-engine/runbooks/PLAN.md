# Runbook: Plan the next experiments (and triage the last night)

> The daily driver: read last night's digest and open PR, decide what to promote or kill, then
> interview the scientist into the next batch of **pre-registered** queue entries — each with a
> kill criterion, a baseline, and a seed and time budget — written into the program's `plan`.
> Smaller and more frequent than **AUTORESEARCH** (which defines the program); it feeds
> **NIGHT-SHIFT** (which works the queue). Conventions:
> [`../guides/AUTORESEARCH-RULES.md`](../guides/AUTORESEARCH-RULES.md).

## When to use

Most mornings of an active program, and any time you want to refill the queue. Morning triage is
the first half of this runbook, not a separate ritual: you open the night's PR, decide, then plan
forward from what you learned. Runs on `main`, no compute.

## What it does

0. **Read the night.** Find the open `night/*` PR for this program and read its description (the
   digest) and the top section of the `log`. If there's no PR, skip to step 3.

1. **Triage the results.** For each experiment the night produced, check it against its
   pre-registered entry:
   - **Confirmed** — met its baseline, survived its kill criterion across the seed sweep → keep.
   - **Killed by its own criterion** — the plan said this would falsify it; record it as a
     negative result, don't rescue it.
   - **Anomaly parked for you** — the night logged something odd and moved on (by design). Decide:
     ignore, or queue a follow-up to chase it.
   Then act on the PR: **merge** (accept the night), **request changes** ("rerun exp014 with more
   seeds"), or **close** (reject). The PR *is* the gate — nothing here republishes without it.

2. **Review NIGHT-SHIFT's proposals.** The night may have appended `origin: proposed` entries to
   the queue — experiments it thinks are worth running. They were **never run** (that's the rule).
   Promote the good ones to `origin: human` `queued`, or delete them. This is where you stop the
   agent from steering itself into cheap parameter-sweep local minima.

3. **Plan the next batch (interview the scientist).** For each new experiment, pin down:
   - **Hypothesis** — stated so it can be wrong.
   - **Kill criterion** — the result that falsifies it or aborts the run. **Required.**
   - **Baseline** — the number/earlier `expNNN` it must reproduce or beat.
   - **Seeds** — the sweep size (default from the contract); one seed is an anecdote.
   - **Budget** — the per-run wall-clock ceiling.
   Keep each experiment small — one variable moved (the demolab loop). Lean on **NEXT** for
   candidate directions from the decision arc; this runbook turns a chosen direction into a
   committed, falsifiable entry.

4. **Refuse the unfalsifiable.** Do not write a queue entry that has no kill criterion. This one
   rule does more for rigor than any amount of after-the-fact red-teaming — enforce it even when
   the scientist is impatient.

5. **Write the queue and commit.** Append the entries to the `plan`'s queue (frontmatter/fenced
   block, AUTORESEARCH-RULES §2) as `status: queued`, `origin: human`. Build to confirm the
   generated status table renders, commit on `main` ("Queue exp0NN–exp0NN"). The queue is now the
   night's worklist.

---

## Agent contract
- **Triggers** — `PLAN`, "plan the next experiments", "triage last night", "refill the queue",
  "what do we run tonight".
- **Gates** — runs on `main`, daytime, no compute. **Every queued entry must carry a kill
  criterion** (step 4). Never run experiments here; never promote a `proposed` entry without the
  scientist's say-so.
- **Report & apply** — triage the PR interactively, review proposals, interview into new entries,
  refuse the unfalsifiable, commit the queue on `main`. Then it's the night's to work.
