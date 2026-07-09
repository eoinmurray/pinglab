# Runbook: Work the queue overnight

> The autonomous shift: on a night branch, pop each `queued` experiment, implement and run it
> within its budget, check the result against its pre-registered kill criterion and baseline,
> draft the writeup, log what happened, and commit — one commit per experiment. End the night by
> writing the digest and opening a PR for the morning gate. This is the only runbook that runs
> unattended, and it runs for **at most one night**. It consumes what **PLAN** queued and obeys
> the program's night-shift contract; conventions in
> [`../guides/AUTORESEARCH-RULES.md`](../guides/AUTORESEARCH-RULES.md).

## When to use

Unattended execution of an already-planned queue — typically kicked off at end of day (by hand or
a scheduler). It never invents science: it runs what was pre-registered, and may *propose* more
but never run its own proposals. If the queue is empty, there is nothing to do — that's PLAN's job.

## What it does

0. **Resume check.** Is there an unmerged `night/*` branch for this program from a prior, crashed
   shift? If so, **stop and surface it** — resume it or abandon it is a human's call, not the
   night's. Never start a fresh branch over an orphan.

1. **Open the branch.** Create `night/YYYY-MM-DD-<collection>` off `main`. Read the night-shift
   contract: wall-clock-per-run, wall-clock-per-night, default seeds, scope, stop conditions.

2. **Work the queue.** For each `status: queued`, `origin: human` entry, in order, until a stop
   condition (queue empty / night budget spent / build red twice):
   - **Implement** — add/adjust the tool subcommand and write `experiments/expNNN.py` (RULES §7),
     modelled on an existing runner. Run the seed sweep from the entry.
   - **Enforce the budget** — abort the run at its wall-clock ceiling; a diverging training run
     does not get to eat the night. Log the abort, move on.
   - **Check against the pre-registration** — did it clear its `baseline`? Did its `kill`
     criterion fire? Set the entry's status from the run output: `done` or `killed`. Never edit
     the hypothesis or kill criterion — those are read-only on a night branch (AUTORESEARCH-RULES §5).
   - **Write it up** — `writings/expNNN.typ`, numbers pulled from the record, not typed.
   - **Log it** — append a line to the `log`: what ran, seeds, outcome, any abort, and any anomaly.
   - **Commit** — one commit per experiment attempt, killed ones included. Merge-workflow only;
     no rebase/squash.

3. **Anomaly policy: log and move on.** If a run looks odd, do **not** chase it at 3 a.m. — that's
   how a night burns on a seed artifact. Record it in the log as an anomaly for the morning and
   continue the queue. Chasing is a human decision made in PLAN.

4. **Propose, don't run.** If the night's results suggest a worthwhile next experiment, append it
   to the queue as `origin: proposed`, `status: queued` — a **draft** for the human to approve in
   PLAN. Never run a proposal, and never run anything that wasn't pre-registered. (Only if the
   contract's `scope.may_propose` is true.)

5. **Write the digest.** Prepend a ruthless triage summary to the `log`: how many confirmed, how
   many killed by their own criteria, which anomaly (if any) deserves attention, what's proposed.
   Not a recitation of everything done — the thing a human reads in thirty seconds over coffee.

6. **Open the PR.** Push the branch, open a PR whose **description is the digest**. Stop. The
   night is over; the morning gate (PLAN / the human review) decides what merges. Publish nothing
   — CI deploys from `main`, and only a human merge reaches `main`.

---

## Agent contract
- **Triggers** — `NIGHT-SHIFT`, "work the queue overnight", "run tonight's experiments",
  "start the night shift".
- **Gates** — **hard rules, non-negotiable:** resume-check before starting (§0); obey the
  contract's budgets and stop conditions; **never invent-and-run** an experiment that wasn't
  pre-registered; never edit the plan's hypothesis/kill criteria; merge-workflow commits only;
  push to a `night/*` branch and open a PR — **never touch `main`**.
- **Report & apply** — the deliverable is the branch + the PR + the digest. Land the finished
  work, log the aborts and anomalies honestly (a killed experiment is a result, report it as
  one), and hand the whole night to the morning review in a single PR.
