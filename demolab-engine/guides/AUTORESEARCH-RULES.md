# AUTORESEARCH-RULES — the semi-autonomous research contract

> The conventions for running a **research program** on demolab: a collection driven by a
> pre-registered plan, worked by an overnight agent, and gated every morning by a human.
> This guide is the source of truth the three runbooks cite — **AUTORESEARCH** (start/steer a
> program), **PLAN** (queue the next experiments), **NIGHT-SHIFT** (work the queue). It defines
> the documents, the queue schema, and the git workflow; the runbooks are the procedures.

## When to use

You and a collaborator want a research question worked semi-autonomously: steered by hand
during the day, run unattended for **at most one night** at a time, and never published without
review. This is not "an AI writes papers" — it's a lab notebook with a night shift. The whole
value is that every overnight claim is auditable the next morning, because demolab already
stamps every run with its git commit and reads every published number straight from the run.

## 1. The shape — one program = one collection

A research program is exactly one demolab **collection** (RULES §6.5). It contains, in the
order the site renders them (articles before experiments, so these sit at the top of the
collection page — question, then journey, then results):

| Document | Article name | Role |
|---|---|---|
| **The plan** | `plan` | Pre-registration + the live experiment **queue**. Owns the hypothesis, the falsifiable arc, and the amendments log. |
| **The log** | `log` | The lab notebook — decisions, failures, aborts, anomalies. **Digest at the top.** Append-only. |
| **Experiments** | `expNNN` | Standard demolab entries (RULES §7), one per queue item that ran. |

Plus one non-article file, the **night-shift contract** (`<collection>/night-shift.yaml` or a
fenced block in the plan — see §4), which carries the budgets and scope the night agent obeys.

**Exactly one `plan` and one `log` per program.** If you want a second plan, that is a second
program — give it its own collection. DOCTOR enforces this.

**Pre-registration is free here.** Because every page carries its commit (RULES §4.7), a
`plan` committed before any experiment ran is *provably* prior to the results. That is the
registered-reports guarantee, obtained from demolab's existing provenance rather than any new
machinery. Protect it with the amendment rule (§2) and the git rule (§5).

## 2. The plan article (`plan`)

A normal writing (`meta` + `body`, RULES §6.1) with `collection: <slug>`, plus a machine-readable
queue. The **body** carries the science: the question, why it matters, the baseline the stack
must reproduce before any novel claim is trusted, and a dated **Amendments** section. The
**queue** is structured data the runbooks read.

**Amendments are append-only.** A hypothesis or kill-criterion is never edited in place after an
experiment has run against it — that would make pre-registration theatre. Changes go in a dated
entry under `== Amendments`. Git would catch a silent rewrite anyway; the rule makes the agent
enforce it on itself.

**The queue** is a list of entries, one per planned experiment, in the plan's frontmatter (a
`queue:` field in `meta`, or a fenced ```yaml queue block the runbooks parse). Each entry:

| Field | Meaning |
|---|---|
| `id` | The `expNNN` id it will become. |
| `hypothesis` | The claim, stated so it can be wrong. |
| `kill` | **Required.** The result that falsifies it / aborts the run. No entry is queued without this. |
| `baseline` | What it must reproduce or beat (a published number, or an earlier `expNNN`). |
| `seeds` | Seed count for the sweep (§6). One seed is an anecdote, not a result. |
| `budget` | Per-run wall-clock ceiling (and any resource cap). Stops a diverging run eating the night. |
| `status` | `queued → running → done → killed`. **Read from run outputs, not hand-edited** (RULES §6.2) — the plan can't claim a success that didn't run. |
| `origin` | `human` (queued via PLAN) or `proposed` (drafted by NIGHT-SHIFT, awaiting human approval). A `proposed` entry is never run. |

The typeset status table in the body is **generated from this queue** (single source, same trick
as `numbers-table`), so the published plan and the agent's worklist can never disagree.

## 3. The log article (`log`)

The notebook demolab's run-stamps don't capture: *why*, and *what happened around the runs* —
decisions, failed builds, aborted runs, "killed exp014 at seed 3 per its own kill criterion,"
anomalies parked for a human. One section per night, **append-only**, newest at the top.

**The digest is the top section.** Every night ends by prepending a ruthless triage summary —
what confirmed, what died by its own criteria, what anomaly (if any) is worth human attention —
*not* a proud recitation of everything the agent did. The morning review reads this first and
often only. It is also the PR description (§5).

**Relaxed style.** The log is deliberately messy and chronological, so LINT exempts it from the
prose rules that govern articles (no lead-with-the-claim, no define-every-symbol demand). It is
still a published page and still may not hand-type run numbers — figures and metrics cite the
record like anywhere else.

## 4. The night-shift contract

Per-program **policy**, separate from the runbook's **procedure**. One NIGHT-SHIFT procedure
serves every program; the contract is what makes two programs run with different allowances.
Keep it small — `night-shift.yaml` beside the collection, or a fenced block in the plan:

```yaml
budgets:
  wall_clock_per_run: 45m      # hard ceiling per experiment
  wall_clock_per_night: 8h     # total shift budget
  seeds_default: 5
scope:
  collection: neuron-models    # the only collection this shift may touch
  may_propose: true            # append `proposed` queue entries (never run them)
stop_when:
  - queue_empty
  - night_budget_exhausted
  - build_red_twice            # bail rather than thrash
```

## 5. Git & PR workflow

- **Main is the published record.** Turn on branch protection; CI deploys the site from `main`
  only, so nothing publishes without a merge.
- **One branch per night per program:** `night/YYYY-MM-DD-<collection>`. Programs touch disjoint
  collection directories, so parallel nights merge cleanly.
- **One commit per experiment attempt**, including killed ones — a negative result is a result,
  and per-experiment commits let you cherry-pick the two good runs out of a mixed night.
- **Merge commits only — never squash or rebase.** Run stamps point at the night-branch commits;
  rewriting history breaks the provenance shown on published pages.
- **The `plan` body changes only on `main`, in a daytime AUTORESEARCH/PLAN session.** A night
  branch may update queue **statuses** and append to the **log**, but never edits the hypothesis
  or kill criteria. Hypothesis changes are human-reviewed by construction.
- **Accountability lives in the log, not git metadata.** Commits carry no agent authorship (house
  rule, RULES §2), so "this ran under the night shift within this budget" belongs in the log
  article — on the page a supervisor actually reads.
- **The PR is the morning gate.** NIGHT-SHIFT ends by opening a PR whose description *is* the
  digest. Triage = review the PR: merge, request changes ("rerun exp014, more seeds"), or close.
- **Reaping:** enable GitHub's *auto-delete head branches on merge*
  (`gh api -X PATCH repos/{owner}/{repo} -f delete_branch_on_merge=true`); the preview teardown
  (the PR-preview workflow) fires on the same close/merge. Orphan branches (a crashed night, a
  closed-unmerged PR) are surfaced by NIGHT-SHIFT's resume check and killed by a human — no
  auto-reaper that could delete the only copy of a half-finished experiment. **Reap branches,
  never PRs:** merged/closed PRs are the audit trail.

## 6. SNN / compute-heavy rigor

- **Baseline gate.** Before any novel claim, the stack must reproduce a known baseline (e.g. a
  LIF network's published accuracy on N-MNIST / SHD). No trust without reproduction — it's the
  `baseline` field on the first queue entries.
- **Multi-seed by default.** A queue entry's `seeds` drives a sweep with error bars, not a
  single-seed anecdote. Bake this into the tool ↔ experiment contract before compute-heavy work.
- **Analysis plan precedes data.** The `kill` and `baseline` are committed before the run — no
  HARKing a story out of the numbers after the fact. PLAN enforces it at queue time.

## 7. How it plugs into the existing rules

- Statuses and metrics are **read from `numbers.json`** (RULES §6.2), so the plan and log can't
  drift from what ran.
- Pre-registration = **provenance** (RULES §4.7): the plan's commit predates the results' commits.
- Long runs use **staged runners** (RULES §7.5) so a night can re-enter compute without repeating it.
- **NEXT** reads the `log` (not just published results), so it proposes from the decision arc; its
  suggestions are the raw material a human turns into `queued` entries via **PLAN**.
- **DOCTOR** checks program structure (one `plan`, one `log`, every `queued` entry has a `kill`).
- **LINT** applies relaxed rules to the `log`.
