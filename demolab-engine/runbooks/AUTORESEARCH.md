# Runbook: Start or steer a research program

> Stand up a **research program** as a demolab collection — define the falsifiable arc with the
> scientist, scaffold the `plan` + `log` articles and the night-shift contract, and over the
> program's life handle amendments and the closing retrospective. This is the **program-level**
> session, run a handful of times per program; day-to-day experiment planning is **PLAN**, and
> the overnight execution is **NIGHT-SHIFT**. Conventions live in
> [`../guides/AUTORESEARCH-RULES.md`](../guides/AUTORESEARCH-RULES.md); this runbook is the procedure.

## When to use

At the birth of a research program, and at its big hinge points — a major pivot, or the close.
It is interactive and human-led: the agent interviews the scientist and writes down what they
decide. It runs no compute. Everything it produces lands on `main` in a daytime session, because
the plan's hypothesis and kill criteria are human-reviewed by construction (RULES / AUTORESEARCH-RULES §5).

## What it does

0. **New program or existing one?** Look for a collection whose entries include a `plan` article.
   - **Exists** → this is a steer: go to step 4 (amend) or step 5 (close).
   - **None** → this is a birth: steps 1–3.

1. **Define the falsifiable arc (interview the scientist).** Draw out, and write down in the
   scientist's words:
   - The **question**, stated so a result could prove it wrong.
   - Why it matters — what it confirms or rules out in the field.
   - The **baseline** the stack must reproduce before any novel claim is trusted (a published
     number, e.g. a LIF net's accuracy on N-MNIST/SHD). This becomes the first queue entry.
   - The rough arc: the two or three questions after the baseline. Not a full programme — enough
     to aim the first nights.
   Don't invent the science. If the scientist is vague, ask; a program with no falsifiable
   question is the thing this runbook exists to refuse.

2. **Pick the collection and register it.** Choose a slug (`neuron-models`, `stdp-window`, …),
   add it to `demolab.yaml` (`collections:` + `collection-order:`, RULES §6.5) with a label and
   one-line description.

3. **Scaffold the three pieces.** Model the articles on an existing writing (`meta` + `body`):
   - `writings/plan.typ` — `collection: <slug>`, the question and baseline in the body, an empty
     `== Amendments` heading, and an empty `queue` (frontmatter or fenced block; schema in
     AUTORESEARCH-RULES §2). Leave the queue for **PLAN** to fill.
   - `writings/log.typ` — `collection: <slug>`, a one-line "program opened" first entry, digest
     slot at top.
   - The **night-shift contract** — `night-shift.yaml` beside the collection (or a fenced block
     in the plan) with budgets, `scope.collection`, and stop conditions (AUTORESEARCH-RULES §4).
   Build to confirm it renders (`task build` / the running `task dev`), then commit on `main`
   ("Open program <slug>"). That commit is the pre-registration anchor.

4. **Amend (existing program, a pivot).** A hypothesis or kill criterion never gets edited in
   place once an experiment has run against it. Add a **dated entry** under `== Amendments` in the
   plan body saying what changed and why, on `main`. Then hand off to **PLAN** to queue whatever
   the pivot implies.

5. **Close (retrospective).** When the arc is done, append a **Retrospective** section to the
   plan: what held, what died, what the log's decision trail says in hindsight, and the honest
   limitations. Flip lingering `queued` entries to a terminal status. The collection now reads,
   top to bottom, as question → journey → verdict.

---

## Agent contract
- **Triggers** — `AUTORESEARCH`, "start a research program", "set up an autoresearch program",
  "steer/amend the program", "close out the program".
- **Gates** — runs on `main`, daytime, no compute. Refuse to scaffold a program with no
  falsifiable question or no baseline. Do not fill the queue here — that's **PLAN**.
- **Report & apply** — interview interactively; write the plan/log/contract; build to confirm;
  commit on `main`. Then point the scientist at **PLAN** to queue the first experiments.
