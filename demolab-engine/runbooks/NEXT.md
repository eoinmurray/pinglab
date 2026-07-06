# Runbook: Propose the next experiments

Triggers: **"what next"**, "what should I run next", "propose next experiments", "where do I go from here", **NEXT**. Goal: read the whole arc of the lab's results and propose the follow-up experiments worth running — the missing control, the ablation, the parameter regime left unprobed, the confound left open — as concrete, scoped suggestions the user picks from.

This is the research-direction pass. Its edge: it reads *every* experiment's runner, numbers, and writeup at once, so it sees the arc a single-experiment view can't — what's been established, what's assumed, what's untested. It **proposes; the scientist decides.** Nothing here runs compute; it's reasoning over the committed record.

## 0. Read the whole lab
For each experiment, read the **writeup** (the claim), the **`numbers.json`** (what was actually measured), and the **runner** (what was done + the parameter ranges). Build a short internal map: what each entry establishes, and the parameters/assumptions it held fixed.

## 1. Find the openings
Across that map, look for:
- **Untested regimes** — a parameter held fixed that the conclusion might hinge on (one value swept, one seed, one input size).
- **Missing controls / baselines** — a claimed effect with nothing to compare against.
- **Unruled-out confounds** — an alternative explanation the current runs can't distinguish.
- **Natural follow-ups** — the obvious next question a reader asks after the last result.
- **Ablations** — which piece of the model is actually doing the work?
- **Robustness** — does the result survive a different seed / dataset / perturbation?

## 2. Propose — scoped and ranked
Present a short **numbered** list (highest-leverage first) of concrete experiments, each with: the **question** it answers, the **one thing to change** from an existing experiment (give the id to model on), roughly **how big** it is, and **why it matters** (what it confirms or rules out). Keep each small — one variable moved — per the loop. Propose the next few runs, not a whole research programme.

## 3. Hand off
The user picks a number; scaffold that one via GETTING-STARTED's first-experiment steps (default inline, model on the existing experiment). The ones they don't pick aren't discarded — note them on record for later.
