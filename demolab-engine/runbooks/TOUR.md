# Runbook: Tour the lab

Triggers: **"tour"**, "walk me through this repo", "give me a tour", "orient me", "what's in here", **TOUR**. Goal: a guided orientation to an existing lab — what's here, how it fits, what each experiment concluded, and where to start — for a new collaborator, a reviewer, or your own future self.

DOCTOR *audits* the repo against the rules; TOUR *orients* a human in it. It reads the content and narrates it — no changes, no fixes, just a clear map. Adapt the depth to who's asking: a new lab member wants the on-ramp, a reviewer wants the results.

## 0. Read the lab
Read `demolab.yaml` (branding + collections), each experiment's **writeup** (the claim + one-line takeaway), and skim the runners/tools (what computes what). Note how entries group into collections.

## 1. Give the tour — top down
Narrate a short guided walk, not a file listing:
- **What this lab is about** — from the branding + the spread of experiments, in a sentence or two.
- **The collections** — each group and what it covers.
- **The experiments** — for each (or each collection), the one-line takeaway: what it asked and what it found. Most important / newest first.
- **The tools** — the reusable science, and which experiments use them.
- **Where to start** — point the newcomer at the best entry to read first, and the `task dev` site to browse.

## 2. Orient for their goal
Ask what they're here to do, and tailor the close:
- *Contributing?* → GETTING-STARTED's add-an-experiment flow, and RULES.
- *Reviewing?* → the results, then RED-TEAM.
- *Taking it over?* → flag anything undocumented or fragile you noticed while reading.

Keep it warm and readable — this is the door into someone else's (or past-you's) thinking.
