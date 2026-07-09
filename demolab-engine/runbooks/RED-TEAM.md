# Runbook: Red-team an experiment

> An adversarial, pre-submission self-review of an experiment's **integrity** — does the writeup's
> claim actually hold up against the code and the data? Pairs with **STEELMAN** (RED-TEAM *attacks*
> a result, STEELMAN *defends* it — run them as a pair to find the weak points *and* the real
> strength).

## When to use
Before anyone else sees the result — the pre-submission integrity pass. It completes the trio:
**DOCTOR** audits the structural rules, **LINT** audits the prose, **RED-TEAM** audits whether the
result is *true and defensible*. Its edge over a human co-author: it reads the code, the data, **and**
the claims, so it can cross-check them against each other — the check nobody does by hand. Scope
defaults to one experiment; pass **"all"** for a whole-lab pre-submission sweep. **Verify every
finding against the actual run before reporting it** — a red-team that flags every hedge is noise
nobody runs twice. Report *"your caption claims a 2× speedup but `numbers.json` shows 1.3×"*
(confirmed, actionable) — not *"consider whether the claim is fully supported"* (vague).

## What it does

0. **Build is green.** `task build` must compile and `task test` must pass first — critiquing a
   broken experiment is noise.

1. **Claim ↔ data cross-check (the core).** Read the writeup's claims against
   `artifacts/data/<id>/numbers.json` and the data behind each figure (its CSV/JSON/`.npz`/…). For
   every substantive claim in the prose and captions, confirm the data actually backs it:
   - A stated **number** matches `numbers.json` (not a drifted literal — HOUSESTYLE H9).
   - A stated **trend** ("increases with", "converges", "plateaus") is actually present — read the
     underlying data, don't take the sentence's word for it.
   - A stated **comparison** ("2× faster", "beats the baseline") has the baseline *in the run*, and
     the ratio is what the numbers say.

   Flag every claim the data doesn't support, as "claims X, data shows Y".

2. **Statistical & experimental rigor.**
   - **Uncertainty** — are error bars / CIs shown where a result is stochastic? Is **n** reported?
   - **Determinism** — is the run **seeded**? An unseeded stochastic result could be a fluke you
     can't reproduce (DOCTOR flags this too).
   - **Baseline / control** — "we improved X" begs *compared to what?* Is the baseline in the repo
     and run the same way?
   - **Cherry-picking smell** — were parameters chosen post-hoc to flatter the result? Are only the
     runs that worked shown? Ask what's *not* here.

3. **Figure & caption integrity.**
   - Does each figure actually **show** what its caption asserts (HOUSESTYLE H16–H20)?
   - Axes labelled with **units**; the panel a caption references actually exists.
   - No visual that oversells — truncated axes, a cherry-picked window — without saying so.

4. **Provenance.**
   - Was the published run built from **committed** code (footer clean, not "uncommitted changes")?
     A dirty run means the page may not match any recoverable code state.
   - Do the tool's tests exercise the science the claim rests on, or just smoke-test that it runs?

5. **Report.** One report, grouped by severity, **each finding verified against the run**:
   - **Integrity** (claim ↔ data mismatch, drifted number, unsupported comparison) — fix before
     anyone sees it.
   - **Rigor** (no error bars, unseeded, missing baseline) — fix or justify.
   - **Presentation** (figure/caption oversell) — the author's call.

   For each finding: the claim, what the data actually shows, the `file:line`, and the concrete fix
   (seed it, add the baseline, show the CI, soften the claim). Apply only what the user approves,
   then re-check the ones you changed.

---

## Agent contract
- **Triggers** — `RED-TEAM`, "red-team", "red team my experiment", "critique this experiment", "poke
  holes in this", "review my results".
- **Gates** — §0 must hold; critiquing a broken experiment is noise.
- **Report & apply** — verify every finding against the actual run before reporting; drive it
  interactively. Report grouped by severity with `file:line` + concrete fix; apply only what the
  user approves, then re-check the ones you changed. If STEELMAN also ran, reconcile: the final
  claim should survive the attack *and* capture the strength.
