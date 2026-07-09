# Runbook: Steelman a result

> Build the strongest *honest* case for an experiment's result — without overclaiming.
> Pairs with **RED-TEAM** (it attacks, this defends). Run them together: the attack finds
> the weak points, the steelman finds the real strength, and the truth is where they meet.

## When to use
When you suspect you're *under*-selling a genuine result — burying your best evidence, or
stating a claim more weakly than the data supports. This is not spin: everything it asserts
must clear the same bar RED-TEAM holds, backed by the run's own numbers. Its job is to make
sure a real result gets stated at full strength, not to invent one.

## What it does

0. **Build is green.** `task build` and `task test` pass — steelmanning a broken run is
   fiction.

1. **Find the real strength.** Read the writeup, `numbers.json`, and the figure data
   (CSV/JSON/…). Look for:
   - **The strongest claim the data actually supports** — often tighter or bigger than the
     writeup currently says (a clean monotonic trend, a large effect, a result robust across
     seeds).
   - **Underused evidence already in the run** — a metric computed but never mentioned, a
     figure that makes the point better than the prose does.
   - **The sharpest framing** — the one-sentence version a reader remembers, still fully
     backed by the data.

2. **Pressure-test it.** For each strengthened claim, confirm it against the run — the same
   verification RED-TEAM demands. If the data doesn't back the stronger version, drop it. A
   steelman that outruns the evidence is just a fresh overclaim.

3. **Report.** Present the strongest honest claim, the evidence for it, and a suggested
   rewrite of the lead/caption that states it crisply. Note where the current writeup
   *under*-sells.

---

## Agent contract
- **Triggers** — `STEELMAN`, "steelman", "make the case for this", "strengthen my claim",
  "best case for this result".
- **Gates** — §0 must hold; steelmanning a red build is fiction.
- **Report & apply** — present claim + evidence + suggested rewrite; apply only what the
  user approves. If RED-TEAM also ran, reconcile: the final claim should **survive the
  attack *and* capture the strength.**
