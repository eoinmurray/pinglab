# Runbook: Steelman a result

Build the strongest *honest* case for an experiment's result — the most compelling framing, the best supporting evidence already in the run, and the tightest way to state the claim — without overclaiming.

**Triggers** — say any of these, or just `STEELMAN`: **"steelman"**, "make the case for this", "strengthen my claim", "best case for this result".

RED-TEAM attacks a result; STEELMAN defends it. **Run them as a pair:** the attack finds the weak points, the steelman finds the real strength, and the truth is where they meet. This is not spin — everything it asserts must be backed by the data (the same bar RED-TEAM holds). Its job is to make sure you aren't *under*-selling a genuine result or burying your strongest evidence.

## 0. Build is green
`task build` and `task test` pass — steelmanning a broken run is fiction.

## 1. Find the real strength
Read the writeup, `numbers.json`, and the figure data (CSV/JSON/…):
- **The strongest claim the data actually supports** — often tighter or bigger than the writeup currently says (a clean monotonic trend, a large effect, a result robust across seeds).
- **Underused evidence already in the run** — a metric computed but never mentioned, a figure that makes the point better than the prose does.
- **The sharpest framing** — the one-sentence version a reader remembers, still fully backed by the data.

## 2. Pressure-test it (so it's not spin)
For each strengthened claim, confirm it against the run — the same verification RED-TEAM demands. If the data doesn't back the stronger version, drop it. A steelman that outruns the evidence is just a fresh overclaim.

## 3. Report
Present the strongest honest claim, the evidence for it, and a suggested rewrite of the lead / caption that states it crisply — and note where the current writeup *under*-sells. Apply what the user approves. If you also ran RED-TEAM, reconcile: the final claim should **survive the attack and capture the strength.**
