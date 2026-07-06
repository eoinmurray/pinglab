# Runbook: Reproduce a paper's result

Triggers: **"from paper"**, "reproduce this paper", "replicate figure N", "implement this paper", **FROM-PAPER**. Goal: given a paper (PDF, arXiv link, or DOI), scaffold experiment(s) that reproduce its **key result** — a figure or a headline number — in your stack, so you can check it yourself rather than take it on faith.

Reproducing a paper is how you learn a method and how you verify a claim, and it's the natural on-ramp for a scientist arriving with "I read this — does it hold up?". The agent reads the paper *and* builds the experiment, wiring the reproduced numbers to the run so what you get is checkable, not asserted. Drive it interactively — **propose the plan and confirm before building**; a paper is a spec you implement, not copy.

## 0. Get the paper + the target
Fetch it (WebFetch the arXiv/DOI, or read the PDF the user points at). Ask **which result** to reproduce — usually one figure or one headline number. Don't attempt the whole paper; pick the load-bearing claim.

## 1. Extract the recipe
Pull what's needed to run it: the **method**, the **parameters** (flag any the paper leaves implicit as *assumptions*), the **inputs / data**, the **metric**, and the **expected result** (the number or trend to match). Write this recipe back to the user so they can catch a misreading before you build anything.

## 2. Plan the experiment(s)
Map the recipe onto demolab: usually one experiment (`expNNN`), **inline** by default (a reproduction is a one-off until reused; RULES §4), the method in the runner, the parameters as config. If the paper's data isn't available, say so and propose the closest substitute — with the caveat recorded in the writeup.

## 3. Build, run, compare
Scaffold runner + writeup (model on an existing experiment). Stage the reproduced metrics to `numbers.json`, `task run` + `task build`, then **compare against the paper's reported value**: does your number / figure match? Report the delta **honestly** — a partial or failed reproduction is a *result*. The writeup says what matched, what didn't, and every assumption you had to make. **Never hand-type the paper's number as if it were yours** — cite it as the target, show yours from the run (RULES §6.2, HOUSESTYLE H9).

## 4. Hand off
Open the page + PDF. The writeup cites the source, states the target, what you reproduced, and the gaps. Offer **RED-TEAM** (does your reproduction actually hold up?) and **GROUND-CLAIMS** (are the paper's claims you leaned on real?).
