# Campaign lifecycle

Campaign state progresses through explicit gates:

1. **Plan** — name the scientific question, experiment set, source revision,
   compute target, expected outputs, acceptance criteria, and stopping rules.
2. **Run or resume** — preserve existing valid outputs, submit only authorized
   work, and record commands, jobs, exit states, and provenance.
3. **Review** — verify every required output, distinguish complete from partial
   collections, and retain scientific uncertainty.
4. **Archive** — inventory and verify the raw execution record through
   `runstore`; upload alone is not verification.
5. **Promote** — use the campaign's dedicated branch and worktree, promote only
   compact publication inputs, and inspect every generated diff.

`runs/` and R2 hold the raw record. `artifacts/` is the selected publication
view, not a duplicate archive. A repair subset is not a complete campaign
unless a reviewed composite explicitly supplies and identifies the unchanged
required outputs.
