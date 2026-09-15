# Goal: A compact autonomous PING experiment

1. Create one experiment in `/Users/eoin/pinglab` inspired by `papers/Lowet-et-al-2015.pdf`.
2. Finish within two hours. Immediately record the timezone-aware start and deadline, reserving about 20 minutes for review and corrections.
3. Use Sol with medium reasoning and record requested and verified runtime settings in the README.
4. Follow all repository instructions and the current Writing, Experiment Runner, and Storage Guides.
5. Allocate a new experiment ID and save this goal verbatim as `PROMPT.md` before implementation.
6. Read the paper and survey existing repository topics, then autonomously choose a focused, tractable question that represents the paper and connects meaningfully to existing work. Record the question, rationale, assumptions, connections, and completion criteria before simulation.
7. Choose the model, design, comparisons, measurements, analysis, and interpretation independently; do not predetermine the method or numerical result.
8. Keep a timestamped README journal of decisions, attempts, observations, failures, revisions, figure inspections, time, and compute budget.
9. Use snnlang, snnsim, and snnviz, reusing existing infrastructure. Keep handwritten experiment code to roughly 2,500 lines, excluding documentation, the article, generated outputs, and reused code; justify any excess beforehand.
10. Use at most ten local compute attempts, each under 180 seconds. Record every attempt and do not search selectively for a preferred result.
11. Produce exactly three final figures. Manually open and sanity-check every generated figure, including diagnostics and discarded plots, after each creation or change; inspect finals at full resolution and article size and journal each decision.
12. Write a concise article with labelled Introduction, Results, Methods, Discussion, and Conclusion sections. Separate observation from interpretation and state limitations and uncertainty.
13. A null, failed, mixed, or ambiguous result is insufficient because the paper establishes feasibility. Iterate within the limits until clear, reproducible success is achieved; otherwise report the project as incomplete without disguising failure.
14. Ask an independent Sol subagent with medium reasoning to review the provisional work read-only, including the actual final figures. It must assess scientific validity, claims, reproducibility, compliance, and clarity without running simulations.
15. Address substantiated review findings within the remaining time and journal changes, disagreements, and unresolved issues.
16. Deliver the reproducible compute, analyse, and present pipeline, validated outputs, article, preview link, saved prompt, and journal; summarize the question, findings, limitations, review, attempts, settings, and elapsed time.
17. Stop at the deadline and report incomplete work honestly. Finish locally; commit, push, and publication require separate instructions.
