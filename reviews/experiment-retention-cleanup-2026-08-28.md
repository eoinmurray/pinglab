# Experiment data-retention cleanup

Scope: future execution of the 11 agreed experiments. This changes recording and
serialization, not scientific settings, durations, seeds, sample grids, checkpoint
selection or publication. Existing v3 evidence remains readable. No existing run
or R2 archive was rewritten or deleted, and no production experiment was run.

| Experiment | Removed from future data operations |
| --- | --- |
| exp023 | Full f–I snapshots, replaced by E/I spike totals; scope input/readout data and unselected voltage/conductance columns. |
| exp025 | Unused snapshot traces, I population traces, unused rate arrays and output events; write-then-extract NPZ compaction. |
| exp037 | Unused snapshot trajectories and write-then-extract compaction; payloads are moved out of job scratch. |
| exp038 | Unused snapshot trajectories and full f–I trace recording; write-then-extract compaction. |
| exp041 | I population traces and unused snapshot/input/readout trajectories. |
| exp044 | Voltage, conductance, input and readout snapshot trajectories. |
| exp046 | Unused trajectories, output events, I per-cell rates and per-sample rate arrays. |
| exp049 | Unrelated weight matrices, I population traces and unused snapshot trajectories; write-then-extract compaction. |
| exp054 | Non-spike trajectories, output events, burn-in events and redundant repacking. Dynamics and full-window rate reporting still include burn-in. |
| exp080 | Native compute's decoder checkpoint files. Selected weights remain in memory through evaluation; histories and held-out correctness remain on disk. |
| exp085 | Unused graph channels, duplicate named spike recordings, and prefix trajectories. PRC and pathway jobs retain only their analysis-required channels. |

Shared NPZ selection is opt-in and uses lossless compression on the first write.
Graph selection skips collection of unrequested channels while preserving outputs
and branch state. The defaults used by other experiments are unchanged.

## Verification

- **399 experiment tests passed.** The fast simulator suite passed 389 tests
  (two expected failures, 103 slow tests excluded). Additional focused runs
  passed 159 simulator/writing-status tests and 158 CLI/config/execution tests;
  these suites overlap and their counts should not be added. Ruff and diff
  whitespace checks passed on the owned Python changes.
- Exact comparisons cover retained inference arrays, sparse events, weight dumps,
  f–I counts, selected scope traces, graph outputs/state and decoder predictions.
- Stage fixtures cover analysis, presentation, old v3 readers, corrupt evidence,
  atomic completion and rejection of v2 inputs.
- Read-only discovery and input-reference checks succeeded. The 81 run manifests
  present at task start are unchanged. Availability badges for the 11 experiments
  and dependent exp092/exp109 agree with their declared inputs and local data;
  no article edits were needed for this cleanup.
- The separate CLI argument-policy gate has six baseline failures: existing
  `--frequency-source` flags in exp033/exp046/exp054 and `--shard-index` flags in
  exp037/exp042/exp082. These exact failures were verified against HEAD and were
  not changed here.

No production runtime benchmark was performed. Runtime savings from reduced
recording and serialization are not quantified.
