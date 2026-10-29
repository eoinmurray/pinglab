---
title: Experiment Rules
author: Eoin Murray
collection: notes
description: Experiment scope, layout, configuration and native evidence.
created_at: "2026-10-14"
updated_at: "2026-10-29T17:43:19Z"
---

# Experiment Rules

<a id="execution-permission"></a>

## Execution and scope

1. Run simulations only with an explicit request or permission, including training,
   inference, sweeps, diagnostics and simulation-backed tests. Code, prose and
   diagram edits do not authorize execution. Existing authorization remains valid
   within scope; ask before expanding it. Compilation for diagrams is allowed.
2. Perform only requested stages. Compute → analyse → present → validate → back up
   → publish → prune describes order, not automatic follow-up work.
3. Inspect source and diff for prose-only edits. For executable changes, use focused
   checks; broaden them when failures or scope justify it. Checks that simulate
   still require execution authorization. State when no simulation was run.

<a id="file-layout"></a>
<a id="structure-and-readability"></a>
<a id="configuration"></a>

## Layout and configuration

1. Keep experiment-specific work in `experiments/expXXX/`: `recipe.py` owns settings
   and network definitions; optional `dataset.py` owns preprocessing; `compute.py`,
   `analyse.py` and `present.py` own their stages; `test.py` holds all experiment
   tests. Limit root Python files to these. Keep imports free of execution effects.
2. Reserve `custom/` for genuine extensions and execution adapters, not ordinary
   stage code. Keep only work needed for the report and its evidence. README:
   brief goals, CLI files and arguments. See `experiments/exp141` for structure.
3. `configuration(...)` returns a fresh resolved dictionary with defaults together
   in `dict(...)`. Validate explicit overrides before deriving dependent values;
   route CLI settings through it and record what execution actually uses. Explain
   scientific meaning, units and assumptions. `author_network(...)` compiles the bundle.

## Identity and dependencies

1. Allocate a never-used, zero-padded `expXXX`, checking directories, Git history
   and Pingstore allocation records, including deleted experiments.
2. Reuse maintained shared helpers directly in `experiments/`; obtain permission
   before adding one. Never import another experiment's code or scratch code.
   Consume external scientific inputs through explicit validated Pingstore runs.

<a id="snnlab-first"></a>
<a id="network-diagrams"></a>
<a id="network-snapshots"></a>

## Native evidence

1. Prefer installed SNNLab features over custom scientific machinery. Save the
   native bundle, one compiled circuit diagram per architecture, and native
   snapshots: simulation-start for simulation/inference; final-training and
   best-validation for training. Additional evidence retention follows Pingstore.

<a id="saving-and-consuming-results"></a>
<a id="migrating-scratch-work-to-an-experiment"></a>
<a id="parallel-compute"></a>

## Task-specific references

Read only the relevant section of [Experiment Procedures](/notes/guides/experiment)
for native calls, migration or parallel workers. Read [Compute](/notes/rules/compute)
when operating resources and [Pingstore](/notes/rules/pingstore) when producing,
consuming or managing runs.
