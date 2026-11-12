---
title: Pingstore Rules
author: Eoin Murray
collection: notes
description: Current scientific runsets, validated storage and inspected cleanup.
created_at: "2026-09-11"
updated_at: "2026-11-12T12:03:36Z"
---

# Pingstore Rules

<a id="scientific-evidence"></a>

## Evidence

Scratch projects under `.scratch/<project>/` use their own `data/` folder,
overwritten on each run, instead of Pingstore. See
[Experiment Rules](/notes/rules/experiment#file-layout). The requirements below
apply to Pingstore runs; existing runs remain immutable.

1. Code lives in Git; scientific runs live in `.pingstore/runs/` with verified R2
   backups at `r2:pinglab/pingstore/runs`. Completed run identities, payloads,
   checksums and input pins are immutable. Changed results or smaller exports need
   new runs. Never invent provenance or substitute figures for missing raw data.
   Record explicit metadata corrections in the run's README history.
2. Experiments use `pingstore.library`: create with `stage_run`, validate explicit
   input IDs and digests with `source_run`. Retained outputs go in `run.export`;
   temporary work in `run.scratch`. Record configuration and provenance in the run,
   without parallel manifests. Never select “latest” scientific inputs or launch
   upstream stages automatically.
3. Design runsets for space efficiency and manageable file counts. Retain native
   evidence required by Experiment Rules and outputs justified by downstream analysis,
   reproduction or continuation. Keep additional snapshots, recordings and optimizer/RNG
   state only for those consumers. Avoid redundant copies, unnecessary intermediate
   outputs and excessive per-step or per-sample files; use compact, consolidated formats
   where practical. Before substantial runs, estimate retained size and file count,
   and justify unusually large outputs. Keep disposable intermediates in `run.scratch`;
   existing runs remain immutable.

<a id="current-reporting-and-status"></a>

## Selection

1. Core resolves the current presentation and exact digest-matched runset for
   reporting. Runset dependency links stay within one experiment and advance from
   compute through analysis to presentation. Historical input pins remain
   provenance, independently of runset membership and retention. Reuse core's
   selection per page request/build; never substitute current
   recipe defaults for recorded evidence. Report broken or unavailable evidence.
2. Use `uv run pingstore status` for evidence and retention reasons. Unknown or
   unresolved does not mean unused. Only runsets selected by local reports or
   recorded active GitHub Pages evidence are marked in use. Protections, pending
   deployments and other retention reasons do not imply report use. A runset may
   contain several compute or analysis branches; there is no fixed three-run rule.

<a id="commands-and-backup"></a>
<a id="one-inspected-cleanup-across-both-stores"></a>
<a id="publication-retention"></a>
<a id="eviction-and-recovery"></a>

## Maintenance

1. Use supported `backup`, `prune` and `evict` commands. Inspect a separate dry-run
   plan, then confirm its complete unchanged hash and scope. Pruning defaults to
   both stores; eviction removes verified local copies while preserving R2 history.
   Never manually delete completed runs, allocation state or catalogs.
2. For retirement, remove the experiment's source/report declarations first, then
   scope pruning explicitly. Preserve shared consumers, protections, unfinished
   work and publication roots. Validation failure never authorizes deletion.
3. Resume interrupted operations with the same hash, scope and journal. Never
   bypass fences or erase journals. Preserve deployment roots according to
   [Publishing Rules](/notes/rules/publishing#evidence-and-retention). Finish with retained-graph
   and scoped inventory validation; report partial completion accurately.

Read only the relevant [Pingstore Operations](/notes/rules/pingstore-operations) section
before maintenance, recovery or changes to storage/reporting internals.
For site builds, deployment and publication retention, follow
[Publishing Rules](/notes/rules/publishing).
