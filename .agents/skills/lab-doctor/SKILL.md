---
name: lab-doctor
description: Use only when the user explicitly invokes $lab-doctor to audit Pinglab governance, structure, provenance, and publication contracts.
---

# Lab Doctor

Activate this skill only for an explicit `$lab-doctor` command. Do not activate
it from semantic similarity, automatic selection, or an ordinary audit request.

`$lab-doctor` is read-only. Audit:

1. The Pinglab lexicon and corresponding project skills.
2. The adopted Demolab contract allowlist without importing other Demolab
   policy.
3. Repository zones, writing metadata, artifact references, and ignored build
   locations.
4. Provenance completeness and separation of raw runs from publication inputs.
5. Campaign/worktree isolation and obvious paid-compute or Git authorization
   hazards.

Run `scripts/validate-agent-system.sh`, then perform proportionate semantic
checks. Report each violation with its local Pinglab rule and file path. Do not
invoke Demolab's `DOCTOR` runbook unless the user explicitly requests it.
