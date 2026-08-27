# Verified Gold-2 ancestor

On 2026-08-27, the user authorized checking R2, retaining exp022's ancestor,
and completing the already staged exp022, exp023, exp024, exp044 and exp081 studies.

## Source and verification

- Exact source: `r2:pinglab/campaigns/gold-2` (archive ID `gold-2`).
- Scientific execution: Wilkes/Slurm, with 60 inherited cells and 42 repaired
  cells. Their original job records and distinct producer commits are retained.
- The repaired bank is complete, not an overlay to merge with the base bank.
- A download-based `rclone check` found zero differences in 1,574 selected
  archive files: the repaired exp022 bank, its archived presentation, and source
  records. This is not a verification of the entire Gold-2 collection.
- All 408 scientific files in the local 102-cell bank matched the R2 inventory.
- All 35 carried diagnostic/comparison images matched the archive. The recorded
  mapping accounts for nested archive raster paths and flat presentation names.

The independent historical snapshot is `.r2/exp022-ancestor-20260827/archive/`.
The adjacent `verification.json`, `verify.py`, `remote-check.txt` and `rclone.log`
retain hashes, exact scope, commands and evidence. R2 and `.r2/gold-2/` were not
modified. The snapshot remains historical evidence, not an operational v3 run.

## Operational ancestry

The current bank is `exp022-r001-compute`. The later
[source-neutral naming migration](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md)
removed suffixes from all 21 completed runs without changing scientific ancestry.
The IDs below record the earlier repairs and handover.

At this ancestry repair, the imported bank retained the earlier
`exp022-r001-compute-slurm` identity. The separately authorized
[local-origin correction](README.md#local-import-origin-correction) subsequently
restored `exp022-r001-compute-local`: that suffix described the local import;
historical scientific execution remains Slurm. Original execution and import
timestamps are unchanged. Each migration records its own local timestamp.

The bank is self-contained and now has no operational inputs. The former missing
`exp022-gold-2-repaired-slurm` input is preserved under `historical_ancestor`,
alongside the verified archive record, rather than being silently discarded.
The same distinction repairs the historical-image input of
`exp022-r003-present-local`. Original manifests and all changed files remain
recoverable. Future presentation can reuse these validated v3 presentations;
it cannot consume v2 or launch new diagnostic simulations.

Twelve affected exp022, exp024 and exp044 runs were re-pinned in dependency order.
Exp044's user-approved selected-bank boundary remains intact. All 18 completed
local runs passed payload, layout and input-pin validation after activation.
There was no training, inference, numerical analysis, plotting, materialization
or publication during this storage repair.

## Retention and rollback

The complete original run tree, including incomplete hidden directories, is at
`.r2/exp022-ancestor-20260827/migration/originals/`. The adjacent migration journal,
before/after inventories, `../migrate.py` and `../activate.py` describe preparation
and activation. Two failed preparation attempts remain separately retained; neither
changed the live tree. Every changed completed run also retains its original
manifest and migration script under `provenance/ancestor-repair-20260827/`.

Do not prune these records or restore an individual run from this checksum-pinned
chain. Recovery requires separately authorized work, stopped readers/writers and
validation of the complete restored chain. The originals retain their known
historical-reference gaps and must not be silently reactivated.

## Five-study handover

Targets: Storage Guide 2.0.0, Experiment Runner Guide 2.0.0 and Writing Guide
8.0.0. These studies are ready for human review; no publication was performed.

| Study | Reviewed presentation | Work completed |
| --- | --- | --- |
| exp022 | `exp022-r004-present-local` | Verified ancestor, repaired lineage, explicit v3 source roles, scientific article terminology and equation layout |
| exp023 | `exp023-r003-present-local` | Retained production evidence; integrated illustrative-cell selection with current reconstruction; updated execution notes |
| exp024 | `exp024-r005-present-local` | Re-pinned repaired bank, rejected v2 inputs, retained drift definitions and Discussion; consolidated routine Methods summaries |
| exp044 | `exp044-r004-present-local` | Preserved selected-bank boundary and production measurements; re-pinned lineage and reviewed the already compliant article |
| exp081 | `exp081-r003-present-local` | Retained full-profile evidence and theoretical appendices; consolidated routine aggregation and updated execution notes |

Verification:

- 366 targeted tests passed, including storage, discovery, publication rejection,
  campaign orchestration, all five staged studies and article inputs.
- The final audit validated 18 completed runs and all input pins.
- All 606 existing export files remained byte-identical; no scientific value changed.
- Discovery returned eight present runs, excluding compute/analyse stages.
- All five articles compiled against real selected presentations: 19, 7, 3, 4
  and 7 pages respectively. Result and Methods pages were visually inspected.
- `git diff --check` passed. Ruff reported no new diagnostics against the starting
  code; 54 pre-existing diagnostics remain, mainly exp022 wildcard imports and
  import ordering. This is not a claim of a clean repository-wide lint run.

Logs, final verification, isolated render pages and lint comparison are retained
under `.r2/exp022-ancestor-20260827/`. The previews use copies of validated
presentation exports, not `.artifacts/` or automatic upstream execution.
Unrelated legacy runners and their historical helpers remain outside this scoped
five-study migration; they are not granted operational v2 compatibility.
