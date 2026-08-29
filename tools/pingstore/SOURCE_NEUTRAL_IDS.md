# Source-neutral run IDs

On 2026-08-27, the user authorized retiring execution-source suffixes from run
IDs. Storage Guide 3.0.0 and Experiment Runner Guide 3.0.0 now require new IDs of
the form `exp022-r001-compute`, `exp022-r002-analyse`, `exp022-r003-present`.
The run-record schema remains `pingstore.run/v4`.

## Execution origin and compatibility

A local import and an HPC execution use the same ID shape. `run.json.origin`
records the execution origin; execution details and scientific ancestry remain
explicit metadata. For example, `exp022-r001-compute` still has `origin: local`
for its import and `scientific_execution.origin: slurm` for the historical training.
Removing the suffix does not make HPC computation local.

The allocator retains the experiment counter and stage, counts historical and
hidden reservations, and reserves before dispatch. Concurrent reservations with
different origins cannot claim the same identity. Origin is recorded in the
reservation and carried into the completed manifest.

Readers still accept existing valid counter-first, origin-suffixed v3 runs until
an explicit migration. Old suffixed reservations cannot be completed: reserve a
fresh neutral identity. This does not admit v2 or stage-first historical runs as
operational evidence, and does not rewrite any remote store.

## Completed migration

All **21 completed runs** in `.pingstore/runs/` were copied, renamed by removing
`-local`, and re-pinned in dependency order. The affected studies are exp022,
exp023, exp024, exp042, exp044 and exp081. No incomplete reservations were present.

Current input references, presentation lineage, selected-bank references and
export identity metadata were updated together. Every run retains its original
manifest, changed files and migration script under
`provenance/source-neutral-id-migration/`. Earlier migration history, execution
commands, timestamps, scientific origin, source patches and historical import
plans remain unchanged. Two exp022 reservations from an earlier stage-first
identity were deliberately retained as historical evidence. Old figure stamps
and execution paths also remain evidence of their original operations.

Verification:

- All 21 runs passed v3 layout, payload checksum and recursive input-pin validation.
- All **685 export files** were checked. Eighteen JSON files changed only in run-ID
  metadata; the other 667 export files are byte-identical. No scientific values,
  checkpoints, arrays or figure bytes changed.
- The exp022 retained-presentation checks and exp042/exp044 bank-evidence checks
  passed against the new identities.
- Discovery returned nine present runs. Demolab preview rebuilt without errors,
  preserving its selections and the same evidence under the new IDs.
- **356 targeted tests passed**, with eight existing exp044 plotting warnings.
  Scoped storage lint and `git diff --check` passed.

No simulation, experimental analysis, plot regeneration, publication, R2 writes,
commit or push was performed as part of this naming migration.

## Recovery evidence

The complete rollback tree is
`.r2/source-neutral-ids-qwijdj6d/originals/`. The adjacent `migration.json`, before
and after inventories, `scientific-preservation.json`, `consumer-verification.json`,
preview snapshots and scripts record preparation, activation and verification.
Preparation stopped once on a historical reservation mismatch; that unused copy
is retained separately and never became operational.

The preview processes were paused only while the complete run tree was switched,
then resumed. Recovery requires separate explicit authorization, stopped readers
and writers, and validation of the complete restored chain. Never restore just
one run: downstream references pin both its payload and authoritative manifest.
The saved `source-before/` files and `working-before.patch` record the pre-change
code and working state; do not blindly apply them over newer unrelated work.
