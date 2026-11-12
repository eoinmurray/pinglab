---
title: Pingstore Rules
author: Eoin Murray
collection: notes
description: Simple run folders, explicit dependencies and incremental backups.
created_at: "2026-09-11"
updated_at: "2026-11-12T20:22:47Z"
---

# Pingstore

1. Scientific runs live in `.pingstore/runs/<id>/`. Each completed directory has a small `run.json`: `id`, `experiment`, optional `stage`, `created_at`, optional `description`, `depends_on` (run IDs), and optional `metadata`. Completed IDs and payloads are immutable; changed results receive a new ID.
2. Use `pingstore.library.Store` to insert, read, list, delete, back up and restore runs. `stage_run` is a convenience for experiment scripts, not an enforced workflow. Temporary work stays in `run.scratch`; retained files go in `run.export`. Configuration, native artifacts and detailed provenance belong in ordinary files such as `details.json`.
3. Reading and listing use metadata only. Scientific consumers validate the files and model properties they need. Storage never rehashes the archive or launches upstream work. Never fabricate missing evidence.
4. `depends_on` records direct scientific inputs, including other experiments. Keep replaced attempts and import history in scientific provenance rather than adding false operational dependencies. Missing dependencies remain visible and are not replaced automatically.
5. Backup uploads new runs concurrently and writes completion receipts last. It never downloads scientific payloads for validation. A receipt records successful transfer, not an independent read-back audit. Backup never deletes remote history or publishes the site.
6. Deletion names explicit run IDs and one storage target, local by default. Inspect dependant warnings, then confirm. Deletion can proceed despite dependencies; never cascade automatically. There is no automatic pruning or deployment retention.
7. Retain native evidence needed for reproduction, analysis or continuation. Avoid unnecessary copies and excessive small files. Keep disposable work outside completed runs.

See [operations](/notes/rules/pingstore-operations) for commands and migration.
