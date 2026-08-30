# Pingstore evidence pruning audit

This is a read-only file-level classification of every file below
`.pingstore/runs/*/export/evidence/`.

- `downstream-used.tsv` lists files read by a completed descendant experiment
  stage or Pinglab's presentation projection.
- `downstream-unused.tsv` is the requested complete list of files not read by
  those downstream consumers.
- `run-metadata-only.tsv` lists files not read by descendants but referenced by
  a path or directory declaration in their owning `run.json`.
- `deletion-candidates.tsv` lists files for which the audit found neither kind
  of reader or reference.

Each TSV row is `<repository-relative path><TAB><size in bytes>`.

The classification traces production readers, not tests, README prose, or the
blanket payload checksum. Deleting any listed file would still mutate an
immutable visible run, invalidate its payload digest, and invalidate every
descendant that pins that digest. This list is therefore an approval and
migration-planning input, not authorization for in-place deletion.

`build_audit.py` regenerates the manifests without modifying Pingstore.
