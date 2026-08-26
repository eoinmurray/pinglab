# Pingstore filesystem convention

Pingstore is a filesystem convention, not a service, database, or CLI.

- Store every completed run at `.pingstore/runs/<run-id>/`, containing only
  `run.json` and `files/`.
- A run ID begins with its experiment and ends with its execution source, for
  example `exp097-r022-local` or `exp097-r023-slurm-wilkes-48291`.
- Write a run first as `.pingstore/runs/.<run-id>.tmp/`; atomically rename it to
  the visible run ID only after every output and `run.json` is complete.
- Treat visible run directories as immutable. A hidden temporary directory is
  incomplete and must never be used as evidence or publication input.
- Keep complete execution provenance in `run.json`; do not infer provenance
  from filesystem timestamps or the run ID alone.
- Local and HPC execution use this same run-folder contract. Allocate HPC run
  IDs before submission so concurrent jobs cannot claim the same identity.
- Do not reintroduce collection/experiment directory nesting, catalogues,
  lifecycle states, automatic official selections, preview overrides, archive
  bundles, or a Pingstore command-line interface.

# Experiment writing

Before creating or editing `writings/expXXX.typ`, read and follow
`writings/README.md`.
