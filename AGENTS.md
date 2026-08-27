# Pingstore filesystem convention

Pingstore is a filesystem convention, not a service, database, or general
management CLI.

- `pingstore discover` is a narrow read-only integration command: validate
  completed runs and emit Demolab discovery JSON. It must not select, mutate,
  materialize, upload, or prune runs, or persist a catalogue.

- Store every completed run at `.pingstore/runs/<run-id>/`, containing only
  `run.json`, `README.md`, `export/`, and `presentation/`.
- `run.json` uses `pingstore.run/v2` and is authoritative provenance. `README.md`
  holds human notes. `export/` contains scientific outputs and execution records
  and may have subdirectories. `presentation/` contains only regular files,
  never subdirectories or symlinks; use descriptive flattened filenames.
- Materialize only `presentation/` into `.artifacts/<experiment>/`, copying it
  exactly without extension filtering. Presentation metadata is a projection,
  never an independent provenance authority.
- Validate the exact root layout and payload checksum before publication or
  consumption. The checksum covers README.md, export/, and presentation/,
  including nested metadata; it excludes only the authoritative run.json.
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
  bundles, or a general Pingstore management CLI. The read-only discovery
  command above is the sole CLI exception apart from the documented migration utility.

# Experiment writing

Before creating or editing `writings/expXXX.typ`, read and follow
`writings/README.md`.
