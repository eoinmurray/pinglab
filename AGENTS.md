# Pingstore filesystem convention

Pingstore is a filesystem convention, not a service, database, or general
management CLI.

- `pingstore discover` is a narrow read-only integration command: validate
  completed runs and emit Demolab discovery JSON. It must not select, mutate,
  materialize, upload, or prune runs, or persist a catalogue.

- Store every completed run at `.pingstore/runs/<run-id>/`. New staged runs use
  `pingstore.run/v3`, with required `run.json` and `export/`, optional `README.md`
  and `provenance/`, and no other root entries or symlinks.
- `run.json` is authoritative provenance and must declare compute, analyse or
  present. `export/` holds stage outputs; nesting is allowed for compute/analyse,
  while present exports contain only flat regular files. `provenance/` holds
  retained execution scripts, patches and evidence records; README holds notes.
- Discovery validates all completed runs, then lists present runs with nonempty
  output beyond bookkeeping. Materialize only a present run's entire `export/`
  into `.artifacts/<experiment>/`, without extension filtering; reject publication
  of compute/analyse runs. Presentation metadata is only a provenance projection.
- Keep v2 reads and unmigrated legacy capture compatible: v2 retains required
  `run.json`, `README.md`, `export/`, `presentation/`. Untyped v2 runs may expose
  their presentation content; typed v2 compute/analyse runs may not. Use shared
  version-aware helpers, not hardcoded consumer paths. Do not rewrite existing
  runs or reservations without an explicitly authorized migration.
- Validate the exact root layout and payload checksum before publication or
  consumption. The checksum covers every payload file, including optional notes,
  evidence and nested metadata; it excludes only the root authoritative run.json.
- New staged run IDs include compute, analyse or present, for example
  `exp022-r001-compute-local`; legacy IDs remain valid and unchanged. IDs begin
  with the experiment and end with execution source. Record stage and input
  references in run.json, never infer provenance from the name.
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

# Experiment execution

Before creating or editing experiment execution code, read and follow
`experiments/README.md`. Compute, analyse and present complete independently;
downstream stages never launch upstream work or automatically publish.

# Experiment writing

Before creating or editing `writings/expXXX.typ`, read and follow
`writings/README.md`.
