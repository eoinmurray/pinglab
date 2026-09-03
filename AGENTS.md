# Repo rules

- Do not use the in app browser skill for checking renderings on .typ files on localhost:3000. Just present the link to me and I'll check myself.

# Pingstore filesystem convention

Before creating or editing storage code, read and follow
the versioned [Storage Guide](tools/pingstore/README.md).

Pingstore is a filesystem convention, not a service, database, or general
management CLI.

- `pingstore discover` is a narrow read-only integration command: validate
  completed runs and emit Demolab discovery JSON. It must not select, mutate,
  materialize, upload, or prune runs, or persist a catalogue.
- `pingstore prune` is the sole mutating maintenance command. It must use a
  no-write dry run followed by the exact complete SHA-256 plan hash, retain HPC
  runs, newest visible presentations, explicit pins, incomplete inputs, full
  ancestry and allocation high-watermarks, and abort on drift or active writers.
  It must not edit visible run contents or create catalogues or lifecycle states.

- Store every completed run at `.pingstore/runs/<run-id>/`. All operational runs
  require `pingstore.run/v4`, with exactly required `run.json`, `README.md`, and
  `export/`, and no other root entries or symlinks.
- `run.json` is authoritative provenance and must declare compute, analyse or
  present. `export/` holds run-wide files and singleton units as flat files;
  only scientific units containing at least two related files use one directory
  level. Deeper nesting and generic container directories are forbidden. Use
  `recording.npz`, `spikes.npz`, and `rasters.npz` for their distinct standard
  roles; `snapshot.npz` and `recordings.npz` are forbidden. Present exports contain only flat regular
  files. Execution metadata and temporary
  records belong in `run.json` or discarded writer scratch space. README holds
  the human-readable dated history. Do not create provenance sidecars.
- Discovery validates all completed runs, then lists present runs with nonempty
  output beyond bookkeeping. Materialize only a present run's entire `export/`
  into `.artifacts/<experiment>/`, without extension filtering; reject publication
  of compute/analyse runs. Presentation metadata is only a provenance projection.
- Require v4 for writers, readers, stage inputs, discovery and materialization.
  Reject v2/v3, including legacy capture and completion of old reservations; there
  is no compatibility exception for flat runners or historical presentations.
  Use shared validated v4 layout helpers, not hardcoded consumer paths.
  Preserve historical runs and reservations unchanged as non-operational evidence.
  Historical inspection for migration/recovery and migration itself require
  separate explicit authorization; never silently rewrite or reactivate them.
- Validate the exact root layout and payload checksum before publication or
  consumption. The checksum covers every file under `export/`; it excludes the
  root authoritative `run.json` and human-readable README history.
- New staged run IDs include compute, analyse or present, for example
  `exp022-r001-compute`; execution origin belongs in run.json, not the ID.
  Historical v2/v3 evidence remains non-operational. Record stage and input
  references in run.json, never infer provenance from the name.
- Write a run first as `.pingstore/runs/.<run-id>.tmp/`; atomically rename it to
  the visible run ID only after every output, `run.json`, and README are complete.
- Treat visible run directories as immutable. A hidden temporary directory is
  incomplete and must never be used as evidence or publication input.
- Keep complete execution provenance in `run.json`; do not infer provenance
  from filesystem timestamps or the run ID alone.
- Local and HPC execution use this same run-folder contract. Allocate HPC run
  IDs before submission so concurrent jobs cannot claim the same identity.
- Do not reintroduce collection/experiment directory nesting, catalogues,
  lifecycle states, automatic official selections, preview overrides, archive
  bundles, or a general Pingstore management CLI. The narrow discovery and
  hash-bound prune commands above are the only operational CLI exceptions. The historical v2 migration
  utility is not a conforming upgrade path and does not authorize v2 use.

# Experiment execution

Before creating or editing experiment execution code, read and follow
the versioned [Experiment Runner Guide](experiments/README.md).
Compute, analyse and present complete independently;
downstream stages never launch upstream work or automatically publish.

# Experiment writing

Before creating or editing `writings/expXXX.typ`, read and follow
the versioned [Writing Guide](writings/README.md).

The `writings/expXXX.typ` files are collectively know as "writings", or
"the writings".

Do not create automated tests for the writings.

Agents must maintain each article's separate `txt` or `data`,
`vX.Y.Z`, and optional author-assigned `reviewed` tags after
relevant article, implementation, execution, Writing Guide or local-data
changes, including dependent comparisons and syntheses. `X.Y.Z` records the
latest Writing Guide version applied and must remain current under Writing Guide
section 3.4. Only the author may add or remove `reviewed`. Follow that section
for validation and uncertainty handling; metadata-only edits do not change
dates.
