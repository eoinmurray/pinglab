# Pingstore

Pingstore is Pinglab's filesystem convention for immutable scientific runs.
It has no database, server, catalogue, or command-line interface.

```text
.pingstore/
├── runs/
│   ├── exp097-r022-local/
│   │   ├── run.json
│   │   └── files/
│   └── .exp097-r023-slurm-wilkes-48291.tmp/
└── collections.json
```

Run IDs encode the experiment, experiment-local identity, and execution source.
Local and HPC runners write the same structure. A hidden `.tmp` directory is an
incomplete run; successful completion atomically renames it to the visible,
immutable run ID.

`run.json` records the structured execution and source provenance. The `files/`
directory contains the complete result payload. `collections.json` is a
manually maintained mapping from named views to run-ID arrays; it is the only
selection mechanism.

R2 backup mirrors `runs/` directly. Pruning is an external filesystem operation:
retain the newest useful runs, every run referenced by `collections.json`, every
run not yet verified remotely, and anything else selected in a manual view.
