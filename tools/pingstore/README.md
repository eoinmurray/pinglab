# Pingstore

Pingstore is Pinglab's single operator interface for collection-scoped scientific
data. A `CollectionDataset` retains experiment-scoped `ExperimentRun` records,
selects one official run per experiment, and may carry temporary preview
overrides. Finalized runs and frozen datasets are immutable.

The canonical layout is:

```text
.pingstore/experiment-runs/<collection>/<experiment>/<run>/
├── run.json
├── payload/                 # immutable complete data
└── authored-sources/        # exact .py and .typ when captured locally
.pingstore/collections/<collection>/collection-dataset.json
.pingstore/frozen/<collection>--<snapshot>/
.artifacts/              # compact materialized PublicationView
r2://pinglab/datasets/<collection>/<snapshot>/
```

A successfully finalized run becomes that experiment's official run. Failed
runs are retained without changing the official selection. `.artifacts/`
is rebuilt from official selections and excludes raw arrays and checkpoints;
those remain in the immutable run and verified dataset archive.

The managed workflow is:

```bash
pingstore migrate inventory
pingstore migrate classify
pingstore migrate plan
pingstore migrate import --shadow
pingstore verify --local
pingstore preview COLLECTION --shadow /absolute/path/to/shadow
pingstore freeze COLLECTION --snapshot SNAPSHOT
pingstore archive-r2 COLLECTION/SNAPSHOT
pingstore inspect-r2 COLLECTION/SNAPSHOT
pingstore restore-r2 COLLECTION/SNAPSHOT /absolute/path/to/new-root
pingstore publication-view --activate
```

Large accepted historical payloads may remain in their existing immutable R2
location and be attached without copying:

```bash
pingstore attach-asset COLLECTION r2://pinglab/campaigns/ARCHIVE
```

An attached asset is retained collection evidence, not an official
experiment-run selection. New archives use the native dataset namespace.

Normal run finalization advances official evidence by contract; arbitrary file
recency never does. Archival, deletion, and publication remain separate gates.
Native archives are portable bundles and are checksum-verified after upload.

Legacy migration first internalizes every referenced payload beneath its native
run record and verifies the copied digest. Historical non-runnable evidence is
retained separately and does not pollute active collection membership.
