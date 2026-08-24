# Pingstore

Pingstore is Pinglab's single operator interface for collection-scoped scientific
data. A `CollectionDataset` retains experiment-scoped `ExperimentRun` records,
selects one official run per experiment, and may carry temporary preview
overrides. Finalized runs and frozen datasets are immutable.

The migration interface is deliberately shadow-first:

```bash
pingstore migrate inventory
pingstore migrate classify
pingstore migrate plan
pingstore migrate import --shadow
pingstore verify --local
pingstore preview COLLECTION --shadow /absolute/path/to/shadow
pingstore freeze COLLECTION --snapshot SNAPSHOT
pingstore archive COLLECTION/SNAPSHOT /absolute/path/to/archive
pingstore restore /absolute/path/to/archive /absolute/path/to/new-root --native
```

The implementation never treats newest data as accepted evidence. R2 writes,
active-view cutover, publication, and deletion remain separate authority gates.
Legacy Runstore archives stay byte-identical and remain verifiable and
restorable through `pingstore verify --r2` and `pingstore restore`.
Native archives are portable local bundles. Uploading them to the new R2
namespace is intentionally unavailable until remote-write authority is given.

After a shadow rehearsal has been reviewed, the same idempotent import may
install local working metadata with `pingstore migrate import --local`. This
does not select official evidence or alter `artifacts/data/`.
