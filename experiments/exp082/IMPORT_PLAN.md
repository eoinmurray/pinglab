# Exp082 selective historical import — approved 2026-08-28

Audit date: 2026-08-28. Anomancer approved this selection and the independent
analysis/presentation rebuild. Publication, materialization, Reviewed status and
commit remain separately gated. The source archive and completed runs stay immutable.

## Sources and producer

Gold-2: `r2:pinglab/campaigns/gold-2`, cached at `.r2/gold-2`.
The corrected producer is `ggs-exp082-repair-20260820-73f0883e`, commit
`73f0883edc14aa634f5a6d55e4f4123fbfeb7508`, reusing the base campaign's
TR-06 weights. Six inference shards ran under Slurm array 34021105; aggregation
34021106 produced the retained 132-row result. The repair concatenated individual
streams along the batch axis. It was not retraining.

The original presentation manifest says `host=local`, while logs establish HPC
execution. Preserve that historical discrepancy verbatim. A new import must
record its actual local operation separately from the original scientific
producer, repair identity, checkpoint hashes and integration lineage.

Fresh `rclone cat` reads, with one retry, matched cached metadata byte for byte:

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| `run.json` | 1,103 | `d4d067148589ae8469a37ee765c77282c9ab081eff22dfda0a24d596cbba913c` |
| `inventory.json` | 2,101,788 | `c7b9455968e34ac3be2df46a57ab4fc0ffcd94dc88799bf11b36c9b673a88f68` |
| `lineage.json` | 1,876 | `d3e128a730ec4f85011e73cafdb0119683b4af25be7daced6d5eace550658e97` |

All 190 exp082-related cached files (3,213,792 bytes) passed the archive
inventory's size and SHA-256 checks. All 132 condition records exactly equal
their historical numerical-summary rows. The separate shared base bank is
1,133 files / 2,258,472,251 bytes; its three TR-06 cells occupy 33 files /
60,813,302 bytes. Neither bank nor checkpoints are proposed for copying.

## Proposed inclusions

- All 132 quantitative condition records, all six shard records, both full
  illustrative measurement streams and their original count-share arrays.
- Both encoded illustrative inputs, simulator configurations, commands and
  output/event logs. These provide independent replay checks.
- Historical numbers and all exp082-specific producer/repair evidence,
  including failed base-attempt logs that explain why repair was required.
- The base campaign manifest, base run and collection-plan records; original
  config, metrics and attempt records for each of the three TR-06 cells; archive
  run, inventory and lineage metadata. Originals belong in provenance.
- Ten explicitly reconstructed digit images, with dataset, selection and
  reconstruction hashes; do not run a dataset fallback during presentation.
- An operational input pin to `exp022-r001-compute`. Both checkpoint roles
  match byte for byte. Retain original base-bank references as historical
  provenance without pretending the repaired campaign trained these cells.

The selection is **199 source files / 6,079,619 bytes**, including archive-wide
metadata and required upstream execution records. This is 197 inventoried files
plus the root run and inventory files. Estimated complete compute retention is
**6.2–6.6 MB**, including reconstructed pixels, import mappings and code evidence;
analysis/presentation add additional files, provisionally **7–9 MB total** across
the three runs. These are estimates, not observed imported sizes.

## Proposed exclusions

Six exp082 files total **672,858 bytes**:

- Four old rendered figures: duration/rate summary, matched stream, variable
  stream and psychometric. They remain unchanged in Gold-2 and are used for
  comparison, not imported as freshly generated evidence.
- Two redundant `rasters.npz` files. Their complete E/I/output spike arrays
  exactly match the arrays already retained in the compact measurements file.

No trials, conditions, seeds, neurons or measurement-array values are dropped.
The complete selected bank is referenced rather than duplicated. This does not
make the standalone exp082 run smaller than its original 3.2-MB directory total:
retained upstream provenance and the 2.1-MB archive inventory add overhead.

## Reconstruction and numerical validation

The cached official MNIST test gzip files match torchvision's published resource
MD5s (`9fb629c4189551a2d022fa330f9573f3` for images and
`ec29112dd5afa0611ce80d1b7f02629c` for labels). The decoded test source SHA-256s are:

- Images: `0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7`.
- Labels: `ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2`.

Selections with RNG seeds 82/83 reproduce the archived labels. Re-encoding
those images with the frozen recipe and seeds 83/84 exactly reproduces both
archived input-spike tensors. This was input reconstruction, not neural
simulation. The two float32 pixel matrices each occupy 15,680 bytes:

- Matched: `a989efa32e44e736f3eef509b39a3d5a7382b583737324cd6422906bd24dabeb`.
- Variable: `ba99310a2d398667d011791fb861cbfc7e97d0dc82b6e03f39705b939f515104`.

Replayed output count shares, predictions and activity summaries exactly match
both historical streams. The first successful matched presentation is index 1,
the digit 4; the variable stream retains three correct decisions out of five.
The complete grid has 26,400 decisions, 4,253 silent windows and 2,267,105 output
spikes. Mean accuracy across rates/seeds is 36.4697%, 49.0455%, 60.3788% and
69.0909% at 25, 50, 100 and 200 ms, respectively.

The historical quantitative evidence does not contain individual labels/counts
for every decision. Preserve its aggregate representation explicitly; it cannot
support invented per-decision results or new uncertainty estimates. The later
importer must admit this evidence only as an explicitly identified historical
import and retain original producer records.

Before import, recheck live metadata and all selected hashes, validate the bank
and its pins, and coordinate shared-store writes. Allocate through existing
validated helpers, never guess an ID. Validate before atomic completion and
again recheck source immutability. Analyse and present independently run against
explicit new inputs; compare all retained scientific results with historical
evidence. Report actual bytes, hashes, output differences and limitations.
Do not materialize, publish, modify Gold-2, mark Reviewed or commit without the
separate requested approvals.
