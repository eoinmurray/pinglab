# exp082 — continuous-stream spike-count classification

## Contract migration

Runner Guide 3.0.0 and Storage Guide 3.0.0. The former flat runner has been
split into independent compute, analyse and present stages. The separately
approved historical import and derived rebuild are complete; no production
simulation, training or publication was performed. The article was revised under
Writing Guide 16.0.0, then updated to the plain Results heading in 17.0.0;
it has not been marked Reviewed. See
[MIGRATION_REVIEW.md](MIGRATION_REVIEW.md) for exact runs, bytes and checks.

```sh
uv run python -m experiments.exp082.compute --source <exp022-compute-run-id>
uv run python -m experiments.exp082.analyse --source <exp082-compute-run-id>
uv run python -m experiments.exp082.present --source <exp082-analyse-run-id>
```

Each stage requires an explicit, completed v3 source. Exact payloads, authoritative
manifests and all ancestral pins are validated before use and again before atomic
completion. Stage outputs remain under a hidden reservation until completion.
Failures retain hidden evidence and never mutate a completed run. There is no
latest-input fallback, automatic upstream work, materialization or publication.
The package's bare entry point rejects monolithic execution.

- Compute runs frozen TR-06 models, never training them. Quantitative evidence is
  retained as per-stream/per-presentation E/I/output counts and labels; illustrative
  evidence retains all E/I/output spikes, pixel arrays, conditions and labels.
  Simulator commands, output logs, dataset hashes and code evidence are retained.
  Inference remains a compute operation, including the two illustrative streams.
- Analyse calculates accuracy, silence, output totals, population firing rates,
  cumulative count shares and seed aggregation. It selects the first correct
  presentation in the matched stream. Only derived display arrays and numerical
  results are copied; large raw recordings and the bank remain referenced.
  A wholly silent quantitative readout fails analysis.
  Historical probabilities are softmax count shares, not calibrated posteriors.
- Present reads saved analysis and its explicitly pinned compute ancestry. It
  draws six result figures plus the prospective protocol schematic and emits
  numbers in a flat export. It does not select an illustrative success, recompute
  condition statistics, fetch MNIST, or invoke inference. Missing pixels fail
  validation; historical reconstruction must be a documented import operation.

## Preserved scientific recipe

- Three independently trained seeds (42–44), validation-selected `weights.pth`;
  `weights_final.pth` remains a distinct validated checkpoint, never substituted.
- Eleven input rates: 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15 and 25 Hz;
  durations 25, 50, 100 and 200 ms; timestep 0.1 ms.
- Forty independent five-digit streams per condition, processed in batches of
  five with the corrected time × batch × input tensor arrangement. Hidden state
  continues within each stream; output state/counts reset at digit boundaries.
  The factorial grid has 132 seed conditions and 26,400 decisions.
- The original image sampling RNG and separate input-encoding RNG seeds are
  unchanged. Count ties retain NumPy argmax's first-class rule, including silent
  decisions. E/I rates use the original 1024/256 population denominators.
- The two illustrative five-digit streams retain their exact condition order,
  sampling RNG seeds 82/83 and encoding seeds 83/84. The matched stream is 200 ms,
  5 Hz throughout. Selection is the first success, not the first presentation.
- The psychometric is the 200-ms slice. Accuracy is averaged over three trained
  seeds; error bars are sample SD divided by sqrt(3). The heatmap retains the
  original float32 rounding, separate from the full-precision psychometric.
- Raster selection and the 91.5–94.5 ms transition window are preserved.
  Review corrected the labels to the actual displayed first 200 E and 64 I
  cells; full-population measurements and retained recordings are unchanged.
  Count-share labels no longer imply calibrated probabilities. Rate ticks and
  square image thumbnails were adjusted for legibility, and figure run stamps
  were removed; provenance remains in the authoritative run records.

`PINGLAB_SMOKE=1` preserves the smaller 18-condition grid, one three-digit stream
per cell, and both full illustrative streams. Existing direct-compute pilot
knobs (`PINGLAB_EXP082_STREAMS_PER_CELL`, `PINGLAB_EXP082_DIGITS_PER_STREAM`,
`PINGLAB_EXP082_STREAM_BATCH_SIZE`) are recorded in the recipe and replay script.
Collection profiles are fixed smoke/production and do not inherit pilot knobs.

## Six-shard collection execution

The exp082-specific collection hooks now dispatch a dedicated staged adapter.
The adapter reserves source-neutral IDs through the shared allocator, pins the
bank and dispatches each stage separately. Six ordered round-robin workers retain
the existing 132-job production / 18-job smoke partition. The compute collector
checks all worker records, source code, bank/recipe identities, dataset bytes and
payloads before generating the two illustrative streams and completing compute.
It never reruns missing condition jobs. This preserves the 1,058-launch production
ceiling (1,056 batched quantitative simulations plus two illustrations).

```sh
uv run python -m experiments.exp082.compute --source <bank-id> \
  --run-id <reserved-compute-id> --shard-index <0-to-5>
uv run python -m experiments.exp082.compute --source <bank-id> \
  --run-id <reserved-compute-id> --collect
```

Distributed compute requires committed execution code. Worker completion markers
are reusable only with unchanged code, inputs, recipe and job checksums. Locks
exclude concurrent collectors and duplicate shard writers. Interrupted work
requires explicit recovery or a fresh reservation. The legacy collection repair
integration route rejects exp082; it does not reactivate historical campaigns.

`training_dir(seed)` remains only a relative cell-name projection for registry
callers. It does not select or resolve a training bank. Legacy `RUN_PATHS`,
publication globals, cache execution and `--replot` are retired.

## Historical evidence and approval boundary

See [IMPORT_PLAN.md](IMPORT_PLAN.md) for the audit and approved import selection.
The current validated `exp022-r001-compute` bank contains byte-identical best and
final TR-06 weights. Its three cells are inherited from the base production bank;
configuration/history changes are campaign annotations, not numerical training
changes. The two banks are not generally interchangeable.

Gold-2 has aggregate quantitative condition records, not the new per-presentation
count exports. The retained compute run admits that representation only under its
explicit historical-import contract and preserves the 199 approved source files
with exact mappings, and verifies live metadata before and after import.
It validates both original checkpoint roles against the operational bank and
reconstructs only the ten illustrative pixel arrays. Exact input re-encoding and
full E/I/output raster equivalence are required before atomic completion.
The one-off importer was retired; its executed code remains in run provenance.

The completed chain is `exp022-r001-compute` → `exp082-r001-compute`
(historical import) → `exp082-r002-analyse` → `exp082-r005-present`.
Earlier presentations r003/r004 remain immutable; r005 includes all visual
corrections. These are local, unmaterialized runs, not published selections.

## Verification and outstanding work

Dedicated synthetic tests cover stage isolation, raw count shapes, corrected
batching and partial batches, explicit checkpoint roles, missing/invalid inputs,
v2 and symlink rejection, ancestry changes, atomic failures, six-shard reuse,
dirty-code rejection, collector locking, staged collection dispatch, preserved
SEM and all seven figure outputs. They run in temporary stores. A pre-existing
exp022 scaffold test now patches the function's actual module globals, keeping
that check inside its temporary fixture after exp022's earlier package split.

After explicit ownership approval, the exp082-only shared-test expectations were
updated for staged dispatch, no legacy-path creation on import, no legacy repair
integration, and exclusion from legacy campaign recapture. The seven former
failures are resolved; no tests were skipped to make the suite pass.
The final combined regression passed **461 tests**: dedicated exp082,
checkpoint-role, collection, multiseed, Pingstore, writing inputs/status/contents,
and simulator model tests. Historical-import tests cover explicit provenance,
aggregate consistency, live-metadata rejection, independent analysis and atomic
failure. Visual regressions check raster labels, equal undistorted thumbnails,
non-overlapping rate ticks, actual simulator entry-point existence, and both
HTML and paged article rendering with correct equation indices. An unavailable
input renders the shared notice without invented results.
Ruff for the exp082 package/dedicated tests, the exp082 type check and diff
whitespace checks passed. Other experiments' shared-test expectations were not
changed. No shared-file conflict was observed during these scoped updates.
Read-only discovery validated the local presentations. The article is now
`[▦ DATA]`; no other article declares exp082 as a data input. Browser inspection
confirmed five loaded figures, the explicit selected presentation and lineage,
working contents anchors, and three numbered equations. All five PDF pages and
both additional exported diagnostic figures were inspected. Commit, push,
publication, materialization and Reviewed status remain separately gated.
