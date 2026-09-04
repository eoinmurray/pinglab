# exp080 — empirical input-rate calibration

## Independent execution contract

Experiment Runner Guide 4.3.0 and Storage Guide 4.3.0. The diagnostic decoder study remains
standalone: it uses MNIST, not an exp022 checkpoint bank. The retired combined
entrypoint rejects execution. Importing the package performs no training,
simulation, plotting, source selection or filesystem writes.

```sh
uv run python -m experiments.exp080.compute
uv run python -m experiments.exp080.analyse --source <exp080-compute-run-id>
uv run python -m experiments.exp080.present --source <exp080-analyse-run-id>
```

The equivalent `experiments/exp080/<stage>.py` paths are supported. Each command
creates one completed v4 run and prints its ID. `--run-id` accepts only a fresh,
source-neutral reservation allocated by the shared Pingstore helpers. Execution
origin and scheduler identity belong in the manifest, not the ID.

- **Compute:** directly simulates illustrative features, trains three decoders,
  selects validation checkpoints, then simulates shared held-out features and
  records per-image correctness. Retains the original image and illustrative
  feature arrays, decoder weights, training histories, dataset hashes and the
  complete recipe. It never aggregates the rate-selection decision or plots.
- **Analyse:** validates the explicit compute evidence and produces per-seed,
  mean, minimum and maximum accuracy at each rate and the selected interval.
  It never downloads MNIST, runs a decoder, simulates features or plots. Large
  inputs stay in compute; analysis pins them instead of copying checkpoints.
- **Present:** renders saved training trajectories, the psychometric curve,
  and saved illustrative features. Its flat export contains three figures,
  `numbers.json`, `decision.json` and shared presentation bookkeeping. It pins
  analysis and the same compute ancestor, without recalculating statistics.

Source schema, exact layout, complete payload checksum, manifest hash, stage,
experiment, recipe and ancestry are checked before consumption. All ancestors
are rechecked before completion. Writes stay in a hidden temporary directory
until atomic completion. Failed work remains hidden for inspection and requires
a fresh identity or explicitly authorized recovery; downstream stages never
resume it. V2, missing sources, changed ancestry and implicit latest selection
are rejected. No stage writes `.artifacts`, materializes or publishes.

## Preserved scientific procedure

- Maximum-pixel rates are 0.1, 0.25, 0.5, 1, 2, 5, 10 and 25 Hz; decoder seeds
  are 42, 43 and 44. Features use 200 ms presentations at 0.1 ms timesteps,
  Bernoulli events, decay-then-add AMPA conductance, the exact voltage update
  at the current conductance, and the mean post-update voltage minus rest.
  Conductance, membrane constants, float32 arithmetic and seed hashing are
  unchanged.
- Training uses the first 20,000 official training images, validation the next
  5,000, and evaluation the first 5,000 official test images. Each presentation
  samples a training rate uniformly and independently. The 784–1024–10 ReLU
  decoder uses Adam, learning rate 0.001, batch size 256 and 50 epochs.
- Checkpoint selection uses the first maximum mixed-rate validation accuracy,
  not final-epoch accuracy. All three decoders receive the same held-out feature
  realization per image and rate. The original batch-dependent seed schedule
  remains unchanged. The selected floor is the first tested rate where every
  decoder reaches 50%; no interpolation is introduced. A missing crossing
  remains a censored result. The plotted band is the seed minimum–maximum,
  not a confidence interval or standard deviation.
- `PINGLAB_SMOKE=1` uses 100 training, 50 validation and 50 test images and two
  epochs; rates, seeds, decoder architecture, dynamics and selection stay fixed.
  The profile is captured at compute invocation. Downstream stages use the
  retained recipe rather than their current environment.
- `EXP080_DEVICE` and `PINGLAB_DATA_ROOT` retain their existing meanings and are
  recorded with the command. Device choice defaults to CUDA, then MPS, then CPU.

One correctness fix accompanies the split: selected CPU checkpoint tensors are
cloned. Previously `detach().cpu()` could alias live CPU weights and silently
turn a selected checkpoint into final weights. The historical producer used
CUDA, where the original CPU transfer already copied tensors. No historical
weights are changed and no retraining is claimed. A focused two-update toy
decoder test verifies that the first selected weights survive later updates.

The convenience recipe exports, including `EPOCHS_STANDARD`, remain available
to existing collection checks. Numerical `analyze()` is now pure and does not
write `decision.json`; that output belongs to the analyse stage.

## Collection integration and ownership

The dedicated `collection.py` adapter reserves compute/analyse/present IDs,
dispatches them independently, verifies their complete ancestry, and rejects
legacy campaign rows and interrupted reservations. Synthetic integration tests
exercise it without launching simulation or training.

With Anomancer's approval and exp082/exp054 ownership coordination, registration
was added to `experiments/collections/gamma_gated_sparsity/{plan,execution,slurm}.py`.
It covers staged plans, `stage-refs.json` completion, adapter dispatch and Slurm
reservation handling. Existing monolithic plans are rejected. No simulator or
workload-sharding change was needed: exp080 remains a single compute job.
Exp082's concurrent hunks were preserved and committed separately by its owner.
Five shared collection fixtures were updated with separate approval: the
staged-mode allowlist, three adapter mock lists, and legacy-output rejection.
All 27 collection tests pass with exp080 registered.

The calibrated interval is reflected in exp022's TR-06 recipe and then exp082.
Neither directly reads exp080's outputs, and neither is an operational upstream
input of exp080. Other articles currently declare no exp080 data input.

## Gold-2 audit and approved selective import

Contract migration initially stopped before import. Anomancer subsequently
approved the selection below; import and independent downstream execution
completed on 2026-08-28 using the validated shared ID allocator.

The 2026-08-28 audit found 17 experiment-associated Gold-2 files totaling
9,927,272 bytes: 14 scientific/presentation files (9,920,527 bytes) and three
dedicated provenance records (6,745 bytes). All matched cached inventory hashes.
The three validation-selected decoder files total 9,775,575 bytes and their
recorded best epochs are 46, 42 and 44. Retained correctness has shape
`(8 rates, 3 seeds, 5000 images)` and exactly reproduces every decision field:
the selected interval is 0.5–25 Hz, with mean floor accuracy 0.6286.

The original producer is `ggs-production-20260818-4ad223d3`, commit
`4ad223d32620dd9f03698b89f28aedfe944d43ac`, Slurm job 33913460 on `gpu-q-63`
(A100). Original execution, recorded dataset hashes, training histories and
checkpoint hashes agree with the base campaign metadata and logs. Gold-2's
flattening operation is distinct from this scientific producer.

Fresh live R2 `run.json` and `inventory.json` copies matched the cached archive
byte for byte on 2026-08-28. The read-only planner verifies these copies, selected
source hashes, historical lineage, checkpoint roles, dimensions, dataset hashes
and exact numerical replay:

```sh
uv run python -m experiments.exp080.historical \
  --archive .r2/gold-2 --live-metadata <fresh-r2-metadata-directory> \
  --plan <new-plan.json>
```

The approved selection retained 19 source files totaling **11,984,329 bytes**,
including full shared archive metadata. Verified retained source bytes are
**10,266,237**, of which **9,825,407** are scientific exports:

- All three decoder checkpoints and complete histories, the complete held-out
  correctness/labels/rates/seeds archive, and the historical feature PNG.
- Original numbers, decision, reproducer, base campaign manifest and plan,
  completion status, Slurm logs, lineage and Gold-2 root manifest.
- The complete original inventory, losslessly gzipped from 2,101,788 to 383,696
  bytes. No trials, seeds, rates, checkpoints or inventory entries are dropped.

Excluded only the two old SVGs and `_run.txt` (69,308 bytes) from this experiment;
they remain unchanged in Gold-2. The SVGs can be independently redrawn from saved
analysis. The illustrative PNG has no retained raw feature arrays and must be
carried unchanged with explicit lineage, never described as newly simulated.
No upstream bank is copied or referenced. Raw MNIST bytes and ephemeral training
features were not retained in the archive; source dataset hashes remain evidence.

These counts exclude new evidence/recipe records, mapping, command and code
provenance, and subsequent analysis/presentation runs. The selected-source saving
is 1,718,092 bytes through lossless inventory compression; it is not a claim that
the complete run is smaller than the original 9.93 MB study payload.

### Completed operational lineage

| Run | Execution | Complete bytes | Export bytes |
| --- | --- | ---: | ---: |
| `exp080-r001-compute` | Local historical import; no training or simulation | 11,262,238 | 9,847,370 |
| `exp080-r002-analyse` | New analysis of imported correctness | 880,571 | 26,634 |
| `exp080-r003-present` | New plots from analysis; original PNG retained | 969,903 | 112,115 |

Analysis pins compute; presentation pins analysis and the same compute ancestor.
There are no upstream banks. All 19 source files round-trip byte for byte, including
the compressed inventory; original numerical fields and the illustration match
the archive exactly. The complete three-run chain occupies **13,112,712 bytes**,
including code/execution provenance. The original HPC producer, source hashes,
commands, logs, approved mapping and live R2 verification are retained separately
from the local import execution. No original run or archive file was rewritten.

The approved one-off importer was retired after completion. Its executed code,
plan and live-metadata verification remain in immutable run provenance.

## Verification and remaining gates

Dedicated tests cover numerical selection and censoring, stage isolation,
retained profiles, source and ancestry mutation, v2 rejection, malformed evidence,
atomic completion, failed-stage hiding, reservation reuse, adapter dispatch,
CPU checkpoint selection, historical figure carry-forward, read-only import
planning, approved import and copy failure. They use synthetic stage fixtures
and tiny unit computations, not production training or simulation.

The article follows Writing Guide 17.0.0: plain Results with three figures and
concise captions, five substantive Methods operations, preserved equations and
useful shot-noise interpretation in an appendix. Creation date remains 2026-08-10;
substantive revision is dated 2026-08-28. Read-only discovery validates `[▦ DATA]`.
The result calibrates the decoder representation, not PING accuracy, gamma timing
benefits or performance between tested rates. No substantive discrepancy was
found between retained numerical results and the scientific recipe.

Browser HTML and four-page PDF previews use the current article and explicit
validated presentation in an isolated rendering fixture, without materializing.
All figures load; rate labels, accuracy ranges, equation notation/numbering and
contents links were checked. Regression tests cover crossed/censored smoke
results, missing inputs and corrupt selected data. They protect against a literal
conditional branch, duplicated References, function arguments inside subscripts
and absent HTML equation numbers, all found during render review.

After separate approval, `writings/templates/dataset.typ` recognises both `import` and
`historical-import`. The exp080 compute row now shows `1 s (import)`, with a tooltip
excluding original training or simulation. Both operation values are covered by
the existing duration tests, including retained HPC timing. Earlier shared
duration/projection work was preserved. Anomancer explicitly approved committing
the migration without the shared dataset component and `test_run_view.py`: their import-label
fix remains uncommitted because it depends on the other task's uncommitted
duration feature. The browser label and tooltip were checked in the shared
working checkout, not established by this migration commit alone.

No production computation, Slurm dispatch, materialization, publication, Reviewed
marking or push has been performed by this task.

Final checks on 2026-08-28: 40 dedicated exp080 tests, 65 run-view tests, all 136
Pingstore tests and 27 collection tests passed; scoped lint and whitespace checks
passed. Exp054 subsequently began its separately approved registration/fixture
work, preserving the exp080 additions; collection files must be rechecked before
any commit. The isolated exp080-only commit candidate passed 173 tests across
the dedicated exp080, collection and committed Pingstore suites, without any
other task's uncommitted changes. No remaining exp080 test failure is known.
All three exp080 manifests/payloads and the 19 selected archive sources revalidated
unchanged. Of 73 pre-existing manifests, 67 remain unchanged; exp083's three runs
were separately deleted with Anomancer's authorization and exp084's three runs
were separately moved to its owner's documented recovery location. Neither change
was performed by this task, and no missing run was recreated.

## Future-run data retention — 2026-08-28

Native compute keeps each validation-selected decoder state in memory through held-out evaluation and does not write decoder.pt. Training histories and held-out correctness remain retained; memory_only is explicit in the training record. The low-level helper still supports checkpoint files for explicit callers. Older v3 evidence remains non-operational historical evidence.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
