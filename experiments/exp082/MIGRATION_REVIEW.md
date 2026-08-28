# Exp082 migration review — 2026-08-28

## Outcome and boundaries

Independent compute/analyse/present contracts, the approved historical import,
derived analysis, presentation and article revision are complete. No new study
simulation, training, production HPC job, publication, materialization, commit or
push was performed. Synthetic simulator tests are not production experiments.
The article is data-backed, not marked Reviewed. All runs and Gold-2 remain
immutable. The staged Git index was empty at handoff.

## Retained runs and byte accounting

Sizes include every regular file, including root `run.json` and provenance;
they exclude the separately referenced training bank and scratch review fixtures.

| Run | Meaning | Files | Bytes |
| --- | --- | ---: | ---: |
| exp082-r001-compute | Local historical import of corrected HPC inference | 212 | 6,514,410 |
| exp082-r002-analyse | Newly executed analysis of retained evidence | 7 | 430,885 |
| exp082-r003-present | First newly rendered presentation | 14 | 948,277 |
| exp082-r004-present | Intermediate visual corrections | 14 | 1,293,942 |
| exp082-r005-present | Final reviewed rendering | 14 | 1,406,887 |

The final compute→analyse→present chain occupies **8,352,182 bytes**; all five
immutable runs occupy **10,594,401 bytes**. Earlier presentations were not
overwritten or deleted. Their figures contain no changed scientific values.
Source-patch sizes differ because each execution captures the shared checkout's
actual state, including concurrent unrelated changes; those patches do not
constitute ownership or commit inclusion.

The approved selection was **199 source files / 6,079,619 bytes**. Six excluded
redundant files total **672,858 bytes** (four old figures and two rasters whose
full spike arrays exactly match retained measurements). Nothing was deleted
from Gold-2. No trial, condition, seed, neuron or measurement-array value was
subsampled. Referencing the existing bank avoids duplicating the three required
TR-06 cells (60,807,975 bytes in the repaired bank; 60,813,302 bytes in the
original base bank), not a newly achieved deletion of the whole 2.26-GB bank.
The standalone exp082 import is larger than its original 3.2-MB footprint
because it adds upstream records, archive-wide inventory and code provenance.

Final-chain payload digests:

- Compute: `sha256:92726f70beccb50a5417fb21a9887f12a2bfcfeda069f41cf79f188c8bf7055e`.
- Analyse: `sha256:905ddfe4232909ef7f301161cf1d22a637f3f3d5c042e9d0bb282b61e49b7849`.
- Present r005: `sha256:36c0db53c9319dfd4f76a7715c61574f8c53ee746a01418dc947f1b101bad5e6`.

## Lineage and limitations

The input is validated `exp022-r001-compute`. Both best-validation and final-epoch
TR-06 checkpoints match the original base bank exactly for seeds 42–44;
configuration/history differences are campaign annotations only.

The scientific producer remains `ggs-exp082-repair-20260820-73f0883e`, commit
`73f0883edc14aa634f5a6d55e4f4123fbfeb7508`, Slurm array 34021105 and aggregation
34021106 on gpu-q-26. It reused training from
`ggs-production-20260818-4ad223d3`; the repair fixed tensor concatenation along
the batch axis, not training. Original commands, failed base attempts, repair
records, cell records and lineage are retained verbatim. The old presentation
manifest's `host=local` conflicts with HPC logs; `run.json.historical_import`
records that discrepancy separately from the actual local import operation.

The grid contains original aggregate records, not individual decision labels or
counts. New trial-level uncertainty estimates cannot be reconstructed. Both
illustrative streams retain all E/I/output spikes and original softmax arrays.
Ten image arrays were reconstructed from hash-verified official MNIST test
files; their labels and re-encoded spike inputs match the archive exactly.
These are documented reconstructions, not newly simulated measurements.

## Numerical and scientific review

- All 132 condition rows and the 200-ms slice equal the historical summary
  exactly. Recomputed illustrative predictions, correctness, output-activity
  summaries and float32 softmax arrays also match exactly.
- Grid: 26,400 decisions; 4,253 silent windows; 2,267,105 output spikes.
- Mean accuracy across rates/seeds: 36.4697%, 49.0455%, 60.3788%, 69.0909%
  at 25, 50, 100 and 200 ms. At 200 ms, 10/15/25-Hz means span 83.1667–88.6667%.
- Matched labels `[8,4,3,1,9]`, predictions `[5,4,3,1,9]`; the explanatory
  success remains segment 1, digit 4. Variable labels `[9,1,6,7,0]`, predictions
  `[0,1,5,7,0]`; three of five are correct, including the retained silent failure.
- The former article incorrectly said accuracy selected checkpoints. Original
  metrics and training code specify minimum validation cross-entropy over three
  encoding draws, with accuracy breaking ties. Only the account was corrected;
  checkpoint roles, epochs and hashes did not change.
- No causal role for gamma, fixed-rate-training advantage or decoder superiority
  is established. Presentation and integration duration remain confounded.
  The selected success and five-digit stress test are illustrative, not accuracy
  estimates. These limits are retained in captions and Methods.

## Writing and visual checks

Creation date remains 2026-08-10; substantive revision date is 2026-08-28.
Results now contain figures and concise captions only, before flat numbered
Methods; useful former narrative/conclusion content was integrated without a
Discussion section. Three numbered measurement equations are preserved or made
explicit. Corrected Typst spacing keeps function arguments outside subscripts
and superscripts; a local HTML rule renders equation numbers in both targets.

The browser and PDF skills guided actual rendered inspection, not just source
checks. The isolated native-presentation fixture selected r005 explicitly and
displayed the correct compute/analyse ancestry. Five article images loaded,
contents anchors resolved and equations (1)–(3) rendered correctly. All five
A4 PDF pages were visually inspected. The two extra exports (matched-stream
raster and 91.5–94.5-ms transition detail) were also inspected. Corrections include
honest 200/64-cell raster labels, non-overlapping compact rate ticks, square
thumbnails with unobscured labels, and softmax-share terminology.

Review files are under `.r2/exp082-review-jk5IS2/native/`, outside publication.
No shared default, generated shared catalogue or `.artifacts` directory was
changed. The fixture copies only validated present exports for isolated rendering.
Typst emitted its standard experimental-HTML and ignored-page-setting warnings.

## Verification and concurrency

Final regression: **461 passed in 17.58 seconds**:

```sh
uv run --no-sync pytest -q \
  experiments/tests/test_exp082_stages.py experiments/tests/test_exp082_scaffold.py \
  experiments/tests/test_checkpoint_roles.py \
  experiments/tests/test_gamma_gated_sparsity_collection.py \
  experiments/tests/test_multiseed_inference.py tools/pingstore/tests \
  experiments/tests/test_writing_inputs.py experiments/tests/test_writing_status.py \
  experiments/tests/test_writing_contents.py tools/snnsim/tests/test_models.py
```

Ruff, package type checking and Git whitespace checks passed. Live R2 run,
inventory and lineage metadata matched their cached hashes before and after
import. Selected source hashes, all operational checksums and ancestral pins
were revalidated after rebuilding. Read-only discovery establishes `[▦ DATA]`;
only exp082 declares exp082 as an article data input.

Shared edits are the previously approved exp082 hooks in collection execution,
plan, Slurm and workloads, plus exp082 expectations in the collection and
multiseed tests. They were unchanged between import and final ownership checks.
Unrelated concurrent edits appeared in exp054, exp080, exp083–086, their tests
and `uv.lock`; they were preserved and are not part of this task's commit scope.
No overlapping edit conflict was detected in this task's files.

Not performed: the entire repository test suite, real distributed/HPC execution,
new production simulation/training, a new independent stream-bank replication,
publication or materialization, Reviewed approval, commit or push. Committing
still requires explicit approval and exclusive staging coordination. Recheck
ownership and every outgoing commit before any later commit/push.

### Review-gate refresh after concurrent changes — 2026-08-28

The earlier 461-test result records the implementation handoff, not the latest
shared-checkout state. A subsequent focused integration run returned **212 passed,
5 failed**. All five failures are in `test_gamma_gated_sparsity_collection.py`:
the allowed runner-kind set and adapter mocks omit newly staged exp080, while
the compose test still expects legacy exp080 acceptance. The affected tests are
`test_plan_paths_are_isolated_and_all_runners_are_integrated`,
`test_local_resume_runs_in_dependency_order`,
`test_compose_campaign_replaces_selected_outputs_and_records_sources`,
`test_publication_build_runs_promotion_from_separate_checkout`, and
`test_publication_build_rejects_stubbed_entries`. The exp080 owner was notified;
no shared implementation or test was edited during this refresh. Their test
changes remain subject to their own ownership/approval coordination.

A separate fresh run of the two exp082 test modules passed **49 tests**. Ruff,
package type checking and Git whitespace checks passed. Recursive validation of
r005, r002, r001 and the pinned exp022 bank passed, including historical evidence
validation; the recorded payload and run-record hashes are unchanged. Read-only
discovery still validates all three exp082 presentations, supporting `[▦ DATA]`.

The separately authorized Writing Guide 17 migration changed the article's
heading to plain `Results` and updated its dedicated rendering assertion. Both
changes were preserved without editing those files. This is mixed ownership in
`writings/exp082.typ` and `experiments/tests/test_exp082_stages.py`, in addition to
shared collection files; the writing task was notified that whole-file staging
requires coordination. Current HTML/PDF rendering assertions pass, but the prior
manual browser and paged inspection predates this heading-only change and the
concurrent shared run-view changes; manual visual inspection was not repeated.

Only this handoff note was edited during the refresh. No operational run was
created or changed, and nothing was staged, committed, published or materialized.
Next gate: review the migration and resolve shared-test ownership/failures before
explicit commit approval and exclusive staging coordination.

### Authorized commit preparation — 2026-08-28

Anomancer explicitly authorized committing and pushing exp082. All active
checkout owners acknowledged the exclusive Git-write window. The writing task
transferred only its exp082 Results-heading and corresponding test-assertion
hunks; other guide-wide edits are excluded. Exp080 registration hunks were
separated in the index without changing shared working files.

An isolated archive of the exact staged code passed the same **461-test suite
in 15.90 seconds**, including all collection checks. The first archive test
attempt lacked the generated `.demolab/lib.typ` dependency (44 metadata-test
failures); supplying that library resolved them without changing tracked code.
The five exp080-related shared-checkout failures are absent from this candidate;
their remaining integration work belongs to exp080. No operational scientific
data was copied into the test snapshot or added to Git. Documentation-only
updates record this verification; publication and Reviewed status remain gated.
