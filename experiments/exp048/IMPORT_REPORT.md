# exp048 historical import — 2026-08-28

## Scope and status

Anomancer approved importing the compact historical summary and rebuilding its
derived outputs, explicitly outside Gold-2. exp048 remains deprecated in favor of
exp082 and excluded from the collection execution graph. This is not a new
simulation or training run, nor a reconstruction of missing raw recordings.

Code contracts: independent native compute/analyse/present stages implemented and
tested. Data: historical import, seed-summary reaggregation and presentation
complete. Writing: `DEPRECATED` title, `Results available`, not author-reviewed.
Creation date remains 2026-06-08; substantive revisions dated 2026-08-28.

## Selected source and exclusions

Approved source:

```
r2:pinglab/datasets/gamma-gated-sparsity/baseline-20260826/experiment-runs/exp048/exp048--r003/
```

All **15 objects / 738,066 bytes** were downloaded and verified against the pinned
live metadata: 13 payload files / 734,494 bytes, plus the archive run record and
inventory. The source contains numbers, eight figure files, a manifest, patch,
run text and shell script. All original bytes are retained under
`provenance/archive/`; nothing was subsampled or lossily repacked.

Metadata SHA-256:

| Record | SHA-256 |
| --- | --- |
| run.json | `2e1c30d9466d455328e4941471001c9dda316b53dc39559379de68443638ca4d` |
| inventory.json | `0488f5f7e156479ee04f0488650095cdf3bdd95cf88428daaa97bacd87cc11b4` |

Every payload size and SHA-256 matches that inventory. These live records matched
the cached audit evidence. Gold-2's separately verified inventory has no exp048
entry. The historical manifest is dated July 24; August 26 is the later archive
inventory date, not a new scientific execution date.

The prior search covered 10,036 R2 objects, four archive members and 64,735 HPC
files outside dependency directories. Fifteen identical HPC copies and one copy
with alternate rendering add no independent numerical evidence and were excluded.
No raw streams were found. No upstream bank was copied or referenced: compatibility
with a current Gold-2 bank does not establish historical checkpoint identity.
exp092 and exp109 consume the presentation interface; all existing figure names
and numerical keys remain available. exp082 is a successor, not a data dependency.

## Completed local runs

IDs were allocated through the shared validated stage helpers, not guessed.
All five directories validate as completed v3 runs with payload and manifest pins.
All have `historical.gold_2: false` in authoritative provenance.

| Run | Operation | Files | Total bytes | Export bytes |
| --- | --- | ---: | ---: | ---: |
| exp048-r001-analyse | Historical import | 22 | 1,091,748 | 47,757 |
| exp048-r002-analyse | Reaggregate saved seed summaries | 7 | 355,260 | 51,199 |
| exp048-r003-present | Initial rendering | 15 | 1,043,269 | 736,076 |
| exp048-r004-present | Intermediate legend placement | 15 | 1,064,193 | 733,878 |
| exp048-r005-present | Final checked presentation | 15 | 949,420 | 734,885 |

The final dependency chain is **r001-analyse → r002-analyse → r005-present**.
r003 and r004 remain immutable evidence of presentation iterations. r003's chance
line crossed legend text; r004's moved legend overlapped the inset label. r005
uses an opaque legend background, verified in scratch before run creation.

Final-chain storage is **2,396,428 bytes**. All five retained runs occupy
**4,503,890 bytes** (regular-file byte totals, including authoritative manifests).
This is larger than the 738,066-byte source because it retains originals, new
derived outputs and execution provenance; no compression saving is claimed.
The five shared-helper source patches alone total 1,421,169 bytes. They capture
the contemporaneous dirty shared checkout, including unrelated changes as
provenance; they do not confer ownership of those changes. No duplicate model bank
or additional HPC copy was imported.

Payload digests for the final chain:

| Run | Payload digest |
| --- | --- |
| exp048-r001-analyse | `sha256:d644b05f83e03eb5a0b875fd6b7c79ccb13b8b80f1fd583fb2f0ac6902865bc8` |
| exp048-r002-analyse | `sha256:c07a98e7399b20c4778586cc74244a27a6dd054358c589cfa614fb19cd65520d` |
| exp048-r005-present | `sha256:515e4d8b6ead51f7aab44468ea774f0250042152244da437ee9d0d22b96eb06b` |

## Numerical and visual verification

- Retained 195 per-seed rows: 24 duration, 144 duration/rate grid, 27 low-rate.
- Recomputed every aggregate with original float32 duration/grid and float64
  low-rate arithmetic, including sample SEM. All historical numerical fields
  match at relative/absolute tolerance 1e-12.
- Re-rendered four quantitative figure files from explicit saved analysis.
  Carried four historical raster figure files byte-for-byte, with individual
  source hashes and carry-forward provenance. No raw raster was reconstructed.
- Final browser HTML: two loaded images, two figures, eight rendered equations,
  no horizontal overflow. All five PDF pages were visually inspected. The local
  isolated review bundle uses the validated r005 input; no shared article binding
  or `.artifacts/` materialization was changed.
- Corrected caption inset range (0–10 Hz), shaded SEM band terminology, legend
  overlap, startup-window equation and unsupported universal duration cutoff.
  Regression checks cover the range, band, legend and removed cutoff claim.

Final verification command (131 passing tests):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q experiments/tests/test_exp048_stages.py tools/pingstore/tests experiments/tests/test_gamma_gated_sparsity_collection.py::test_graph_orders_dependencies_and_replaces_exp048_with_exp082 experiments/tests/test_gamma_gated_sparsity_collection.py::test_plan_paths_are_isolated_and_all_runners_are_integrated
.venv/bin/ruff check experiments/exp048 experiments/tests/test_exp048_stages.py
.venv/bin/ty check experiments/exp048
git diff --check
```

Tests cover stage isolation, v3 rejection, checksum and semantic corruption,
checkpoint roles, exact input reconstruction, complete ancestry before/after
work, atomic failure, reservation retry, historical identity and non-Gold-2 flags.
Stage tests use a synthetic simulator; the real parser is exercised without
running inference. The complete repository test suite was not run.

## Provenance limits and outstanding science

The original producer is recorded as **local**, not HPC. Its source revision,
dirty patch, command evidence (including an empty command record), absent
completion timestamp and recorded duration are retained without invention.
Archive identity r003 and numerical identity r001 disagree; the fixed raster
is stamped r002 and the varying raster `exp048-replot`. The low-rate extension
attributes itself to "exp065 initial computation". These are unresolved, not
rewritten as consistent new history.

Exact checkpoint and simulator lineage, raw prediction replay and raster
regeneration remain unavailable. No operational compute run or bank pin was
fabricated. The approximate 28 ms gamma marker lacks an independent retained
frequency measurement; a historical 9 ms GABA comment differs from the current
compatible bank's 6 ms setting. Neither establishes which bank produced these
results.

The decoder knows segment durations and endpoints. exp048 now states that;
cross-article claims in exp092/exp109 about absent segmentation cues remain for
separate review. Raster endpoint labels denote full populations although only
sampled ranks are plotted; the caption explains this without altering source
images. The 10 ms grid spans about 17–63% accuracy, so the abstract's former
universal failure-floor claim was replaced with rate-dependent degradation.

No shared collection/simulator code or shared tests were edited, and no overlap
conflict was encountered. Other tasks' working-tree edits remain untouched.
No production inference/training, archive mutation, materialization, publication,
staging, commit or push was performed by this task. Commit and author-review
approval remain outstanding.
