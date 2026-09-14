# Exp110 grounding review — 14 September 2026

The manuscript's principal accuracy, loop-insertion and spike-perturbation
numbers already reflected the completed COBA damping replacements. This pass
verified those results against the completed bank, compute outputs and current
implementation, then corrected remaining numerical and provenance wording.
Writing Guide **36.0.0** applies; the article retains its `data` tag and
14 September update date. No author-review tag was added.

## Corrections applied

- At the 10-Hz training ceiling, the COBA mean is **8.970000 Hz**, which rounds
  to **9.0 Hz**; PING is **9.083757 Hz**, or **9.1 Hz**. The previous statement
  that both averaged approximately 9.1 Hz was stale. At tighter ceilings PING
  also fires faster, so the revised interpretation identifies those comparisons
  as not rate-matched.
- Methods now distinguish the **18 retrained manuscript COBA classifiers**
  from reused PING training, and identify the repeated evaluations underlying
  Figures 3, 4 and 7. The old blanket statement about reusing the other spiking
  measurements no longer obscures those reruns. Damping is explicitly a
  backward derivative change, not a change to the forward update rule.
- The full inhibitory-timescale sweep is described as **population-cycle
  participation**. Its slowest mean spectral peak is **11.748626 Hz**, below
  conventional gamma frequencies; see [Buzsáki and Wang (2012)](https://www.cns.nyu.edu/wanglab/publications/pdf/buzsaki_ARN2012.pdf).
  The manuscript and its synthesis figures now use `f_peak` for spectral peaks,
  and Figure 2I's axis says “frequency”. The fitted rate–frequency slope uses
  the Writing Guide's `beta_rf` notation. Neither measurements nor estimators
  changed.
- The twelve excluded cycle-analysis presentations are now identified as
  having **zero inhibitory spikes despite nonzero excitatory activity**.
  Their cause remains unresolved; the editorial note now asks about failed
  recruitment rather than treating burst detection as an unknown explanation.
- Rendered word counts were recalculated with an explicit token rule. These
  totals are not an estimate of how much prose this editing pass added.

## Replacement training and implementation

The operational bank is `exp022-r015-compute`, imported from the completed HPC
bank with unchanged scientific exports. All **21** COBA replacements have
`v_grad_dampen = 1000` and `ei_strength = 0` in their saved configurations and
completed training records. Eighteen are the six manuscript activity conditions
at seeds 42–44; the remaining three are the separate full-pool canonical family.

All **324 files** belonging to the **81 reused models** were compared directly
with `exp022-r007-compute`: configuration, metrics, validation-selected weights
and final weights were byte-identical. Both checkpoint hashes were also verified
for every replacement model. Across replacement configurations, the only changed
shared scientific setting is the damping divisor; two reported input-weight
initialization means differ by at most 1.87e-9, while the initialization rules
and parameters are unchanged.

All **84 included classifiers** have damping 1000 and complete histories for
epochs 1–50. All **4,200 epoch records** report zero skipped optimizer steps
and zero NaN-output batches. Recomputing validation-loss selection, with accuracy
and earliest-epoch tie rules, reproduces all 84 selected epochs. Recorded
training performance identifies CUDA and PyTorch 2.11.0+cu128 throughout.

The current [training recipe](/Users/eoin/pinglab/experiments/exp022/recipe.py)
and [replacement contract](/Users/eoin/pinglab/experiments/exp022/reuse_contract.py)
agree with those configurations. The exponential-Euler update in
[models.py](/Users/eoin/pinglab/tools/snnsim/models.py:252) damps the membrane
increment for both hidden populations before voltage clipping and reset.
The simulator, trainer and bank recipe have no changes between the recorded
replacement training commit `e94a3c2a` and this review's starting HEAD
`4b3009f9`. Output updates remain undamped.

## Numerical checks

| Evidence | Verified result |
| --- | --- |
| Figure 3 | Unpenalised COBA **90.900000% at 88.168810 Hz**; PING **89.766667% at 16.606688 Hz**. Rate ratio **5.30924** and accuracy difference **1.133333 percentage points**. At the 10-Hz ceiling, accuracies **83.5% / 88.6%** and rates **8.970000 / 9.083757 Hz**, COBA/PING. All 36 endpoint accuracy and rate rows match compute metrics; accuracies were recalculated from correct/total counts. |
| Figure 4 | COBA E rate **85.883789 → 7.222057 Hz**; accuracy **90.433333 → 43.166667%**; terminal I rate **39.995736 Hz**. All 33 accuracy rows match correct/total counts in compute outputs. |
| Figure 7 | At 80% deletion, **88.933333% / 89.166667%**, COBA/PING. At 100% insertion, **76.4% / 38.5%**; at 200%, **53.133333% / 11.366667%**. All 192 perturbation accuracy rows match compute correct/total counts. The plotted insertion doses remain relative to each network's unperturbed test E rate. |
| Figure 6 | All 18 spectral peaks were reproduced from recorded population traces with zero numerical difference. The saved affine fit gives slope **0.284848306**, intercept **−0.699151219 Hz**, and R² **0.997394710**. Summed per-network cycle buckets reproduce **167,178,240** pairs and **95.374034%** one-spike participation among active pairs. |
| Figure 8 | Independent jitter at 14 ms gives **0.007480469 Hz**, correctly reported as **0.0075 Hz**. Group jitter gives **68.317920 Hz**; inhibitory rate remains **108.270111 Hz**. |
| Figure 9 | All 132 grid accuracies were recalculated from recorded output-count argmax decisions, covering **26,400 decisions**. Label arrays agree across conditions. The reported rounded endpoints remain **75.2% → 90.3%** and **26.3% → 82.7%**. |
| Mean-field criticality | Recalculation from retained trajectories gives amplitude-squared slope **0.000113450295**, R² **0.999436785**, and maximum hysteresis gap **1.345330e-6 ms⁻¹**, retaining the numerical supercritical verdict. |

The untrained circuit summaries, recurrent-weight changes, timestep summaries,
and pooled/equal-network cycle fractions also agree with the manuscript's
reported values. For the twelve zero-inhibition presentations, the endpoint
indices are 37, 106, 136, 157, 224, 329, 424, 435, 478, 584, 748 and 853, all at
27-ms inhibitory decay and seed 43. They contain 398–1,153 excitatory spikes.
This pass did not repeat the previous review's full event-by-event cycle
recount or every equation derivation.

## Selected evidence and presentation refresh

Default article selections at this review, without URL overrides:

| Input | Completed present run |
| --- | --- |
| exp023 | exp023-r014-present |
| exp025 | exp025-r010-present |
| exp037 | exp037-r020-present |
| exp038 | exp038-r011-present |
| exp042 | exp042-r019-present |
| exp044 | exp044-r009-present |
| exp046 | exp046-r008-present |
| exp049 | exp049-r015-present |
| exp082 | exp082-r030-present |
| exp110 | exp110-r024-present |

The three replacement compute lineages terminate at `exp022-r015-compute`:
`exp025-r008-compute → exp025-r009-analyse → exp025-r010-present`,
`exp037-r018-compute → exp037-r019-analyse → exp037-r020-present`, and
`exp038-r009-compute → exp038-r010-analyse → exp038-r011-present`.

The new manuscript presentation has payload
`sha256:980bbd9c827739fb81beafce927df061e828be8ef3c627d6d74b1fe2548f58a8`.
Its seven input references, including digests, exactly match
`exp110-r022-present`. It redraws the existing measurements with corrected
labels. The intermediate `exp110-r023-present` and all earlier runs remain
immutable. The final default selections span **42 ancestry-linked runs**.
Selections are a dated observation, not a guarantee about future default inputs.

Read-only discovery validated the completed store, payload checksums and input
digests. The final local Demolab build passed for 20 entries; its exp110 input
selection matches the table above. Structural HTML checks found 33 sequential
equations, 14 labelled figures/tables, 11 images with alternative text, no
broken fragment links and resolving local experiment links. Word-count values
match the final rendered text, and `git diff --check` passed. The revised
Figures 2 and 6 were inspected as stored images.
No network training, inference, ODE integration, automated test suite, source
commit, push or publication was performed by this review.

Remaining scientific/editorial gaps are the uncalibrated mean-field noise
scale, the mechanism of failed inhibitory recruitment, the small replicate
count and sampling rationale, and complete archived source/patch records for
earlier executions marked dirty. Agreement with current code cannot reconstruct
missing executed patches or establish the absence of discarded attempts.
