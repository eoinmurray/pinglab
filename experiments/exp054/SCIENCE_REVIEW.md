# exp054 science and writing review

Reviewed against the retained evidence on 2026-08-28. This is a technical review,
**not a Reviewed status**. The author subsequently approved the listed
scientific corrections and scoped collection registration. They are now applied
under Writing Guide 17.0.0, with creation date preserved and `updated_at` set to
2026-08-28. Approval to revise does not mark the article Reviewed.

## Evidence and scope

The empirical evidence is one seeded, untrained 256 E / 256 I network per
condition: 121 coupling points and 16 null conditions, sharing the origin probe.
There are 136 unique full recordings, not 136 independent replicates. The
historical import is `exp054-r001-compute`; independently rebuilt measurements
and initial figures are `exp054-r002-analyse` and `exp054-r003-present`.
The corrected figures are `exp054-r004-present`. Their complete ancestry validates. No production simulation, training or new mean-field solve
was performed for this migration.

All historical configuration, grid and null-control scalar sections are exactly
preserved. The independent numerical recheck differs only in contrast at floating
precision (maximum absolute difference 4.440892098500626e-16). All original NPY
member bytes survive compression. Mean-field evidence is the original exp054
producer's numerical subset already retained by exp033, not a replacement study.
Full historical mean-field ODE trajectories are absent; they were not fabricated.

## Scientific conflicts and approved resolutions

| Original article claim | Code and retained evidence | Approved correction |
| --- | --- | --- |
| Private input makes the score rate-invariant by construction. | Eight finite private-input null conditions give contrasts up to 0.0671258, with one seed. Removing shared afferents does not establish universal rate invariance of the estimator. | State the observed low null contrast over the tested firing range; distinguish this finite control from an invariance theorem. |
| Both COBA edges are exactly zero; contrast rises smoothly with coupling. | Edge contrast is 0.001690592634090043. Interior values are not strictly monotonic. Strong-corner contrast is 0.9843690649700508; the grid maximum is 0.9970636405187142 elsewhere. | Say near-zero edges and generally stronger contrast in the coupled interior; distinguish the corner from the maximum. |
| The null panels have the same rates and spike counts. | The plotting code selects the nearest available rates to 1, 2.5 and 5 Hz separately. Private panels are 0.9722, 1.8750 and 4.9002 Hz; shared panels are 0.3993, 1.7188 and 4.1970 Hz. | Describe approximate rate matching and retain the actual panel labels. Do not imply a controlled equality that was not performed. |
| The central lobe means spikes recur one cycle apart. | The score uses the maximum preceding the first nonzero-lag trough, often near 1 ms. Shared-input nulls can also produce this short-lag structure. | Explain central coincidence/clustering separately from a subsequent peak at a full period. Use the rasters and longer-lag structure as additional rhythm evidence. |
| Contrast is always in `[0,1)`, and chance is a floor. | The algebra permits 1 when the trough is zero and the lobe is positive. The finite overlap-normalized estimate can fall below chance; undefined scores remain missing. Floating-point artifacts also limit unconditional numerical bounds. | State `[0,1]` conditional on nonnegative lobe/trough and a positive denominator; call 1 the chance reference, not a floor. Preserve the estimator and its missing-value behavior. |
| The mean-field bifurcation explains/predicts this same spiking transition and establishes a supercritical, reversible Hopf. | The mean-field scan varies external drive; the spiking map varies two coupling weights. The 4 mV effective-noise reference crosses near 0.5963371 nA at 27.5664 Hz. Finite amplitude/up-down sweeps are compatible with a soft onset but do not identify the spiking transition or calculate its first Lyapunov coefficient. | Present a separate mechanistic comparison, distinguish the control axes, and qualify the criticality inference. Preserve the 4 mV reference, numerical curves and spiking frequency overlay. |

The compound figure now says “Mean-field amplitude”, with explicit units for
its amplitude and eigenvalues and a qualified frequency-comparison title. These
changes are retained in **`exp054-r004-present`**, independently generated from
the same `exp054-r002-analyse`. The earlier presentation is unchanged.
The frequency comparison is qualitative: mean-field and measured spiking
frequencies are not quantitatively coincident across inhibitory decay times.

## Applied figure and writing corrections

- Selected `turnon_maps_compound.png`, which matches the first caption's three
  maps above three rasters, instead of the single-map `turnon_compound.png`.
  The caption is shorter and the underlying data are unchanged.
- Corrected the raster caption's map position and clarified that every other
  grid coordinate is displayed. Rasters show the first 160 E and 48 I neurons
  over 200 ms; measurements use all neurons over the full 900 ms post-burn window.
- Merged useful onset interpretation into a concise Results caption, qualified
  causal claims and removed repository mechanics from rendered science. There
  was no Discussion heading to remove.
- Preserved both equations and the FFT autocorrelation, finite-overlap and
  mean-count explanations. Numbered the displayed equations and distinguished
  lag in bins from physical lag in milliseconds.
- Expanded Methods into a concise six-step procedure covering the grid, null
  recipes, timestep, recording and burn-in, single-seed scope, smoothing,
  mean-field scans, ramp measurement window and median frequency aggregation.
  The appendix retains useful technical explanation.
- Preserved creation date 2026-06-15 and set `updated_at` to 2026-08-28 for the
  approved substantive revision. No Reviewed label was assigned.

## Final verification and remaining integration

The earlier scientific conflicts above are now corrected in the article,
including alternative text. Useful onset interpretation is confined to the
Results caption; there is no Discussion section. Both equations remain, with
numbered HTML/PDF rendering and explicit lag/normalization definitions. The
FFT and finite-overlap explanation is preserved in a concise appendix. Methods
now covers the full scientific procedure in six steps.

The current six selected images all load in browser HTML at desktop and 390 px
mobile widths. Both equation numbers appear, the Methods contents link works,
and neither viewport has page-level horizontal overflow. The five-page A4 output
and corrected compound PDF were visually inspected. Figure 1 now matches its
caption; Methods starts on a fresh page with its equation attached to the step.
The missing-glyph, first-figure binding and HTML equation-number failures have
regression coverage. All scalar results in the new presentation remain identical
to the preceding presentation, apart from its run identifier.

Final review artifacts/logs are in `.r2/exp054-final-m9bwak6v/`; the initial review
is retained in `.r2/exp054-import-ch71rm5x/`. The browser uses an isolated copy
of the live article/helpers and a fresh validated projection, with only the
projection path redirected. No shared renderer, defaults or published artifacts
were modified. Full application URL routing was not tested.

The reviewed snapshot of the shared Datasets view reports the import stage's
8-second local duration without an import suffix, rather than identifying the retained original
Slurm execution (02:47:14–03:27:16 UTC, 40 minutes 2 seconds). It recognizes an
`import` operation and a separate `scientific_execution` declaration, while this
established importer uses `historical-import` and retains the original scheduler
record in `historical_import.producer`. The authoritative run provenance is
correct. Exp080 subsequently added recognition of `historical-import` to its
uncommitted shared view changes; that work is not part of this migration. Showing
the original producer span still needs coordinated projection support and tests.
The completed run must not be altered to accommodate the display.

Shared collection registration and its exp054-specific tests were subsequently
authorized, coordinated and completed. All 232 tests in the integrated target
suite pass, plus five exp054 writing checks. The separate exp033 suite has 37
passes and its existing missing-`contents.typ` fixture failure. Details and the
exact shared-file ownership boundary are documented in README.md.
