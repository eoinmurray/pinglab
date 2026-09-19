# Plan — replace exp033 with exp117 in exp110

Objective: replace exp110's exp033 mean-field dependency with exp117 while preserving the existing Figure 2 G–I presentation. Figure 2I must remain a genuine synthesis: the mean-field curve comes from exp117 and the spiking-network measurements come from exp041.

1. Preserve the live baseline.
   1. Treat the existing modifications to `experiments/exp110/README.md`, `__main__.py`, `plots.py`, `present.py`, `recipe.py`, `test.py`, `writings/demolab_pingstore.py`, and `writings/exp110.typ` as user work.
   2. Treat the untracked exp117 files as live work.
   3. Do not reset, regenerate, or overwrite unrelated changes.

2. Replace the exp033 analysis reader in `experiments/exp110/present.py`.
   1. Replace `_exp033_analysis()` with `_exp117_analysis()`.
   2. Require an `analyse` run for experiment `exp117` with exactly one compute input.
   3. Validate the exp117 recipe and configuration consistently across its compute and analysis runs.
   4. Require schemas `exp117.recipe/v3`, `exp117.analysis/v3`, and `exp117.plot-coordinates/v4`.
   5. Read `results.json` and `plot_coordinates.json` from the validated exp117 analysis run.

3. Add an exp110-local adapter from the exp117 outputs to the existing Figure 2 data contract.
   1. For Figure 2G, use `rows[*].I_ext_nA`, `rows[*].eigenvalues_per_ms`, `I_ext_star_nA`, and `omega_Hopf_rad_per_ms`.
   2. For Figure 2H, use `criticality.relative_drive_nA`, `amplitude_up_Hz`, and `amplitude_down_Hz`.
   3. Reconstruct absolute drive as `I_ext = I_ext_star_nA + relative_drive_nA`.
   4. Divide the amplitudes in hertz by 1000 if the existing panel axis remains in inverse milliseconds.
   5. For the black curve in Figure 2I, use `frequency_vs_tau_GABA[*].tau_GABA_ms` and `frequency_vs_tau_GABA[*].f_Hopf_Hz`.
   6. Preserve the current exp110 panel semantics and axes rather than claiming that exp117 implements the old exp033 schema.

4. Preserve the two-source merge in Figure 2I.
   1. Draw the black solid mean-field curve from exp117.
   2. Draw the red dashed spiking-network medians from exp041.
   3. Keep `exp041-r005-present` as the canonical exp041 source and `exp041-r002-analyse` as its pinned analysis input.
   4. Validate all 18 exp041 rows: six tau values crossed with seeds 42, 43, and 44.
   5. Join exp117 and exp041 values by tau value, never by array position.
   6. Require the exact shared grid `[4.5, 6, 9, 12, 18, 27]` ms.
   7. Attribute Figure 2I and its caption to both exp117 and exp041, never to exp117 alone.

5. Update exp110 provenance and recipe metadata.
   1. Rename the presentation argument `exp033_identity` to `exp117_identity`.
   2. Rename the input role `exp033_analysis` to `exp117_analysis`.
   3. Replace `source_recipes.exp033` and the corresponding configuration argument with exp117.
   4. Bump the presentation schema to `exp110.presentation/v15`, since v14 already exists historically.
   5. Change the CLI `--theory-source` help text and `__main__.py` wiring to identify an exp117 analysis run.
   6. Use the raw analysis run `exp117-r024-analyse`; do not use an exp117 presentation SVG as source data.

6. Update publication dependencies without erasing exp033's independent publication.
   1. In `writings/demolab_pingstore.py`, replace exp110's exp033 dependency with exp117.
   2. Leave exp033's standalone declaration intact.
   3. Leave the exp033 source in `demolab.yaml`, because exp033 still has its own article.
   4. Retain the existing exp117 source in `demolab.yaml`.
   5. Keep the input list in `writings/exp110.typ` exp110-only, because upstream provenance belongs in the synthesized exp110 presentation run.

7. Update the exp110 manuscript.
   1. Change the Figure 2 source link from exp033 to the exact title `exp117 — Mean-Field Analysis of PING Bifurcations`.
   2. Preserve and verify the reported values against `exp117-r024-analyse`: Hopf drive approximately 0.594 nA, Hopf frequency 27.6 Hz, criticality fit R² approximately 0.999, and frequency decreasing from 30.2 Hz to 17.9 Hz.
   3. Update Methods, Appendix C2, and Table C1 to describe exp117 accurately.
   4. Describe the scalar Brent equilibrium root rather than MINPACK two-variable continuation.
   5. Describe the analytical Jacobian rather than centred finite differences.
   6. Record equilibrium tolerances `xtol = 1e-13`, `rtol = 1e-12`, and residual at most `1e-11 ms⁻¹`.
   7. Record positive-imaginary eigenvalue eligibility above `1e-8 ms⁻¹`.
   8. Record Hopf refinement tolerances `xtol = rtol = 1e-12` and the transversality step of `1e-5 nA`.
   9. Preserve the criticality-ramp description: 25 drive points from −0.1 to +0.55 nA, 2000 ms integrations, the final 500 ms analysed, LSODA with `rtol = 1e-7`, `atol = 1e-10`, and `max_step = 1 ms`.
   10. State that the final 500 ms are measured on 1001 fixed samples, not adaptive solver-output times.
   11. Preserve the limitation that no first Lyapunov coefficient was calculated, so narrow bistability or unstable cycles are not excluded.
   12. Keep the estimator distinction explicit: exp117 supplies the Hopf eigenfrequency, whereas exp041 supplies finite-drive spectral peaks.
   13. Retain the current substantive-edit date if it is already 2026-09-19; never move it backwards.

8. Update the exp110 tests.
   1. Import the exp117 recipe instead of the exp033 recipe.
   2. Stage exp117 fixtures with the required exp117 schemas.
   3. Monkeypatch `_exp117_analysis()` rather than `_exp033_analysis()`.
   4. Update the expected inputs and `source_recipes` assertions.
   5. Add an explicit assertion that Figure 2I contains the merged exp117 and exp041 series on the exact shared tau grid.
   6. Verify that the presentation still exports the same six files and that unrelated compound figures are unchanged.
   7. Do not run any tests until the implementation has been committed and pushed.

9. Update `experiments/exp110/README.md`.
   1. Change the current example command so `--theory-source` points to `exp117-r024-analyse`.
   2. Change the current contract prose from exp033 to exp117.
   3. Append a dated history entry for the migration.
   4. Preserve the historical exp033 and exp115 entries unchanged.

10. Commit, push, and then validate in repository-required order.
    1. Review the exact diff and confirm that unrelated dirty changes remain intact.
    2. Commit and push the implementation before running tests or builds.
    3. Run the targeted exp110 tests after the push.
    4. Generate the new presentation with `uv run python -m experiments.exp110.present --source exp054-r016-analyse --theory-source exp117-r024-analyse`.
    5. Confirm that the new run pins `exp054-r016-analyse`, `exp117-r024-analyse`, the exp041 presentation and analysis pair, the exp046 presentation and analysis pair, `exp037-r020`, and `exp044-r009` as appropriate.
    6. Inspect Figure 2 G–I and compare its values with the validated exp117 and exp041 inputs.
    7. Confirm that Figures 6 and 7 remain byte-identical to `exp110-r030-present` where their inputs and rendering are unchanged.
    8. Run `uv run pingstore discover`.
    9. Build the manuscript and inspect the rendered Figure 2 source links, caption, methods, equations, and appendix text.
    10. If corrections are needed, commit and push them before rerunning tests or builds.

11. Check the completion criteria.
    1. Confirm that exp110 has no runtime exp033 dependency in code, configuration, input lineage, publication dependency metadata, or the current prose source link.
    2. Allow exp033 references only in preserved historical documentation.
    3. Confirm that Figure 2G and Figure 2H use exp117 only.
    4. Confirm that Figure 2I uses exp117 and exp041 together.
    5. Confirm that the new exp110 presentation run validates and is discoverable.
    6. Confirm that the exp110 article builds successfully.
    7. Confirm that no unrelated working-tree edits were overwritten.
