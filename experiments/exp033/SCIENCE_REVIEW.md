# Scientific and writing review — author decision pending

Reviewed against the historical evidence retained by `exp033-r001-compute`,
independent analysis `exp033-r002-analyse`, presentation `exp033-r003-present`,
and the current numerical implementation on 2026-08-28. This is an audit, not an
approved scientific rewrite. The article remains unchanged and is not Reviewed.

## Substantive corrections requiring author approval

1. **The equilibrium is not silent.** At the reference crossing the retained E
   and I rates are 4.1041319230864385 Hz and 0.705727461978268 Hz. Describe loss of
   stability of a low-rate equilibrium. The mean-field model provides a possible
   mechanism for the spiking recruitment transition; it does not establish that
   the separately observed spiking cliff is this exact bifurcation.
2. **Supercriticality is an empirical classification here.** The 25-point ramps
   give a maximum branch difference of 2.207843534999393e-6 per ms, zero measured
   width at the selected amplitude threshold, and an amplitude-squared fit with
   R² 0.9989735207704955. These support a continuous, reversible onset at the
   sampled resolution; they do not prove absence of an unstable cycle or a narrow
   bistable interval. No first Lyapunov coefficient was computed. Preserve the
   classifier and measurements, but qualify prose and success-criterion labels.
3. **The dimension argument overreaches.** A periodic orbit is a one-dimensional
   curve, not necessarily a planar curve. A local centre manifold is generally
   curved and tangent to the critical eigenspace; closed pairwise projections do
   not demonstrate its existence or dimension. A local two-dimensional reduced
   vector field is compatible with Hopf theory. Quasi-steady elimination of two
   original variables is a different approximation. Restrict the three-variable
   minimum to the tested ring/QSS family; preserve the negative-divergence
   calculations and all tested reductions, without claiming that every possible
   two-dimensional model would introduce a different physical mechanism.
4. **The frequency comparison is misstated.** Both curves decrease, but the
   mean-field curve crosses above the spiking measurement at 27 ms. It does not
   remain below throughout. The proposed spike-synchrony explanation of the gap
   was not isolated by an intervention in this experiment and should be labelled
   as a hypothesis. Make the conversion from rad/ms to Hz explicit (factor 1000).
5. **The lag is not a signed causal-delay estimate.** The implemented estimator
   takes the absolute location of the peak in the centred cross-correlation.
   Retain that estimator, but distinguish its magnitude from the ordering seen
   in the displayed traces and from a measured synaptic round-trip delay.

The distinction between centre-manifold reduction, QSS reduction and Hopf
criticality is supported by Zhang, Kirk, Sneyd and Wechselberger (2011),
[Changes in the criticality of Hopf bifurcations due to certain model reduction
techniques in systems with multiple timescales](https://link.springer.com/article/10.1186/2190-8567-1-9),
especially the introduction and section 3. This reference supports the theoretical
qualification; it is not evidence about the numerical results of this study.

## Evidence limitations to retain

- The effective voltage-noise scale is free; 3–6 mV is a sensitivity sweep, not
  a calibration to the spiking network. Driving forces are fixed and shunting
  dynamics are omitted. This is a phenomenological population-rate closure;
  calling it a fully self-consistent dynamical mean-field theory needs justification.
- The original execution reports numerical quadrature subdivision, roundoff and
  convergence warnings. They remain in provenance. Without production reruns or
  missing raw trajectories, this migration cannot certify their scientific impact.
- Four waveform figures are retained historical observations, not regenerated
  trajectories. The matching 401-point continuation is separately produced
  mean-field evidence from the compound analysis, with its own producer identity.
- The retained summary's seed medians differ from the current upstream medians
  by at most 2.5458741532702334e-6 Hz. Original scalars remain unchanged.
- Local amplitude regression differs from the original slope by
  5.421010862427522e-20. The calculation passes the recorded tolerance; the original
  scalar is preserved, and the local calculation is separately recorded.

## Writing and display corrections

- Preserve the long derivation and useful equations in an appendix. Compress
  Methods to scientific operations, including continuation/refinement, time
  integration, measurement windows, sensitivity grids, reductions and seed medians.
- Results currently contain narrative outside captions and unnumbered subsection
  headings. Merge those results into concise captions with numbered headings.
  There is no Discussion section to remove.
- Hardcoded figure references predate insertion of the sensitivity figure and
  are off by one. Use semantic Typst figure labels/references.
- Run stamps, internal experiment labels and identifiers remain visible in
  figures/prose. Remove these only in a new presentation; never alter imported
  evidence or completed runs. Preserve source and transformed hashes for carried
  figures if their presentation copies are sanitised.
- The sensitivity frequency axis currently expands approximately 1e-8 Hz
  fluctuations about 27.566 Hz. Use a sensible absolute-Hz range, no offset, so
  numerical noise is not visually presented as substantial sensitivity.
- The last historical figure's legend overlaps the high-amplitude trace. Its
  correction must be a disclosed SVG presentation edit, not a fabricated rerun.
- Preserve creation date 2026-05-28. Set updated_at only when substantive revisions
  are made. Advance status on evidence, but do not mark Reviewed without approval.

## Rendering checked at this checkpoint

The unchanged article was compiled against the explicitly validated presentation
in a private preview (no shared publication binding). All nine HTML images loaded,
225 MathML expressions were present, and the desktop viewport had no horizontal
overflow. The 11-page PDF was rasterized and every page visually inspected at
contact-sheet scale; figures and equations render, but the above scientific,
label and axis defects mean this is **not a completed writing review**. Full-size
post-correction visual checks, responsive checks and regression tests for the
writing/display corrections remain pending approval.
