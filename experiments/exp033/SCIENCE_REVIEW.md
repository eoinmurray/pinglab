# Scientific and writing review — corrections applied, author review pending

Reviewed against the historical evidence retained by `exp033-r001-compute`,
independent analysis `exp033-r002-analyse`, presentation `exp033-r003-present`,
and the current numerical implementation on 2026-08-28. The list below records
the initial audit. The author subsequently authorized the writing/science
corrections; their implementation is recorded at the end. The article is not Reviewed.

## Initial substantive findings (now corrected in the article)

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

## Authorized correction outcome

All five substantive findings above are addressed without changing scientific
measurements or numerical recipes. The long two-dimensional reduction argument
is retained but scoped to the tested QSS ring family; the curved centre manifold
and its restricted two-dimensional vector field are distinguished from coordinate
elimination. Conductance variables and current-valued coordinates now use
different symbols, and frequency conversion explicitly includes 1000 ms/s.

Connections to the recruitment, inhibitory-decay and coupling-map studies are
linked by scientific names and distinguished as motivation, measured comparison
and reuse, respectively. The independent-cell finite-size scaling is
qualified rather than asserted for a synchronous network. Fixed driving forces,
free noise scale, unavailable raw waveforms, heuristic criticality and unresolved
quadrature warnings remain explicit. Literature supports the theoretical
qualifications, not the experiment's numerical outcomes.

The article preserves its creation date and all numbered derivations, moves
procedural detail to appendices, and uses five scientific Methods steps with
numbered Results captions. There was no Discussion section. The final explicitly
selected presentation is exp033-r005-present, with the same analysis as before;
all retained numerical results and decision flags are exactly unchanged.

The HTML and 16-page PDF were checked with nine loaded figures, resolved
references and equations, plus 390px reading views. The gain equation was split
without changing its mathematics. Visual inspection also found and fixed a
Matplotlib annotation overlap and a Typst gain-argument subscript error;
regression tests cover both. The combined suite passed 197 tests; final targeted
render/layout tests passed again. See README for exact provenance and audit paths.
No author-reviewed status, publication, materialization or new simulation is claimed.

## Author-requested restoration and visible corrections

The author requested restoration of omitted scientific material and bracketed
`[(!) ...]` markers for substantial changes. Appendix A again restores the
1,024/256 population sizes, the strength mapping `s` and `r*s`, the numerical
angular frequency, and the fuller eigenvalue/double-Hopf explanation. Appendix B
restores the dimensionality question and the van der Pol/FitzHugh–Nagumo comparison.
Appendix C restores the amplitude definition, square-root law and derivative,
unstable-cycle basin-boundary explanation, and the original mechanistic proposals.
Unsupported assertions are preserved as earlier interpretations alongside
explicitly marked qualifications, not reinstated as verified results.

One additional check used the retained 401-point eigenvalue sweep: at most two
eigenvalues are unstable, and the trace stays negative. The earlier claim of a
second pair crossing at higher drive is unsupported within 0–4 nA. The displayed
Jacobian's strictly negative trace also excludes two simultaneously imaginary
pairs for this four-filter model. This algebraic correction is bracketed.

The restored article passes 43 dedicated tests, including rendered correction
markers and the restored theory reference link. Nine HTML figures and all 18 PDF
pages were checked; restored amplitude formulas also fit the 390px reading view.
Read-only discovery confirms local presentation data; the concurrent status
migration's `[DATA]` badge is preserved under Writing Guide 11.0.0. Dates and all
40 original numbered equations remain. No numerical payload, run, publication
binding or completed run was changed. The author authorized committing and
pushing this restoration; the separate shared status migration is excluded.
