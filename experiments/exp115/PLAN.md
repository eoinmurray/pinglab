# exp115 transition plan

Status: **numerical redesign executed and integrated; exp110 migration pending**

1. [x] Remove first-Lyapunov-coefficient and centre-manifold calculations from
   the current recipe and implementation.
2. [x] Add upward/downward time-domain ramps at all six inhibitory decay times
   for the reference noise scale and relaxation multiplier.
3. [x] Define the finite-grid criticality assessment from branch agreement and
   amplitude-squared scaling, retaining an explicit inconclusive outcome.
4. [x] Replace coefficient-based presentation fields and panel B with
   excitatory-rate amplitude ramps.
5. [x] Revise the article to remove the normal-form derivation and mark it
   article-only until a new presentation exists.
6. [x] Review and commit the revised recipe before running tests, as required by
   the repository workflow.
7. [x] Execute a fresh compute → analyse → present lineage and inspect numerical
   convergence, ramp duration sufficiency and all six decay-time assessments.
8. [x] Update the article from the validated new presentation.
9. [ ] Migrate exp110's consumer and scientific claims in a separate scoped
   change.

The older coefficient-based runs remain immutable historical evidence and are
not valid inputs for the revised amplitude-ramp article.
