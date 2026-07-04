#let meta = (
  title: "House Rules",
  date: "2026-06-20",
  description: "The conventions every contributor (human or agent) must follow when working in this repository: tooling, running experiments, version control, and write-up / figure style.",
  collection: "documentation",
)

#let body = [
  The conventions every contributor (human or agent) must follow when working in this repository. These were previously kept in _CLAUDE.md_ at the repo root; that file now points here. Read this before making any edits, alongside the reference articles (parameters, metrics, training, the #link("/ar011/")[pinglab-cli]) and the project-memory _feedback\_notebook\__ entries, which together hold the layout, glossary, invariants, and conventions every edit must respect. (The former _src/docs/src/pages/styleguide.md_ was migrated into those articles and memory and no longer exists as a single file.)

  == Tooling

  - Use _uv_ for Python and _bun_ for JavaScript.
  - Read all the READMEs in the repo before working.
  - There are no side effects — running simulations and tests locally is safe.
  - Run things in the background where possible.

  == Running experiments

  - When making runs, always report timing: start time, current time, and ETA.
  - On Modal, also report the estimated cost.

  == Modal spends real money

  - Do *not* dispatch jobs to Modal (the _--modal-gpu_ flag) without explicit permission. Default to local runs; only use _--modal-gpu_ when told to.

  == Version control

  - Do *not* create branches or open pull requests unless explicitly asked — commit to the current branch (usually _main_) and stop. "Commit and push" means commit + push, not branch + PR.
  - The author drives version control. Do not repeatedly offer to commit; end a piece of work with the result and a _science-focused_ next step, not repo bookkeeping. Still commit or push when explicitly asked.

  == GitHub

  - *Never* comment on GitHub issues — no posting or replying. Reading issues is fine; writing to them is not.

  == House notebook rules

  - Notebook runners (_src/notebooks/nbNNN.py_) take only _--modal-gpu_ (dispatch target) plus any lifecycle flags they support (_--no-wipe-dir_, _--skip-training_). The old _--tier_ size knob is *retired* — each runner hardcodes its own run scale.
  - Every hyperparameter — learning rate, epochs, samples, readout, surrogate slope, regulariser strengths, and so on — is hardcoded as a literal in the runner. The runner declares its run scale in a _SCALE_ dict passed to _run\_dirs.prepare_, which stamps it into the run manifest; the entry renders that scale as a Methods table via the _RunScale_ component (single source of truth — the prose never restates the numbers). A training notebook must declare _SCALE_ and carry a _RunScale_ block in its Methods section.
  - New scientific knobs go on the #link("/ar011/")[pinglab-cli] (_src/cli/cli.py_) as flags; the notebook just passes the recipe value inline. The notebook _is_ the recipe — reproducing a result must not require remembering flags.
  - *Provenance is captured, not assumed.* _run\_dirs.prepare_ stamps a _\_manifest.json_ every run (id, UTC time, commit + dirty flag, host, scale). A run with uncommitted dependency code also captures a _\_dirty.patch_, so even an uncommitted run is reproducible (checkout the commit, apply the patch, re-run) — you do *not* need to commit before running. The per-entry status bar reads this to show last-run date, the locked commit, and staleness.
  - *Train once, reuse many (gamma-gated-sparsity collection).* The canonical networks are trained in one place — #link("/exp022/")[exp022] (Training) — to a shared artifact root, and analysis notebooks load those cells via _exp022.load\_cell_ instead of retraining their own. This deliberately overrides the older "standalone runner, no cross-notebook helpers" convention for this collection: the redundant baseline was being retrained five times over, and the probe notebooks had drifted to lighter epoch budgets. The standard is 100 epochs at _dt_ = 0.1 ms (#link("/exp044/")[exp044]'s dt sweep is the documented exception). A migrated notebook keeps only its analysis; its training block is gone.

  Write-up conventions for a _paper_-structured entry:

  - *Structure: Abstract → Methods → Results.* Methods is a numbered (itemised) list of the steps. Results is *figures only* — no prose body; the findings live in the captions.
  - *Abstract shape: hypothesis → experiments → verdict.* Short and high-level: state the hypothesis clearly first, then a one-line description of the experiment(s) run, then a sentence on whether it holds. No long motivation.
  - *Captions are standalone, for a cold read.* Each caption names the axes/panels, the conditions, defines its terms inline (SNR, M, raster, and so on), and ends with the takeaway — so a reader who jumps straight to a plot understands it without the body text.
  - *No number prefix in the entry title.* The id lives in the status badge (id · status · date · collection); the title is the bare sentence (for example "Gamma band tightness is a precision signal a neuron can read", not "055 — …").
  - *Report results honestly, including partial or negative ones*, folding the caveats into the captions rather than hiding them; prefer one coherent entry over several linked stubs.

  == House figure rules

  - *Plots are black by default — keep them black as much as possible.* Use ink black (_theme.INK\_BLACK_, _\#1a1a1a_) for a trace; introduce a second colour, signal red (_theme.DEEP\_RED_, _\#c8102e_), only when two or more traces share one axis and must be told apart. *Separate subplots stay black* — do not colour panels differently for decoration or just because they show different conditions. Fixed exception: an excitatory (E) population trace is *always black* and an inhibitory (I) population trace is *always red*. Reach for the theme accents (amber, electric cyan) or the grey ramp only when a third-or-later overlaid series strictly needs it — never an ad-hoc hex.
  - Figures are *strictly 16:9* — no exceptions, including reliability diagrams and anything you would be tempted to make square. Use a figsize whose width:height is 16:9 (e.g. 8×4.5, 10×5.625, 12×6.75), with wider widths for more side-by-side panels; never a square or portrait frame.
  - The matplotlib theme saves with a tight bounding box, which trims whitespace and skews the saved ratio. Set _savefig.bbox_ to _standard_ in the plot (after applying the theme) so the saved canvas is exactly figsize × dpi, i.e. exactly 16:9. Verify the saved pixel dimensions if in doubt.
  - No log axes unless instructed.
  - In matplotlib titles, labels, and annotations, use the Unicode ≈ for "approximately" — never an ASCII tilde.

  == Prose style (docs)

  - Never use a tilde or _\\sim_ for "approximately". Use ≈ in prose and _\\approx_ inside math. _\\sim_ is reserved for "distributed as" (for example, $u tilde "Uniform"(0,1)$).
  - No backtick inline code in _src/docs/_ markdown — use plain text, italics, or math instead (this article included).
]
