#let meta = (
  title: "Runbooks",
  date: "2026-07-06",
  description: "Named, on-demand procedures the agent runs step by step: linting, migrating code, updating the engine, and more. Each runbook links to its source on GitHub, never repeated here.",
  collection: "documentation",
)

#let base = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/runbooks"

#let body = [
  Runbooks are demolab's *procedure layer*: named, repeatable jobs the agent runs step by step,
  showing each result and confirming before it moves on. Where a guide is always-on reference, a
  runbook is invoked when you want it. Type its name in capitals (`LINT`, `DOCTOR`,
  `GETTING-STARTED`) and the agent opens the runbook and drives it; type `HELP` for the full menu.

  Each entry links to its source on GitHub, so this list stays in step with the engine.

  - #link(base + "/GETTING-STARTED.md")[*GETTING-STARTED*]: set up a fresh lab, interactively.
  - #link(base + "/TOUR.md")[*TOUR*]: walk through the repository.
  - #link(base + "/LINT.md")[*LINT*]: audit the prose and figures against the house style.
  - #link(base + "/DOCTOR.md")[*DOCTOR*]: audit the structure against the rules.
  - #link(base + "/RED-TEAM.md")[*RED-TEAM*]: attack the result's validity.
  - #link(base + "/STEELMAN.md")[*STEELMAN*]: make the strongest case for it.
  - #link(base + "/NEXT.md")[*NEXT*]: suggest what to run next.
  - #link(base + "/GROUND-CLAIMS.md")[*GROUND-CLAIMS*]: back every claim with a run or a citation.
  - #link(base + "/MIGRATE-CODE.md")[*MIGRATE-CODE*]: wrap an existing codebase.
  - #link(base + "/MIGRATE-STACK.md")[*MIGRATE-STACK*]: switch language (MATLAB, R, Julia, Octave).
  - #link(base + "/FROM-JUPYTER.md")[*FROM-JUPYTER*]: convert a notebook.
  - #link(base + "/FROM-PAPER.md")[*FROM-PAPER*]: reproduce a paper.
  - #link(base + "/EMBED-DOCS.md")[*EMBED-DOCS*]: use demolab as a docs subfolder in another project.
  - #link(base + "/UPDATE.md")[*UPDATE*]: vendor the latest engine into your lab.

  Their always-on counterpart, *guides*, are documented in a companion article in this collection.
]
