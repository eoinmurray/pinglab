#let meta = (
  title: "Runbooks",
  date: "2026-07-07",
  description: "Named, on-demand procedures the agent runs step by step: onboarding, linting, migrating a codebase or a language, reproducing a paper, updating the engine.",
  collection: "documentation",
  status: "final",
)

#let base = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/runbooks"

#let body = [
  Runbooks are demolab's *procedure layer*: named, repeatable jobs the agent runs step by step,
  showing each result and confirming before it moves on. Where a guide is always-on reference, a
  runbook is invoked when you want it: type its name in capitals (`LINT`, `DOCTOR`,
  `GETTING-STARTED`) and the agent opens it and drives it. Say `HELP` for the full menu. The
  canonical text is the Markdown source, linked below.

  == Getting in

  - #link(base + "/GETTING-STARTED.md")[*GETTING-STARTED*]: set up a fresh lab end to end:
    scaffold, brand, pick a stack, get it building and optionally published, then land a first
    experiment.
  - #link(base + "/TOUR.md")[*TOUR*]: a guided walk through an existing repository, so you (or a
    new collaborator) learn where everything lives.

  == Checking the work

  - #link(base + "/LINT.md")[*LINT*]: audit the prose _and_ the figures against the house style.
  - #link(base + "/DOCTOR.md")[*DOCTOR*]: audit the repository's structure against the rules.
  - #link(base + "/RED-TEAM.md")[*RED-TEAM*]: attack a result's validity, hard, to surface what
    would break it.
  - #link(base + "/STEELMAN.md")[*STEELMAN*]: build the strongest honest case for a result.
  - #link(base + "/GROUND-CLAIMS.md")[*GROUND-CLAIMS*]: back every claim with a run or a citation.
  - #link(base + "/NEXT.md")[*NEXT*]: suggest what to run next.

  == Bringing work in

  - #link(base + "/MIGRATE-CODE.md")[*MIGRATE-CODE*]: wrap an existing codebase, one experiment at
    a time, without rewriting the science.
  - #link(base + "/MIGRATE-STACK.md")[*MIGRATE-STACK*]: switch the language your tools are written
    in (MATLAB, R, Julia, Octave).
  - #link(base + "/FROM-JUPYTER.md")[*FROM-JUPYTER*]: convert a notebook into a tool, a runner, and
    a write-up.
  - #link(base + "/FROM-PAPER.md")[*FROM-PAPER*]: reproduce a paper's result from scratch.

  == Housekeeping

  - #link(base + "/EMBED-DOCS.md")[*EMBED-DOCS*]: use demolab as a docs subfolder inside another
    project.
  - #link(base + "/UPDATE.md")[*UPDATE*]: vendor the latest engine into your lab and review what
    changed between versions.

  Their always-on counterpart, the *guides*, each has its own page in this collection.
]
