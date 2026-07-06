// ar004: the demolab talk — a ~10-slide intro for an academic lab group. It's a *deck*
// (`.slide.typ`), so it declares `#let meta` but no `#let body`: compiled standalone to a PDF and
// linked from the homepage, excluded from the HTML/book passes. Compiled with `--root .`. Every
// slide is lifted from a catalog layout (SLIDES.md D11 / the gallery ar005.slide.typ) — the
// `// layout: <name>` note on each marks which block it came from.
#import "@preview/touying:0.6.1": *
#import themes.simple: *

#let meta = (
  title: "Demolab — the talk",
  date: "2026-07-06",
)

#show: simple-theme.with(aspect-ratio: "16-9", header: none)

#set text(font: "New Computer Modern", size: 22pt)
#show raw: set text(font: "DejaVu Sans Mono")

// demolab two-ink palette (see SLIDES.md D4): ink headings + bold, muted secondary.
#let ink = rgb("#1a1a1a")
#let muted = rgb("#666666")
#show heading: set text(fill: ink)
#show strong: it => text(fill: ink, weight: "bold", it.body)
#set align(horizon)  // vertically centre slide content (adaptive)

// layout: title
#title-slide[
  = Demolab
  #v(0.3em)
  An agent-operated lab notebook for computational science.
  #v(1.2em)
  #text(size: 24pt)[#link("https://demolab.eoinmurray.info")[demolab.eoinmurray.info]]
  #v(1.0em)
  #text(size: 17pt, fill: muted)[Eoin Murray · lab group talk]
]

// layout: two-column
== What demolab is, and isn't

#grid(columns: (1fr, 1fr), gutter: 28pt,
  [
    *It is*
    - *Self-publishing*: repo → site + PDFs.
    - *Reproducible*: numbers read from the run.
    - *Agent-operated*: plain language.
    - *Yours*: your science and git history.
  ],
  [
    *It isn't*
    - Another hand-maintained notebook.
    - A library you write code against.
    - Tied to Python.
    - Autonomous — you stay in the loop.
  ],
)
#v(1em)
#align(center)[*A structure and a workflow, not a framework.*]

// layout: bullets
== The gap

- In my circles, *software engineers* use coding agents constantly.
- *Academics* — far less.
- Yet our work is exactly what they're good at: *code + write-up + literature*.
- Meanwhile results rot: scripts, stray plots, numbers retyped into the paper.
- *demolab is an attempt to close that gap.*

// layout: bullets
== Coding agents — what are they?

- AI assistants that read a repo and *do the work* — not just autocomplete.
- Run in your terminal or editor: read files, run commands, edit, verify.
- You drive them in *plain language*; they follow instructions in the repo.
- Prominent ones: *Claude Code*, *Cursor*, *Copilot* (agent mode), *Gemini CLI*, *aider*.
- demolab is *operated by* one — so there's no web dev and no build config for you.

// layout: two-column
== Before and after coding agents

#grid(columns: (1fr, 1fr), gutter: 28pt,
  [
    *Before*
    - You hand-write boilerplate, plots, build config.
    - Docs and code quietly drift apart.
    - Reproducing an old result is archaeology.
  ],
  [
    *After*
    - You describe the result; it scaffolds and runs it.
    - Prose, figures, and numbers stay in sync.
    - Roughly *5× the throughput*, more projects in flight.
  ],
)
#v(1em)
#align(center)[*The plumbing stops being your job.*]

// layout: bullets (+ a centred takeaway)
== Problems with coding agents

- *Confidently wrong*: they fabricate APIs, numbers, and citations.
- *No memory*: they forget context between sessions unless it's written down.
- *Weak judgement*: they won't tell you the experiment itself is a bad idea.
- *Drift & noise*: code and prose fall out of sync; they over-produce.

#v(0.9em)
#align(center)[*demolab's answer: guardrails* — numbers from the run, a fixed structure,\
git provenance on every result, and lint / doctor / red-team checks. Trust the structure.]

// layout: code-panel
== The shape of a demolab repo

#align(center)[
  #block(fill: luma(245), stroke: 0.75pt + luma(210), radius: 12pt, inset: 26pt)[
    #set align(left)
    #text(size: 20pt)[
      ```
      tools/          the science — models & solvers
      experiments/    the runners — expNNN.py per experiment
      writings/       the writeups — one .typ per entry
      temp/           the raw run data — gitignored, big
      artifacts/      the distilled record — data/ + pdfs/ (committed)
      demolab.yaml    branding + collections (optional)
      demolab-engine/ the engine (black box you never edit)
      ```
    ]
  ]
]

// layout: two-column
== Guides and runbooks

// command chips: the names are things you type, so set them as monospace pills (atomic, so a
// kebab name never breaks mid-hyphen).
#let cmd(n) = box(fill: luma(244), stroke: 0.5pt + luma(214), radius: 5pt, inset: (x: 7pt, y: 3pt), text(size: 15pt, raw(n)))
#set par(leading: 1em)

#grid(columns: (1fr, 1fr), gutter: 28pt, align: top,
  [
    *Guides* — the rules, always on
    #v(0.7em)
    #cmd("RULES") #cmd("HOUSESTYLE") #cmd("SLIDES") #cmd("STRUCTURE") #cmd("GLOSSARY") #cmd("SUPPORT")
  ],
  [
    *Runbooks* — 14, on demand
    #v(0.7em)
    #cmd("GETTING-STARTED") #cmd("TOUR") #cmd("LINT") #cmd("DOCTOR") #cmd("RED-TEAM") #cmd("STEELMAN") #cmd("NEXT") #cmd("MIGRATE-CODE") #cmd("MIGRATE-STACK") #cmd("UPDATE") #cmd("FROM-JUPYTER") #cmd("FROM-PAPER") #cmd("EMBED-DOCS") #cmd("GROUND-CLAIMS")
  ],
)
#v(1.2em)
#align(center)[*Type the NAME — a guide walks you through, a runbook runs.*]

// layout: code-panel
== Driven by a handful of commands

#align(center)[
  #block(fill: luma(245), stroke: 0.75pt + luma(210), radius: 12pt, inset: 26pt)[
    #set align(left)
    #text(size: 20pt)[
      ```
      task install         set up (Python deps via uv)
      task run -- exp000   run an experiment end-to-end
      task dev             serve the site, live-reload on save
      task build           website + a PDF per entry + a book
      task test            run the test suite
      ```
    ]
  ]
]

// layout: bullets
== Use any stack you want

- Ships set up for *Python*: `uv`, tools + runners as `.py`, `pytest`.
- But the contract is *files, not a language*: a tool just writes `numbers.json` + data.
- So switching is a conversation: *"migrate the stack to MATLAB"* — the agent follows a runbook.
- Works for *MATLAB · R · Julia · Octave*; the Typst publishing and the contract stay put.
- Keep Python only for plotting, or drop it entirely — your call.

// layout: closer
== Try it

#align(center)[
  #text(size: 30pt)[*#link("https://demolab.eoinmurray.info")[demolab.eoinmurray.info]*]
  #v(1.0em)
  #box[
    #set align(left)
    - Open a coding agent in a new, empty folder.
    - Point it at the repo; approve as it sets things up.
    - It hands you a live URL and your first experiment.
  ]
  #v(1.0em)
  #text(size: 18pt, fill: muted)[Hand it the loop, keep the science yours.]
]
