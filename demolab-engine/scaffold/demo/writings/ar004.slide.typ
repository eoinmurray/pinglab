// ar004: the demolab talk — an intro for an academic lab group, in two acts: coding agents in
// computational science (part one), then demolab as the workflow built on them (part two). It's a
// *deck* (`.slide.typ`): declares `#let meta` but no `#let body`, compiled standalone to a PDF and
// linked from the homepage, excluded from the HTML/book passes. Compiled with `--root .`. Most
// slides are lifted from a catalog layout (SLIDES.md D11 / the gallery ar005.slide.typ) — the
// `// layout: <name>` note marks which block it came from; the `// custom:` ones are bespoke.
#import "@preview/touying:0.6.1": *
#import themes.simple: *
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot

#let meta = (
  title: "Coding agents in computational science",
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
  #set align(left)
  = Coding agents in computational science
  #v(0.4em)
  What AI can and can't do for our work, and a reproducible workflow built on it.
  #v(1.2em)
  #text(size: 17pt, fill: muted)[Eoin Murray · lab group talk]
]

// ═══════════════════════════════ PART ONE ═══════════════════════════════
// layout: section-divider
#focus-slide(background: white, foreground: ink)[
  #text(size: 15pt, fill: muted)[PART ONE]
  #v(0.4em)
  #text(size: 44pt, weight: "bold", fill: ink)[Coding agents]
  #v(0.5em)
  #line(length: 20%, stroke: 1pt + muted)
]

// layout: bullets
== An observation

#text(size: 32pt)[
  #v(0.5em)
  - *Software engineers* use coding agents like crazy.
  #v(0.5em)
  - *Scientists*, far less so.
]

// custom: SWE-bench Verified capability curve — real numbers (agents resolving real, human-validated
// GitHub issues, ~33% at the 2024 launch to ~77% in late 2025), with the autocomplete → chat → agent
// arc as context. Shows a skeptical room how fast this got good. Source: swebench.com.
== Coding agents are now useful

#align(center)[
  #cetz.canvas(length: 1cm, {
    import cetz.draw: *
    plot.plot(
      size: (16, 6.2),
      x-min: 2024.42,
      x-max: 2026,
      y-min: 0,
      y-max: 100,
      x-tick-step: 0.5,
      y-tick-step: 25,
      x-minor-tick-step: none,
      y-minor-tick-step: none,
      y-grid: true,
      x-label: none,
      y-label: none,
      x-format: v => {
        let months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        let y = calc.floor(v + 0.001)
        let mi = int(calc.round((v - y) * 12))
        if mi == 12 {
          y = y + 1
          mi = 0
        }
        text(size: 13pt)[#months.at(mi) #str(y)]
      },
      y-format: v => text(size: 15pt)[#str(calc.round(v))%],
      axis-style: "left",
      {
        let data = (
          (2024.6, 33),
          (2024.85, 49),
          (2025.05, 55),
          (2025.2, 63),
          (2025.4, 72),
          (2025.7, 74),
          (2025.92, 77),
        )
        plot.add(
          data,
          mark: "o",
          mark-size: 0.15,
          fill: true,
          fill-type: "axis",
          style: (stroke: 2.8pt + ink, fill: rgb("#ecece9")),
          mark-style: (fill: ink, stroke: none),
        )
        plot.annotate(resize: false, {
          content((2024.62, 43), text(size: 14pt, weight: "bold", fill: ink)[33%])
          content((2025.78, 88), text(size: 17pt, weight: "bold", fill: ink)[77%])
        })
      },
    )
  })
]
#v(0.4em)
#align(center)[#text(
  size: 15pt,
  fill: muted,
)[Share of real, human-validated GitHub issues an agent resolves end-to-end — SWE-bench Verified (swebench.com). Software bugs, not science.]]
#v(0.6em)
#align(center)[#text(
  size: 16pt,
)[autocomplete #text(fill: muted)[(2021)] #sym.arrow.r chat #text(fill: muted)[(2023)] #sym.arrow.r *agents that do real work* #text(fill: muted)[(2025)]]]

// layout: bullets
== What a coding agent is

- *Does the work*, not just autocomplete: reads, runs, edits, verifies.
- Driven in *plain language*, from your terminal.
- Many exist, commercial and open — no vendor lock-in.
- Only recently good enough — still uneven, still supervised.

// layout: bullets
== More is possible

- Not the same work done faster — a *lower bar for what's worth doing at all*.
- A task that cost an afternoon, you skipped. At ten minutes, you just do it.
- So the skipped work happens: the extra *baseline*, the *ablation*, reproducing a *cited result*, the *README* nobody writes.
- You take on *more — and things you'd never have bothered with*.

// layout: bullets
== Weaknesses of coding agents

- *Confidently wrong*: they fabricate APIs, numbers, and citations.
- *Comprehension debt*: they implement things you don't understand — and you ship them.
- *Weak judgement*: they won't tell you the experiment itself is a bad idea.
- *Drift & noise*: code and prose fall out of sync; they over-produce.
- *Brainrot*: prompt-and-wait is a slot machine — personally, I can't agent-code and read deeply the same day.

// ═══════════════════════════════ PART TWO ═══════════════════════════════
// layout: section-divider
#focus-slide(background: white, foreground: ink)[
  #text(size: 15pt, fill: muted)[PART TWO]
  #v(0.4em)
  #text(size: 44pt, weight: "bold", fill: ink)[Demolab]
  #v(0.4em)
  #text(size: 20pt, fill: muted)[Rails for the agent]
]

// layout: bullets (numbered — the principles)
== Principles of demolab

+ `tools/` compute, `experiments/` do analysis and plots.
+ Each experiment is a runner (`.py`) and a write-up (`.typ`).
+ Tools and experiments talk through *data files*, never imports.
+ *Raw data is disposable* (`temp/`); the distilled record is committed (`artifacts/`).
+ Bring any stack.
+ *Agent-operated*: plain language in, you stay in the loop.
+ Provenance built in.

// layout: two-column
== Typst: a modern LaTeX

#grid(
  columns: (1fr, 1fr),
  gutter: 28pt,
  [
    *LaTeX*
    - Slow, multi-pass compiles.
    - Cryptic error messages.
    - PDF only.
    - Macro soup and package hell.
  ],
  [
    *Typst*
    - *Instant*, incremental compiles.
    - *Readable* errors with line numbers.
    - One source → *web and PDF*.
    - *Imports files*, no hardcoded numbers.
  ],
)
#v(1em)
#align(center)[*The same beautiful math: selectable MathML on the web, typeset in the PDF.*]

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

// custom: guides + runbooks rendered as a stylised terminal panel — dark, monospace, with a cursor —
// so the slide *shows* the command grammar: you type a NAME to your coding agent and it drives
// demolab. The dark panel also gives the light deck a striking break.
== Guides and runbooks

#let fg = rgb("#f4f4f2")
#let dim = rgb("#8f8f8f")
#align(center)[
  #block(fill: ink, radius: 12pt, inset: (x: 30pt, y: 26pt), width: 90%)[
    #set align(left)
    #set text(font: "DejaVu Sans Mono", fill: fg, size: 16pt)
    #set par(leading: 0.85em)
    #text(fill: dim)[my-coding-agent ▸ ]#text(weight: "bold")[HELP]#h(0.2em)#box(
      fill: fg,
      width: 0.5em,
      height: 0.95em,
      baseline: 0.12em,
    )
    #v(1.1em)
    #text(weight: "bold")[GUIDES]#h(1.4em)#text(fill: dim)[\# always on — walk me through it]
    #v(0.4em)
    RULES HOUSESTYLE SLIDES STRUCTURE GLOSSARY SUPPORT
    #v(1.1em)
    #text(weight: "bold")[RUNBOOKS]#h(1.1em)#text(fill: dim)[\# on demand — run it, step by step]
    #v(0.4em)
    GETTING-STARTED TOUR LINT DOCTOR RED-TEAM STEELMAN \
    NEXT MIGRATE-CODE MIGRATE-STACK FROM-JUPYTER FROM-PAPER \
    EMBED-DOCS GROUND-CLAIMS UPDATE
  ]
]

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
