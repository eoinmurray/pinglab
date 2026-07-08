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
  title: "An opinionated take on coding agents in computational science",
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
  = An opinionated take on coding agents in computational science
  #v(1.2em)
  #text(size: 17pt, fill: muted)[Eoin Murray]
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
== What a coding agent is (demo)

- *Does the work*, not just autocomplete: reads, runs, edits, verifies.
- *Can do anything your terminal can do*.
- Many exist, commercial and open — no vendor lock-in.
- Only recently good enough — still uneven, still supervised.

// layout: bullets
== Strengths of coding agents

- *The same work faster*.
- *Do the things you always wanted but didn't have time*.

// layout: bullets
== Weaknesses of coding agents

- *Confidently wrong* — fake APIs, numbers, citations.
- *Comprehension debt* — you ship code you don't understand.
- *Weak judgement* — won't kill a bad experiment.
- *Brainrot* — can't agent-code and read deeply the same day.

// layout: table (custom: agent pricing — seat price + usage tail; June 2026 billing)
== The price of coding agents

#v(0.2em)
#align(center)[
  #table(
    columns: (1.2fr, 1fr, 1.2fr),
    stroke: none,
    align: (left, right, right),
    inset: 7pt,
    table.header([*Agent*], [*Entry*], [*Heavy use/mo*]),
    table.hline(),
    [GitHub Copilot], [\$10/mo], [\$39–100/mo],
    [Cursor], [\$20/mo], [\$60–200/mo],
    [Windsurf], [\$20/mo], [\$200/mo],
    [Claude Code], [\$20/mo], [\$100–200/mo],
    [Codex (ChatGPT)], [\$20/mo], [usage tail],
    [Antigravity], [\$20/mo], [\$100–200/mo],
    [DeepSeek (API)], [API key], [\$15–90/mo],
    [Local Llama (8B)], [\$0], [not agent-grade],
  )
]
#v(0.5em)
#align(center)[#text(
  size: 15pt,
  fill: muted,
)[Bundled agents: seat + usage meter. DeepSeek: pay-per-token. Local Llama: free, assist-tier only.]]

// ═══════════════════════════════ PART TWO ═══════════════════════════════
// layout: section-divider
#focus-slide(background: white, foreground: ink)[
  #text(size: 15pt, fill: muted)[PART TWO]
  #v(0.4em)
  #text(size: 44pt, weight: "bold", fill: ink)[Demolab]
  #v(0.4em)
  #text(size: 20pt, fill: muted)[Rails for the agent]
]

// layout: code-panel
== The shape of a demolab repo (demo: pinglab)

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

// layout: bullets (numbered — the principles)
== Principles of demolab

+ *Tools compute, experiments analyse* — clean split.
+ *One experiment* = a runner + a write-up.
+ *Talk through files*, never imports.
+ *Raw data is disposable* — the record is committed.
+ *Bring any stack.*
+ *Provenance built in.*

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
#v(0.9em)
#align(center)[*Or just operate manually — no agent required.*]

// layout: bullets
== Use any stack you want

- *Default: Python* — `uv`, `.py` runners, `pytest`.
- *Files, not imports* — tools write `numbers.json`; runners never import tools.
- *Switch by asking* — *migrate the stack to MATLAB*; the agent runs a runbook.
- *MATLAB · R · Julia · Octave* — Typst and the contract stay put.
- *Python optional* — plotting only, or drop it entirely.

// layout: closer — bare focus-slide; no card, just a clean centred stack
#focus-slide(background: white, foreground: ink)[
  #align(center)[
    #text(size: 12pt, fill: muted, tracking: 2pt)[GETTING STARTED]
    #v(0.25em)
    #link("https://demolab.eoinmurray.info")[
      #text(size: 34pt, weight: "bold", fill: ink)[demolab.eoinmurray.info]
    ]

    #v(1.6em)

    #text(size: 12pt, fill: muted, tracking: 2pt)[SUPPORT]
    #v(0.25em)
    #link("https://github.com/eoinmurray/demolab")[
      #text(size: 22pt, weight: "bold", fill: ink)[github.com/eoinmurray/demolab]
    ]
    #v(0.4em)
    #text(size: 15pt, fill: muted)[
      Open a #link("https://github.com/eoinmurray/demolab/issues")[GitHub issue] and I'll fix it quickly.
    ]
  ]
]
