// ar005: the layout-gallery deck — the canonical source for every slide layout in SLIDES.md (D11).
// Each layout is a named block marked `// layout: <name>`; to reuse one, copy from its marker to
// the next marker and swap the demo content for yours. SLIDES.md D11 indexes the names + when-to-use;
// this deck holds the one tested, page-count-checked copy. (test_engine_build.py asserts the marker
// names here match the D11 index, so they can't drift.) It's a *deck* (`.slide.typ`): declares
// `#let meta` but no `#let body`, compiled standalone to a PDF and linked from the homepage,
// excluded from the HTML/book passes. Compile with `--root .` so /artifacts/... resolves (D3).
#import "@preview/touying:0.6.1": *
#import themes.simple: *
#import "/demolab-engine/build/lib.typ": data-file

#let meta = (
  title: "Slide layout gallery",
  date: "2026-07-06",
)

// header: none drops touying's running section header (the small gray repeat of the title on
// every slide). Turn it back on for a long, multi-section talk where that navigation helps.
#show: simple-theme.with(aspect-ratio: "16-9", header: none)

#set text(font: "New Computer Modern", size: 22pt)
#show raw: set text(font: "DejaVu Sans Mono")

// Align the deck to demolab's two-ink web palette (RULES/style.css): ink for headings + bold,
// muted for secondary text — instead of touying's default teal accent. `strong` needs a transform
// show rule; a plain `set text` on it is overridden by the theme.
#let ink = rgb("#1a1a1a")
#let muted = rgb("#666666")
#show heading: set text(fill: ink)
#show strong: it => text(fill: ink, weight: "bold", it.body)

// Vertically centre each slide's content — adaptive: a sparse slide's title + body sit as a
// balanced block, while a nearly-full slide keeps its title near the top (no room to float).
#set align(horizon)

// Real run data for the table / figure / big-number slides — read from the record, never typed.
#let run = json(data-file("exp000/numbers.json"))
#let cap(body) = text(size: 14pt, fill: muted)[#body]

// layout: title — opens the deck; mirror the closer so it bookends
#title-slide[
  #set align(left)
  = Slide layout gallery
  #v(0.4em)
  A worked example of every demolab slide layout.
  #v(1.4em)
  #text(size: 17pt, fill: muted)[Demolab · SLIDES.md D11]
]

// layout: bullets — the workhorse; bold the load-bearing phrase, five bullets max
== Bullets

- *Bold the load-bearing phrase* so the eye lands on it first.
- Two lines per bullet, maximum — past that it's prose, not a slide.
- Five bullets is the ceiling; split the slide before a sixth.
- Nested points are fine *once*, for a short aside.
- The last bullet is the *so what* — say what it means.

// layout: two-column — comparison; parallel sides + a centred takeaway
== Two-column

#grid(columns: (1fr, 1fr), gutter: 28pt,
  [
    *What a tool does*
    - Holds reusable computation.
    - Writes `numbers.json` + data.
    - Stays language-agnostic.
  ],
  [
    *What a tool does not*
    - Render plots.
    - Import a runner.
    - Hard-code a result.
  ],
)
#v(1em)
#align(center)[*Parallel sides, one takeaway: tools compute, runners narrate.*]

// layout: three-column — triads; header + one line each
== Three-column

#v(0.6em)
#grid(columns: (1fr, 1fr, 1fr), gutter: 22pt,
  [
    *Tools*
    #v(0.2em)
    Reusable computation; emit data, not plots.
  ],
  [
    *Experiments*
    #v(0.2em)
    Runners: call the tools, render the figures.
  ],
  [
    *Writings*
    #v(0.2em)
    One `.typ` per entry; reads the run.
  ],
)
#v(1em)
#align(center)[*Three parallel items — four wants a table.*]

// layout: code-panel — one snippet in a luma(245) rounded box, one idea
== Code panel

#align(center)[
  #block(fill: luma(245), stroke: 0.75pt + luma(210), radius: 12pt, inset: 24pt)[
    #set align(left)
    #text(size: 20pt)[
      ```python
      def cmd_lif(args):
          v, spikes = simulate(args.current, args.tau_m)
          write_output(run_dir,
              {"firing_rate_hz": rate(spikes)},
              manifest={"headline_metrics": ["firing_rate_hz"]})
      ```
    ]
  ]
]

// layout: equation-terms — displayed equation, then a where: list defining every symbol
== Equation + terms

$ tau_m (dif V) / (dif t) = -(V - V_"rest") + R_m thin I_"ext" (t) $

#v(0.6em)
where:
- $V$ — membrane potential (mV)
- $tau_m$ — membrane time constant (ms)
- $V_"rest"$ — resting potential (mV)
- $R_m$ — membrane resistance (MΩ)
- $I_"ext" (t)$ — external input current (nA); the space keeps $(t)$ out of the subscript (D10)

// layout: quote — centred italic pull-quote + muted attribution
== Quote

#v(1fr)
#align(center)[
  #block(width: 82%)[
    #set align(center)
    #text(size: 24pt, style: "italic")[
      "An article about computational science is not the scholarship itself; it is merely
      advertising of it. The actual scholarship is the complete code and data that generated
      the figures."
    ]
    #v(0.7em)
    #text(size: 18pt, fill: muted)[— Buckheit & Donoho, 1995]
  ]
]
#v(1fr)

// layout: section-divider — chrome-free focus-slide signposting a new part (no stale header)
#focus-slide(background: white, foreground: ink)[
  #text(size: 15pt, fill: muted)[PART TWO]
  #v(0.4em)
  #text(size: 44pt, weight: "bold", fill: ink)[Figures]
  #v(0.5em)
  #line(length: 20%, stroke: 1pt + muted)
]

// layout: centered-figure — one image + muted caption (16:9 centers; a 2:1 plot fills the width)
== Centered figure

#v(0.5em)
#align(center)[
  #image(data-file("exp000/lif.svg"), height: 250pt)
  #cap[Membrane potential of a single LIF neuron under tonic input; it charges and resets at threshold, firing at #calc.round(run.lif.firing_rate_hz) Hz.]
]

// layout: figure-bullets — figure left (~55%), three reading notes right, so-what last
== Figure + bullets

#grid(columns: (55%, 1fr), gutter: 22pt, align: horizon,
  image(data-file("exp000/lif.svg"), height: 200pt),
  [
    - Each sweep is the membrane charging through its RC constant.
    - Every vertical drop is a *reset* after a spike.
    - *So what:* constant drive gives a fixed rate, #calc.round(run.lif.firing_rate_hz) Hz.
  ],
)

// layout: figure-pair — (a)/(b) panels, captions under each, comparison line below
== Figure pair

#grid(columns: (1fr, 1fr), gutter: 20pt,
  [
    #align(center)[
      #image(data-file("exp000/lif.svg"), height: 160pt)
      #cap[(a) LIF: linear integrate-and-fire.]
    ]
  ],
  [
    #align(center)[
      #image(data-file("exp001/eif.svg"), height: 160pt)
      #cap[(b) EIF: exponential spike onset.]
    ]
  ],
)
#align(center)[*Same drive, sharper threshold: the EIF spike initiates faster.*]

// layout: figure-grid — 2×2, four panels max, per-row heights (D7)
== Figure grid (2×2)

#grid(columns: (1fr, 1fr), rows: (auto, auto), column-gutter: 18pt, row-gutter: 10pt, align: center,
  image(data-file("exp000/lif.svg"), height: 120pt),
  image(data-file("exp001/eif.svg"), height: 120pt),
  image(data-file("exp000/net.png"), height: 150pt),
  image(data-file("exp001/enet.png"), height: 150pt),
)

// layout: hero-stack — the result large on the left, evidence stacked right
== Hero + stack

#grid(columns: (60%, 1fr), gutter: 20pt, align: horizon,
  [
    #align(center)[
      #image(data-file("exp000/net.png"), height: 255pt)
      #cap[The network sustains irregular firing — the result being argued.]
    ]
  ],
  [
    #stack(spacing: 12pt,
      image(data-file("exp000/lif.svg"), height: 105pt),
      image(data-file("exp001/eif.svg"), height: 105pt),
    )
    #cap[Single-cell traces as supporting evidence.]
  ],
)

// layout: diagram — boxes + arrows drawn in Typst (no image asset)
== Diagram

#let node(b) = box(fill: luma(245), stroke: 0.75pt + luma(200), radius: 8pt, inset: (x: 14pt, y: 10pt))[#text(size: 18pt)[#b]]
#let flow(lbl) = stack(dir: ttb, spacing: 3pt, align(center, text(size: 12pt, fill: muted)[#lbl]), align(center, text(size: 22pt, fill: muted)[#sym.arrow.r]))

#align(center)[
  #grid(columns: 7, align: horizon, column-gutter: 8pt,
    node[Tool], flow[data], node[Runner], flow[figures\ + numbers], node[Writeup], flow[build], node[Site + PDF],
  )
]
#v(0.9em)
#align(center)[*Boxes and arrows in Typst — no image asset needed.*]

// layout: table — stroke:none + one hline under the header; numbers from the run
== Table

#v(0.6em)
#align(center)[
  #table(
    columns: (auto, auto), stroke: none, align: (left, right), inset: 8pt,
    table.header([*Parameter*], [*Value*]),
    table.hline(),
    [Input current], [#run.lif.config.current nA],
    [Membrane time constant $tau_m$], [#run.lif.config.tau_m ms],
    [Threshold $V_"th"$], [#run.lif.config.v_thresh mV],
    [Firing rate], [#calc.round(run.lif.firing_rate_hz) Hz],
  )
]
#v(0.4em)
#align(center)[#cap[Every value read from `numbers.json`, so the slide can't drift from the run.]]

// layout: big-number — one headline metric huge + a muted one-line gloss
== Big number

#align(center + horizon)[
  #text(size: 120pt, weight: "bold", fill: ink)[#calc.round(run.lif.firing_rate_hz) Hz]
  #v(0.2em)
  #text(size: 20pt, fill: muted)[single-neuron firing rate under tonic input — read from the run]
]

// layout: big-statement — focus-slide(background: ink); don't bold on it (accent == background)
#focus-slide(background: ink)[
  One layout per slide.
  #v(0.3em)
  Title, plus bullets or one visual.
]

// layout: closer — mirror the title slide; left-align the bullets in a #box
== Closer

#align(center)[
  #text(size: 28pt)[*demolab-engine/guides/SLIDES.md*]
  #v(0.8em)
  // Left-align the list inside a shrink-wrapped box, then centre the box — otherwise the
  // bullet markers pin to the far-left margin while the text centres (detached markers).
  #box[
    #set align(left)
    - Copy a layout from this gallery; don't re-derive it.
    - Check the page count after every edit (D9) — overflow paginates silently.
    - Numbers and figures come from the run, never the keyboard.
  ]
  #v(1em)
  #text(size: 17pt, fill: muted)[Every layout, one demolab look.]
]
