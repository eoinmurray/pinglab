# SLIDES — authoring decks

How to create slide decks (`writings/*.slide.typ`) and lay them out. Decks are talks:
paged PDFs built with [touying](https://typst.app/universe/package/touying), linked from
the site but never rendered into HTML or the book. For prose writeups see
[`HOUSESTYLE.md`](HOUSESTYLE.md); for what a *deck* is in one line see
[`GLOSSARY.md`](GLOSSARY.md). Rules are numbered `D<n>` so other docs can cite them.

## 1. What a deck is

**D1 — The `.slide.typ` marker.** A deck lives at `writings/<id>.slide.typ`. The filename
marks it: the build compiles it standalone to `artifacts/pdfs/<id>.pdf` and links it from
the site, but excludes it from the HTML and book passes (touying is paged-only and does
not survive HTML export).

**D2 — `meta` but no `body`.** Declare `#let meta = (title: …, date: …)` so the site can
list and link the deck. Never declare `#let body` — that would make it a bundle entry.

**D3 — Compile with `--root .`.** Always from the repo root:
`typst compile --root . writings/<id>.slide.typ artifacts/pdfs/<id>.pdf`, so absolute
paths like `/artifacts/data/...` resolve. Figures come from `artifacts/data/` — the same
committed run outputs the writeups use, never ad-hoc images.

**D4 — The skeleton.**

```typst
#import "@preview/touying:0.6.1": *
#import themes.simple: *

#let meta = (
  title: "…",
  date: "YYYY-MM-DD",
)

// header: none drops touying's running section header (a small gray repeat of the title on every
// slide). Leave it off for most decks; turn it on for a long, multi-section talk where it navigates.
#show: simple-theme.with(aspect-ratio: "16-9", header: none)

#set text(font: "New Computer Modern", size: 22pt)
#show raw: set text(font: "DejaVu Sans Mono")

// Match demolab's two-ink web palette (ink for headings + bold, muted for secondary) instead of
// touying's teal accent. `strong` needs a *transform* rule — a plain `set text` on it is overridden
// by the theme. Use `#focus-slide(background: ink)` for a big statement, `background: white` for a
// chrome-free section divider.
#let ink = rgb("#1a1a1a")
#let muted = rgb("#666666")
#show heading: set text(fill: ink)
#show strong: it => text(fill: ink, weight: "bold", it.body)

// Vertically centre each slide's content — adaptive: a sparse slide centres as a balanced block;
// a nearly-full slide keeps its title near the top. Drop this line for classic top-aligned slides.
#set align(horizon)

#title-slide[
  #set align(left)  // demolab titles are left-aligned, not centred
  = Title
  #v(0.4em)
  Tagline.
]

== A content slide

- …
```

## 2. Sizing — budget the canvas in points

**D5 — The canvas is fixed: 842 × 474 pt (16:9).** A title header eats ~90 pt, footer
margin ~30 pt, leaving **~350 pt of usable height** on a titled slide. Budget layouts
against that number.

**D6 — Size images in absolute `pt`, never `%`.** Relative lengths resolve against the
enclosing region — inside grid cells that is rarely the slide, so percentage-sized images
come out unpredictably small *and* can overflow. `#image(..., height: 165pt)` does what
it says.

**D7 — Size each figure row to its aspect.** In multi-figure layouts, wide plots and tall
plots must not share one height — give each grid row its own: wide traces ~130 pt, taller
panels ~165 pt in a 2×2; a hero figure ~285 pt beside a stack of two ~135 pt. Sum rows +
gutters + captions and check it stays under ~350 pt.

**D8 — Fill or split.** If a slide is mostly whitespace, the figures are too small — grow
them to the budget or merge slides. If content doesn't fit at readable sizes, split the
slide; never shrink text below ~19 pt or captions below ~14 pt to force a fit.

## 3. Verify — overflow paginates silently

**D9 — Always check the page count after a layout change.** Oversized content does not
clip or error — touying silently spills it onto an extra page, and the deck grows without
warning. After every edit, compare pages against the intended slide count:

```sh
typst compile --root . --format png --ppi 36 writings/<id>.slide.typ "temp/check-{0p}.png"
ls temp/check-*.png | wc -l   # must equal the intended slide count
```

Do not trust Finder/Spotlight metadata (`mdls`) for the count — it caches stale values.

**D10 — Eyeball the risky slides.** Render the dense ones (figures, grids, math) at
`--ppi 96` and look: overflow, clipped captions, terms swallowed into subscripts. In
math, `$I_"ext"(t)$` puts the `(t)` in the subscript — write `$I_"ext" (t)$`.

## 4. Layout catalog

**D11 — The layouts are a named, liftable catalog.** Every layout is one named block in the gallery
deck (`writings/ar005.slide.typ`), marked `// layout: <name>`. To build a slide, find the name below,
then **copy that block out of the gallery** — from its `// layout:` marker down to the next marker —
and swap the demo content for yours. The gallery holds the one tested, page-count-checked copy; this
list is the index (name → when-to-use). Each is a `==` slide unless noted, in the order you'll reach
for them:

- **`layout: title`** — opens the deck, left-aligned (`#title-slide[#set align(left) …]`). Mirror it with the closer.
- **`layout: bullets`** — the workhorse. Bold the load-bearing phrase; two lines per bullet max; five bullets is the ceiling.
- **`layout: two-column`** — paired panels for comparisons (does/does-not, before/after). Parallel sides — same count, same shape — with a centred bold takeaway.
- **`layout: three-column`** — triads only; a fourth column wants a table. Header + one line each.
- **`layout: code-panel`** — one fenced block in a `luma(245)` rounded box, one idea per snippet; trim imports and error handling.
- **`layout: equation-terms`** — the displayed equation, then `where:` and a list defining *every* symbol.
- **`layout: quote`** — a centred italic pull-quote in a `~80%`-width block, attribution muted below.
- **`layout: section-divider`** — signpost a new part: `#focus-slide(background: white, foreground: ink)` (chrome-free + centred, no stale header) with a small kicker + a big part title.
- **`layout: centered-figure`** — one image, one muted caption; let the plot talk. A wide (2:1) plot fills the column; a 16:9 plot sizes by height and *centres* (full-width would overflow, D7).
- **`layout: figure-bullets`** — figure left (~55% width), ≤3 reading-notes right; the *so what* is the last bullet.
- **`layout: figure-pair`** — two panels labelled (a)/(b), captions under each, comparison line below.
- **`layout: figure-grid`** — 2×2, four panels max; beyond that it's a poster. Per-row heights per D7.
- **`layout: hero-stack`** — the result you're arguing for large on the left, supporting evidence stacked right.
- **`layout: diagram`** — boxes-and-arrows drawn in Typst (`box` nodes + `sym.arrow.r`), no image asset; for dataflow / architecture. One row where it fits.
- **`layout: table`** — no grid lines: `stroke: none` + a single `table.hline` under the header. Numbers from the run (`numbers.json`), never typed.
- **`layout: big-number`** — one headline metric huge and centred (`text(size: 120pt)`), a muted one-line gloss beneath; the number comes from the run.
- **`layout: big-statement`** — `#focus-slide(background: ink)[…]` for the one line they must remember. Don't bold on it — emphasis is the accent colour, which is the background (it vanishes).
- **`layout: closer`** — a centred close mirroring the title slide: parting bullets in a left-aligned `#box` (or the markers detach), tagline.

**D12 — The gallery is the source; this catalog is the index.** Don't re-derive a layout — lift the
named block from the gallery (`writings/ar005.slide.typ`) between its `// layout:` markers, then edit.
Keeping one tested copy (rather than a snippet pasted into this guide) is the same no-drift discipline
as numbers-from-the-run: `test_slide_catalog.py` asserts the `// layout:` names in the gallery match
the `` `layout: …` `` names in this catalog, so an index entry can't point at a missing block. Adding a
layout means adding both a marked gallery slide *and* a catalog entry here (the test fails otherwise).

## 5. Serving

**D13 — Decks hot-reload like entries.** The dev server rebuilds the whole bundle (via
`build.py`, which re-globs the filesystem) on every source change, so a **new** `.slide.typ`
appears and an **edited** deck reloads without restarting `task dev` — same as an ordinary
entry. A deck that fails to compile shows the Typst error as a full-screen overlay in the
browser, not just in the terminal.
