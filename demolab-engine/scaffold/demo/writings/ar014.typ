#let meta = (
  title: "Authoring decks",
  date: "2026-07-07",
  description: "How to build a slide deck in demolab: a Touying PDF sized against a fixed canvas, verified for overflow, laid out from a catalog.",
  collection: "documentation",
  status: "final",
)

#let src = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/guides/SLIDES.md"

#let body = [
  A deck in demolab is a talk, not a writeup. It lives at `writings/<id>.slide.typ` and is
  built with #link("https://typst.app/universe/package/touying")[Touying] into a standalone
  PDF. The `.slide.typ` filename is the marker: the build compiles the deck on its own, links
  it from the site, and deliberately excludes it from the HTML pages and the book. Touying is
  paged-only and does not survive HTML export, so a deck stays a PDF and nothing else.

  One structural difference from a prose entry: a deck declares `#let meta` so the site can
  list and link it, but it never declares a `#let body`. Its figures come from the same
  committed run outputs the writeups read, out of `artifacts/data/`, never from ad-hoc images.

  == Size against the canvas

  A slide is a fixed rectangle: 842 by 474 points at 16:9. Once the title header and footer
  margins take their share, you have roughly 350 points of usable height on a titled slide.
  That number is the budget you lay everything out against. Size images in absolute points,
  not percentages: a percentage resolves against whatever region encloses it, which inside a
  grid cell is rarely the whole slide, so percent-sized figures come out unpredictably small
  and can overflow. Give each figure row a height that suits its shape, sum the rows, gutters,
  and captions, and confirm the total stays under budget. If a slide is mostly whitespace the
  figures are too small; if things do not fit, split the slide rather than shrinking text
  below readable sizes.

  == Check the page count

  Overflow is silent. Oversized content does not clip or raise an error; Touying quietly spills
  it onto an extra page, and the deck grows without warning. So after every layout change, count
  the pages and compare against the number of slides you meant to have. Render the deck to images
  and count them, rather than trusting cached metadata from Finder or Spotlight, which goes stale.
  Then eyeball the dense slides at higher resolution to catch clipped captions or terms swallowed
  into a subscript.

  == Lift layouts from the catalog

  You do not re-derive a layout from scratch. Every layout is a named, tested block in the
  gallery deck, and the guide carries an index of names paired with when to use each one: bullets,
  two-column comparisons, code panels, equation-with-terms, figure grids, big-number statements,
  section dividers, and more. You find the name, copy that block out of the gallery, and swap in
  your own content. The gallery holds the single page-count-checked copy of each layout, and a
  test keeps the index and the gallery in sync, so an entry can never point at a missing block.

  You can also just type `SLIDES` in capitals and the coding agent will walk you through all of
  this interactively.

  The full reference lives in #link(src)[SLIDES.md].
]
