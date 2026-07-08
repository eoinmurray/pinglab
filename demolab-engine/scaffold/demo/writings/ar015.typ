#let meta = (
  title: "HOUSESTYLE: writing style",
  date: "2026-07-07",
  description: "How a demolab write-up should read: prose, math, figures, captions, and citations, and how to override the defaults per lab.",
  collection: "documentation",
  status: "final",
)

#let src = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/guides/HOUSESTYLE.md"

#let body = [
  HOUSESTYLE is the taste demolab writes with, so the whole lab reads as one voice. It is *style*, softer than the hard invariants, and you follow it unless you have a reason not to. If you type `HOUSESTYLE` in capitals, the coding agent will walk you through it interactively.

  == How the prose reads

  A write-up is written for a scientist, not a changelog. The title is a plain sentence about the science, never prefixed with the entry id or date, since those already sit in the meta strip. Lead with the claim in plain language, then let the figure and the numbers carry the evidence, and keep the motivation short. Report results honestly, including partial and negative ones, folded into the prose or a caption rather than hidden. One notable habit: no em-dashes in running prose. They are the loudest tell of machine-written text, so a comma, colon, parentheses, or full stop stands in.

  == How math is set

  Equations are written in native Typst so they render as selectable math on the web and typeset in the PDF, never pasted as images. The single most important rule: after an equation, define _every_ symbol in a list. A reader should never have to guess what a term means. For "approximately" use `≈` in prose and `approx` in math, never a tilde, because in Typst a tilde is a non-breaking space and quietly vanishes.

  == Figures, captions, and numbers

  Every figure is drawn by a runner from the tool's data and embedded in both the web page and the PDF from one rendered asset, so the plot style lives in one shared place instead of being reset per figure. Line plots go to vector SVG, dense or photographic content to high-dpi PNG, always on a white background with alt text. Ink is near-black by default; a single accent colour is earned only to separate traces sharing an axis. Captions are written to be read cold, standing on their own: a lead clause saying what the figure shows, a body naming each axis with units and defining every symbol, then one honest takeaway.

  Numbers a run produced are never hand-typed, in the prose or in a caption. They come from the run's recorded output, so a stated result always traces back to the computation and cannot drift. Likewise, citations use the citation helpers rather than typed brackets, so numbering, links, and hover-popovers stay correct as you reorder.

  == Making it your own

  This is demolab's _default_ style, and you can override it. Drop a `HOUSESTYLE.local.md` in the repo root: by default your rules add to and supersede these, or you can tell it to replace them entirely and inherit none of demolab's taste. Either way the underlying tool-to-experiment contract still holds; only the style is yours to bend.

  The full reference lives in #link(src)[HOUSESTYLE.md].
]
