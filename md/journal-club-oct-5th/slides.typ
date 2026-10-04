// Compile from the repository root (the shared template uses root-relative imports):
// typst compile --root . md/journal-club-oct-5th/slides.typ md/journal-club-oct-5th/slides.pdf

#import "../../writings/templates/references.typ": journal-references

#set document(title: "Journal club — 5 October 2026")
#set page(
  width: 320mm,
  height: 180mm,
  margin: (x: 22mm, y: 16mm),
)
#set text(font: "Helvetica", size: 15pt, fill: rgb("242424"))
#set par(leading: 0.4em, spacing: 0pt)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 32pt)
#show heading.where(level: 2): it => block(above: 0pt, below: 16pt)[
  #text(size: 15pt, weight: "bold")[#it.body]
]
#set enum(indent: 0pt, body-indent: 12pt, spacing: 24pt)
#show link: set text(fill: rgb("242424"))

#let references = (
  (
    text: [
      #text(size: 20pt, weight: "bold")[Input-Dependent Frequency Modulation of Cortical Gamma Oscillations Shapes Spatial Synchronization and Enables Phase Coding]
      #v(6pt)
      Eric Lowet, Mark Roberts, Avgis Hadjipapas, Alina Peter, Jan van der Eerden & Peter De Weerd.
      #v(4pt)
      _PLOS Computational Biology_ 11(2), e1004072 (2015).
    ],
    doi: "10.1371/journal.pcbi.1004072",
  ),
  (
    text: [
      #text(size: 20pt, weight: "bold")[Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference]
      #v(6pt)
      Rodrigo Echeveste, Laurence Aitchison, Guillaume Hennequin & Máté Lengyel.
      #v(4pt)
      _Nature Neuroscience_ 23, 1138–1149 (2020).
    ],
    doi: "10.1038/s41593-020-0671-1",
  ),
  (
    text: [
      #text(size: 20pt, weight: "bold")[Neural basis of social hierarchy across species]
      #v(6pt)
      Rongzhen Yan & Dayu Lin.
      #v(4pt)
      _Nature Reviews Neuroscience_ 27, 494–512 (2026).
    ],
    doi: "10.1038/s41583-026-01047-z",
  ),
)

#align(horizon)[
  #grid(
    columns: (1fr, auto),
    align: bottom,
    text(size: 32pt, weight: "bold")[Journal club],
    text(size: 16pt)[5 October 2026],
  )
  #v(12pt)
  #line(length: 100%, stroke: 0.4pt + rgb("242424"))
  #v(18pt)
  #journal-references(references)
]
