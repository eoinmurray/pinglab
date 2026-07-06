#let meta = (
  title: "Guides",
  date: "2026-07-06",
  description: "The always-on reference that defines how a demolab lab works: the rules, the house style, the structure, the vocabulary. Each guide links to its source on GitHub, never repeated here.",
  collection: "documentation",
)

#let base = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/guides"

#let body = [
  Guides are demolab's *reference layer*: the conventions an agent, and you, can lean on at any
  moment. They live inside the engine (`demolab-engine/guides/`), so they update when the engine
  does, and they are always in force rather than run on demand. Type a guide's name in capitals
  (`RULES`, `SLIDES`) and the agent walks you through it.

  Each entry below links to its source on GitHub, so this page can never drift from the guide it
  points at.

  - #link(base + "/RULES.md")[*RULES*]: the contract. The toolchain, the framework/content firewall, the tool and experiment schema, and how to add a tool, an experiment, or a writing.
  - #link(base + "/HOUSESTYLE.md")[*HOUSESTYLE*]: how a writing reads. Prose, math, figures, and captions, as numbered H-rules.
  - #link(base + "/SLIDES.md")[*SLIDES*]: deck conventions, the reusable layout catalog, and sizing.
  - #link(base + "/STRUCTURE.md")[*STRUCTURE*]: the annotated file tree.
  - #link(base + "/GLOSSARY.md")[*GLOSSARY*]: the vocabulary (tool, experiment, deck, collection, provenance).
  - #link(base + "/SUPPORT.md")[*SUPPORT*]: reaching a human.

  Their on-demand counterpart, *runbooks*, are documented in a companion article in this
  collection.
]
