#let meta = (
  title: "The vocabulary",
  date: "2026-07-07",
  description: "The demolab vocabulary defined precisely: tool, experiment, writing, deck, collection, artifact, provenance, and the easy-to-confuse distinctions.",
  collection: "documentation",
  status: "final",
)

#let src = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/guides/GLOSSARY.md"

#let body = [
  demolab has a small vocabulary for its parts, and using the words precisely saves a lot of
  confusion. This page defines the ones you will meet most, with the distinctions that are easy to
  mix up called out explicitly. If you would rather be walked through it interactively, type
  `GLOSSARY` in capitals and the coding agent will take you term by term.

  == The core pieces

  - *Tool*: a small program (Python by default) holding _reusable_ science: a model or solver
    behind a small CLI that writes the contract files. It emits data, not plots, and reuse is the
    bar for making one.
  - *Experiment*: a runner plus writeup sharing an id: `experiments/exp<NNN>.py` runs a tool and
    stages results, and `writings/exp<NNN>.typ` is the writeup. A tool is _reusable_ science; an
    experiment is _one run_ of it. Do not confuse the two.
  - *Runner*: the `experiments/exp<NNN>.py` file: it runs a tool's CLI (or computes inline),
    renders figures from the data, and stages the record. It never imports a tool.
  - *Article*: a prose-only writeup, `writings/ar<NNN>.typ`, with no runner and no data pipeline.
    Contrast an experiment, which does run one.
  - *Writing*: any `writings/<id>.typ`: an experiment's writeup, an article, or a deck. It is a
    `meta` plus `body` pair the engine discovers and publishes.
  - *Deck*: a `writings/<id>.slide.typ` Touying slide deck. It is paged-only, compiled to a
    standalone PDF and grouped under the `slides` collection: listed, but not an entry.
  - *Entry*: a published item that becomes its own page, meaning an experiment or an article.
    Decks are listed but are not entries.

  == The record and its wiring

  - *Contract*: the file-based, language-neutral interface between a tool and an experiment: the
    tool writes a fixed set of files, the runner reads them by running the tool's CLI.
  - *Artifacts*: the committed record under `artifacts/`: `artifacts/data/<id>/` holds a run's
    figures and `numbers.json`, `artifacts/pdfs/` holds shareable PDFs, and `artifacts/site/` is the
    gitignored web build.
  - `numbers.json`: the aggregated, committed record of a run: each command's config plus headline
    metrics plus a provenance stamp. Writings read it so the numbers cannot drift.
  - *Provenance*: the block stamped into every run: git commit SHA, a `dirty` flag, and a UTC
    timestamp, surfaced as a page and PDF footer.
  - *Collection*: a group of entries sharing a `collection:` slug in their `meta`, grouped together
    on the homepage.

  The full reference lives in #link(src)[GLOSSARY.md].
]
