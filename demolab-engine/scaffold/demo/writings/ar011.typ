#let meta = (
  title: "Getting help",
  date: "2026-07-07",
  description: "Where to get help with demolab, how to report a bug, and what to include so it's answerable fast.",
  collection: "documentation",
  status: "final",
)

#let src = "https://github.com/eoinmurray/demolab/blob/main/demolab-engine/guides/SUPPORT.md"

#let body = [
  Stuck, or found a bug? Try self-service first: the runbooks under `demolab-engine/runbooks/` and the other guides cover most operating questions. When you still need a human, here's where to go.

  == Where to ask

  - *GitHub issues*: #link("https://github.com/eoinmurray/demolab/issues")[github.com/eoinmurray/demolab/issues]. Use these for bugs, feature requests, and questions about the framework. Prefer this: issues are searchable, so your question helps the next person and the fix lands in the open.
  - *Email*: #link("mailto:em586@cam.ac.uk")[em586\@cam.ac.uk]. Use this for private matters, or if you can't or don't want to post publicly.

  == Writing a report that gets answered

  Before opening an issue, run the doctor (say _"doctor the repo"_), check the guides and runbooks, and search existing issues. Then include, in order:

  - *What you ran and what happened*: the exact command and the _full_ error output, not a paraphrase.
  - *Toolchain versions*: `uv --version`, `typst --version`, `task --version`.
  - *The commit*: demolab stamps the git SHA into the page/PDF footer, so paste that line or run `git rev-parse --short HEAD`.
  - *Framework or content?*: state whether it's the engine (a demolab bug) or your own tool/experiment/writing. Ask if you're unsure.
  - *A minimal repro* if you can: the smallest writing or tool that triggers it.

  You can also just type `SUPPORT` in capitals and the coding agent will walk you through all of this interactively.

  The full reference lives in #link(src)[SUPPORT.md].
]
