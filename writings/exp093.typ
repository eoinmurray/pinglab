#let meta = (
  title: "Gold-star manuscript figure comparison",
  date: "2026-08-22",
  description: "Compare the legacy and current gamma-gated sparsity manuscript plots side by side.",
  collection: "gamma-gated-sparsity",
  status: "ExpScout",
)

#let comparison = json("/artifacts/data/exp093/numbers.json")

#import "ar092.typ": body as manuscript-body

#let manuscript-figures = manuscript-body.children.filter(
  item => item.func() == figure,
)

#let manuscript-figure-index = (
  exp025: 2,
  exp038: 3,
  exp049: 4,
  exp041: 5,
  exp046: 6,
  exp037: 7,
  exp042: 8,
  exp044: 9,
)

#let manuscript-caption(experiment) = {
  let index = manuscript-figure-index.at(experiment)
  manuscript-figures.at(index).caption.body
}

#let short(hash) = hash.slice(0, 12)

#let wide-html-layout() = context {
  if target() == "html" {
    html.elem(
      "style",
      "body:has(.exp093-wide) { max-width: 72em; }\n"
        + "body:has(.exp093-wide) td { overflow-wrap: anywhere; }\n"
        + "@media (max-width: 76em) {\n"
        + "  body:has(.exp093-wide) { max-width: calc(100% - 2rem); }\n"
        + "}",
    )
    html.elem(
      "div",
      attrs: (class: "exp093-wide", "aria-hidden": "true"),
    )[]
  }
}

#let body = [
  #wide-html-layout()

  This review compares the manuscript plots retained in both immutable gold-star campaigns. The left column is the historical #raw(comparison.legacy.run_id) archive; the right column is the current #raw(comparison.current.run_id) publication campaign. A _changed_ label means the rendered file bytes differ; it is not by itself a scientific judgement.

  #table(
    columns: (auto, 1fr),
    inset: 5pt,
    align: (left, left),
    [*Legacy archive*], [#comparison.legacy.uri],
    [*Legacy payload*], [#comparison.legacy.payload_digest],
    [*Current archive*], [#comparison.current.uri],
    [*Current commit*], [#raw(comparison.current.git_commit)],
    [*Current payload*], [#comparison.current.payload_digest],
  )

  #for pair in comparison.figures [
    == #pair.title

    #pair.experiment · #raw(pair.filename) · *#pair.status*

    #table(
      columns: (1fr, 1fr),
      gutter: 10pt,
      inset: 5pt,
      align: (center, center),
      [*Legacy*], [*Current*],
      [#link(pair.legacy_path)[#image(pair.legacy_path, width: 100%)]],
      [#link(pair.current_path)[#image(pair.current_path, width: 100%)]],
      [#raw("sha256:" + short(pair.legacy_sha256))],
      [#raw("sha256:" + short(pair.current_sha256))],
    )

    #manuscript-caption(pair.experiment)
  ]
]
