#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets
#import "run-inputs.typ": input-assets
#import "manuscript-figures.typ": figure-description
#let data-file = data-file.with(article: "exp093")

#let meta = (
  status: "[▦ DATA]",
  title: "Old and New Manuscript Figures",
  date: "2026-08-22",
  description: "Compare the legacy and current gamma-gated sparsity manuscript plots side by side.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp093", "exp025", "exp038", "exp049", "exp041", "exp046", "exp037", "exp042", "exp044")

// Evaluate historical records and figures only when every required input exists.
#let render-report(data-file) = [
#let comparison = data-json(data-file("exp093/numbers.json"))
#let figures = comparison.figures

// Use logical keys, never absolute paths embedded in a historical JSON record.
// The comparison run's legacy figures obey Pingstore's flat presentation layout.
#let comparison-image(pair, legacy: false) = {
  assert(pair.experiment in inputs,
    message: "comparison figure requires a declared experiment input")
  assert(not pair.filename.contains("/") and not pair.filename.contains("\\"),
    message: "comparison filenames must be flat presentation filenames")
  let key = if legacy {
    "exp093/legacy__" + pair.experiment + "__" + pair.filename
  } else { pair.experiment + "/" + pair.filename }
  data-image(data-file(key), width: 100%)
}

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

  Compare historical manuscript figures retained by an exp093 run with figures
  from the independently selected experiment runs. Each column follows this
  article's own selectors; neither column inherits exp092's selection.

  *Historical reference:* #raw(comparison.legacy.run_id).
  The comparison record names #raw(comparison.current.run_id) as its original
  current-side reference. That identity and its recorded differences do not
  describe a different run selected here. Authoritative provenance remains in
  each selected run's `run.json`.

  #for pair in figures [
    == #pair.title

    #pair.experiment · #raw(pair.filename)

    #table(
      columns: (1fr, 1fr),
      gutter: 10pt,
      inset: 5pt,
      align: (center, center),
      [*Historical comparison input*], [*Selected experiment input*],
      comparison-image(pair, legacy: true),
      comparison-image(pair),
    )

    #figure-description(pair.experiment)
  ]
]

#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}

#let meta = meta + (assets: input-assets("exp093", inputs))
#let body = with-datasets("exp093", inputs, report-body)
#let body = with-contents(body)
