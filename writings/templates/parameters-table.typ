// Article parameter-table contract.
//
// Use `parameters-table(headers, rows, columns: auto, align: left)` for a
// compact comparison of scientific parameters across conditions, reference
// systems or empirical ranges. `headers` must contain at least two content
// cells. Every row must contain exactly the same number of cells as the header.
// Pass either an integer column count or an explicit array of relative/absolute
// column widths through `columns`; the default creates equal-width columns.
// `align` accepts the same values as Typst's table alignment argument.
//
// The caller owns the surrounding heading, introduction, definitions,
// evidential qualifications, citations and interpretation. State units in the
// cells or headers, define distribution summaries, and distinguish measured
// values from model settings and broad literature comparisons. Do not use the
// component to imply that unlike quantities are directly calibrated.

#let parameters-table(headers, rows, columns: auto, align: left) = {
  assert(headers.len() >= 2, message: "a parameter table requires at least two columns")
  assert(
    rows.all(row => row.len() == headers.len()),
    message: "every parameter-table row must match the header width",
  )

  let resolved-columns = if columns == auto { headers.len() } else { columns }

  let rendered = table(
    columns: resolved-columns,
    align: align,
    inset: (x: 4pt, y: 3pt),
    table.header(
      repeat: true,
      ..headers.map(header => strong(header)),
    ),
    ..rows.flatten(),
  )

  context {
    if target() == "html" {
      html.elem("style", ".parameter-table {margin:.75rem 0 1rem;overflow-x:auto;} .parameter-table table {width:100%;min-width:760px;margin:0;border:0;border-collapse:collapse;font-size:.82rem;line-height:1.28;} .parameter-table th,.parameter-table td {padding:.4rem .5rem;text-align:left;vertical-align:top;border:0;border-bottom:1px solid var(--rule-strong,#ddd);} .parameter-table th {font-size:.78rem;line-height:1.2;} .parameter-table th:nth-child(1) {width:18%;} .parameter-table th:nth-child(2),.parameter-table th:nth-child(3) {width:20%;} .parameter-table th:nth-child(4) {width:42%;} .parameter-table tbody tr:last-child td {border-bottom:0;}")
      html.elem("div", attrs: (class: "parameter-table"), rendered)
    } else {
      block(width: 100%)[
        #set text(size: 8pt)
        #rendered
      ]
    }
  }
}
