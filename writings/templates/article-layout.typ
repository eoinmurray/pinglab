// Shared article-shell contract.
//
// `journal-article(article, inputs, body, dataset-placed: false,
// dataset-status: none, dataset: true)` is the final
// wrapper for every writing. Documentation-themed collection articles pass
// `dataset: false` and omit explicit `run-view` calls. Other articles receive
// the Dataset section unless the body already placed it. The shell then applies
// article-wide equation numbering and the article-scoped contents wrapper in
// that exact order. Pass `dataset-placed:
// true` only when the body already renders the same Dataset view.
//
// Keep this shell outside data-readiness branches. When present, order complete
// level-2 sections as Abstract, Results, Methods, Dataset, appendices,
// References. Other main-text sections may appear where scientifically
// appropriate before Dataset. Reference pages and galleries do not need
// invented sections.
//
// The imported templates own their local contracts. Use `result-card.typ`
// directly for Results wrapping and cards; the article shell does not re-export
// component APIs.

#import "contents.typ": with-contents
#import "dataset.typ": with-datasets
#import "equations.typ": with-numbered-equations

// `dataset-status` forwards none or "locked" to the inserted Dataset heading.
// When dataset-placed is true, set status on the explicit run-view instead.
#let journal-article(article, inputs, body, dataset-placed: false, dataset-status: none, dataset: true) = {
  let body = if dataset {
    with-datasets(article, inputs, body, placed: dataset-placed, status: dataset-status)
  } else { body }
  let body = with-numbered-equations(body)
  with-contents(body)
}
