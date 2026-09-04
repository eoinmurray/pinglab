// Reusable References-section contract.
//
// `journal-references(entries, status: none)` renders exactly one level-2 References section
// from a nonempty tuple of reference records. Each record requires authored
// `text` and may provide a DOI through `doi`. Keep entries in order of first
// citation, pass the same tuple once, and place this component after every
// appendix as the article's final section. Inline citations remain article-owned;
// reuse the same citation number for repeated citations.
//
// The underlying Demolab helper owns the heading, ordered-list rendering,
// anchors and DOI links. Do not add a separate References heading or manually
// number entries around this component.
//
// Include only cited sources. Verify that each source supports its associated
// claim. Author each entry with authors, title, publication venue, year, and a
// DOI or stable URL where available. Keep literature references distinct from
// upstream experiment and run provenance. Apply the Writing Guide's global
// evidence and repository-independence rules.

#import "/.demolab/lib.typ": reference-list

#import "status.typ": component-title

// `status: none` or "locked" applies to the complete section; see status.typ.
#let journal-references(entries, status: none) = {
  assert(type(entries) == array and entries.len() > 0,
    message: "journal-references requires at least one reference")
  assert(entries.all(entry => "text" in entry),
    message: "each reference requires authored text")
  // Preserve the engine-owned entries, anchors and DOI links.
  for item in reference-list(entries).children {
    if item.func() == heading {
      heading(level: 2, component-title(item.body, status: status))
    } else { item }
  }
}
