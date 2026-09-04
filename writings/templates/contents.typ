// Shared article-scoped contents contract.
//
// Apply `with-contents(body)` once as the final navigation wrapper. It disables
// general heading numbering and renders exactly one linked contents list from
// the current article's level-2 headings. Direct level-3 Results subsections are
// included as a nested numbered list; other deeper headings and neighbouring
// book entries are excluded.
//
// In an article with an Abstract, place exactly one `contents-here()` marker
// after the complete Abstract and before the next level-2 section. With no
// Abstract, omit the marker and `with-contents` places the list first. Keep this
// wrapper outside data-readiness branches so populated and unavailable views
// both receive navigation. Never maintain a manual contents list.
// https://typst.app/docs/reference/introspection/query/

#import "status.typ": component-plain-title

#let heading-text(body) = {
  if body.has("text") { body.text }
  else if body.has("children") { body.children.map(heading-text).join() }
  else if body.has("body") { heading-text(body.body) }
  else if body == [ ] { " " }
  else { "" }
}

#let toc-list(items, spacing: 0.25em) = {
  list(tight: true, spacing: spacing, ..items)
}

#let toc-enum(items, spacing: 0.25em) = {
  enum(tight: true, spacing: spacing, numbering: "1.", ..items)
}

#let contents-here() = metadata("pinglab-contents-here")

#let render-contents(sections, end-location) = {
  let entries = sections.enumerate().map(((index, section)) => {
    let entry = link(section.location(), component-plain-title(section.body))
    if lower(heading-text(section.body)).trim() == "results" {
      let children = if index + 1 < sections.len() {
        query(heading.where(level: 3).after(section.location()).before(sections.at(index + 1).location()))
      } else {
        query(heading.where(level: 3).after(section.location()).before(end-location))
      }
      if children.len() > 0 {
        let nested = children.map(child => link(child.location(), component-plain-title(child.body)))
        [#entry #toc-enum(nested, spacing: 0.15em)]
      } else { entry }
    } else { entry }
  })
  if target() == "html" {
    html.elem("style",
      "nav[aria-label=\"Table of Contents\"] ul { margin: .35rem 0; } "
      + "nav[aria-label=\"Table of Contents\"] li > p { margin: 0; } "
      + "nav[aria-label=\"Table of Contents\"] ul ol { margin: 0; } "
      + "nav[aria-label=\"Table of Contents\"] ul ol > li:first-child { margin-top: 0; }",
    )
    html.elem("nav", attrs: ("aria-label": "Table of Contents"), toc-list(entries))
  } else {
    toc-list(entries)
  }
}

#let with-contents(body) = [
  #set heading(numbering: none)
  #context {
    let ends = query(metadata.where(value: "pinglab-contents-end").after(here()))
    if ends.len() > 0 {
      let end-location = ends.first().location()
      let sections = query(heading.where(level: 2).after(here()).before(end-location))
      let markers = query(metadata.where(value: "pinglab-contents-here").after(here()).before(end-location))
      assert(markers.len() <= 1, message: "article must contain at most one contents marker")
      if markers.len() == 1 {
        show metadata.where(value: "pinglab-contents-here"): _ => render-contents(sections, end-location)
        body
      } else {
        render-contents(sections, end-location)
        body
      }
    } else {
      body
    }
  }
  #metadata("pinglab-contents-end")
]
