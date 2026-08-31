// Article-scoped navigation; never query headings from neighbouring book entries.
// https://typst.app/docs/reference/introspection/query/
#let heading-text(body) = {
  if body.has("text") { body.text }
  else if body.has("children") { body.children.map(heading-text).join() }
  else if body.has("body") { heading-text(body.body) }
  else if body == [ ] { " " }
  else { "" }
}

#let result-numbering(..numbers) = {
  let values = numbers.pos()
  if values.len() == 3 { numbering("1.", values.last()) }
}

#let with-result-sections(body) = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-result-sections { counter-reset: pinglab-result; } "
      + ".pinglab-result-sections > h4, .pinglab-result-sections > article > h4:first-child { counter-increment: pinglab-result; } "
      + ".pinglab-result-sections > h4::before, .pinglab-result-sections > article > h4:first-child::before { content: counter(pinglab-result) \". \"; }",
    )
    html.elem("section", attrs: (class: "pinglab-result-sections"), body)
  } else {
    [
      #set heading(numbering: result-numbering)
      #body
    ]
  }
}

#let toc-list(items, spacing: 0.25em) = {
  list(tight: true, spacing: spacing, ..items)
}

#let toc-enum(items, spacing: 0.25em) = {
  enum(tight: true, spacing: spacing, numbering: "1.", ..items)
}

#let with-contents(body) = [
  #set heading(numbering: none)
  #context {
    let ends = query(metadata.where(value: "pinglab-contents-end").after(here()))
    if ends.len() > 0 {
      let sections = query(heading.where(level: 2).after(here()).before(ends.first().location()))
      let entries = sections.enumerate().map(((index, section)) => {
        let entry = link(section.location(), section.body)
        if lower(heading-text(section.body)).trim() == "results" {
          let children = if index + 1 < sections.len() {
            query(heading.where(level: 3).after(section.location()).before(sections.at(index + 1).location()))
          } else {
            query(heading.where(level: 3).after(section.location()).before(ends.first().location()))
          }
          if children.len() > 0 {
            let nested = children.map(child => link(child.location(), child.body))
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
  }
  #body
  #metadata("pinglab-contents-end")
]
