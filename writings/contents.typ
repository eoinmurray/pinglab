// Article-scoped navigation; never query headings from neighbouring book entries.
// https://typst.app/docs/reference/introspection/query/
#let with-contents(body) = [
  #set heading(numbering: none)
  #heading(level: 2, outlined: false)[Table of Contents]
  #context {
    let ends = query(metadata.where(value: "pinglab-contents-end").after(here()))
    if ends.len() > 0 {
      let sections = query(heading.where(level: 2).after(here()).before(ends.first().location()))
      let entries = sections.map(section => link(section.location(), section.body))
      if target() == "html" {
        html.elem("nav", attrs: ("aria-label": "Table of Contents"), list(..entries))
      } else {
        list(..entries)
      }
    }
  }
  #body
  #metadata("pinglab-contents-end")
]
