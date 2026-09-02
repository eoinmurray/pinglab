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
      + ".pinglab-result-sections > h4::before, .pinglab-result-sections > article > h4:first-child::before { content: counter(pinglab-result) \". \"; } "
      + ".pinglab-result-card { margin: 1.25rem 0; padding: 1.2rem 1.35rem 1.3rem; border: 1px solid var(--rule-strong); border-radius: 3px; background: var(--paper); } "
      + ".pinglab-result-card > h4:first-child { margin-top: 0; } "
      + ".pinglab-result-card > :last-child { margin-bottom: 0; } "
      + "@media (max-width: 520px) { .pinglab-result-card { margin: 1rem 0; padding: .95rem 1rem 1.05rem; } }",
    )
    html.elem("section", attrs: (class: "pinglab-result-sections"), body)
  } else {
    [
      #set heading(numbering: result-numbering)
      #body
    ]
  }
}

#let result-card(body) = context {
  if target() == "html" {
    html.elem("article", attrs: (class: "pinglab-result-card"), body)
  } else {
    body
  }
}

#let with-numbered-equations(body) = [
  #set math.equation(numbering: "(1)")
  #counter(math.equation).update(0)
  #show math.equation.where(block: true): equation => context {
    if target() == "html" {
      html.elem("div", attrs: (
        class: "pinglab-numbered-equation",
        style: "display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:1em",
      ), {
        html.elem("div", attrs: (style: "min-width:0;overflow-x:auto;overflow-y:hidden"), equation)
        html.elem(
          "span",
          attrs: (class: "pinglab-equation-number"),
          numbering(equation.numbering, ..counter(math.equation).at(equation.location())),
        )
      })
    } else {
      equation
    }
  }
  #body
]

#let toc-list(items, spacing: 0.25em) = {
  list(tight: true, spacing: spacing, ..items)
}

#let toc-enum(items, spacing: 0.25em) = {
  enum(tight: true, spacing: spacing, numbering: "1.", ..items)
}

#let contents-here() = metadata("pinglab-contents-here")

#let render-contents(sections, end-location) = {
  let entries = sections.enumerate().map(((index, section)) => {
    let entry = link(section.location(), section.body)
    if lower(heading-text(section.body)).trim() == "results" {
      let children = if index + 1 < sections.len() {
        query(heading.where(level: 3).after(section.location()).before(sections.at(index + 1).location()))
      } else {
        query(heading.where(level: 3).after(section.location()).before(end-location))
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
