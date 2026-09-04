// Shared component status contract.
//
// Titled components accept `status: none` (default; no indicator) or
// `status: "locked"` (small muted text beside the title). A lock records the
// author's instruction to preserve that component's authored content. Only an
// explicit author request may change locked content or remove its lock.
// This is an editorial status, not filesystem protection or article review.
// Wrappers without a title do not have an independent status.
// `component-title` retains the plain title for contents links and inspection.

#let component-title(title, status: none) = {
  assert(status in (none, "locked"), message: "status must be none or locked")
  if status == none { return title }
  [#metadata((kind: "pinglab-component-title", title: title))#title#context {
    if target() == "html" {
      html.elem("span", attrs: (
        class: "pinglab-component-status",
        style: "margin-inline-start:.65em;font-size:.6em;font-weight:400;letter-spacing:.035em;color:var(--muted,#666);white-space:nowrap;vertical-align:baseline",
      ), [locked])
    } else {
      h(.65em)
      text(size: .6em, weight: "regular", fill: rgb("666666"))[locked]
    }
  }]
}

#let component-plain-title(body) = {
  if body.has("children") {
    for child in body.children {
      if (child.func() == metadata and type(child.value) == dictionary
        and child.value.at("kind", default: none) == "pinglab-component-title") {
        return child.value.title
      }
    }
  }
  body
}
