// Shared displayed-equation numbering contract.
//
// Apply `with-numbered-equations(body)` once to the complete article body. It
// resets the article-wide counter and numbers every displayed equation
// continuously as `(1)`, `(2)`, and so forth in HTML and PDF. Inline mathematics
// remains unnumbered. Do not add local counter resets or individually assigned
// display numbers around this component. Displaying an equation does not require
// citing it in prose, and numbering does not justify displaying trivial
// arithmetic.

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
