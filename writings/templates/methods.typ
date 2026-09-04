// Reusable Methods writing template.
//
// Methods is a sequence of cards. Each card contains one scientific operation
// or one existing authored content block. Structured `method-card(...)` values
// become continuously numbered cards; authored content preserves its existing
// lists and numbering. The template owns the Methods heading, with no required
// stage groups or subheadings.
//
// `orientation` (optional)
//   A short introduction that adds information not repeated by the first step.
//
// `body`
//   The scientific procedure in dependency order, as structured steps or prose.
//
// GROUNDING AND PROSE
//
// Explain how the experiment was actually performed and how its reported
// measurements and reusable outputs were obtained. Write for a computational-
// neuroscience colleague who understands the field but not this experiment.
// Use the shortest account that remains scientifically reproducible. There is
// no minimum length. Use compact numbered operations
// and add or lengthen operations only when a distinct procedure, definition,
// decision or limitation is needed. Every sentence must be necessary to reproduce
// or interpret the experiment; if deleting it loses neither, delete it. There is
// no target length to fill and no permission to pad a complex subsection merely
// because another operation is longer.
//
// Read the execution code, scientific definitions, analysis and relevant
// helpers. Check completed-run provenance, retained configurations and outputs;
// resolve differences between current code and historical execution. Outline
// the complete procedure, distinguishing newly executed, reused and planned
// work. Never run an experiment merely to write Methods.
//
// Cover applicable starting data, models, interventions, simulation, training,
// primary measurements, selection, estimators, aggregation, comparisons,
// inferential decisions and displayed evidence. Follow actual dependencies and
// give each item one scientific purpose. Do not invent work to fill a category.
// Use direct prose: what was done, essential settings, and what it produced.
// Exclude repository bookkeeping, implementation narration and interpretation.
//
// Put exhaustive grids, initialization distributions and derivations in
// appendices, but keep essential model differences, selection criteria and
// measurement definitions here. Select only equations central to the model,
// intervention or measurement; define every symbol and unit, prefer scientific
// notation over implementation-shaped indices, and omit decorative arithmetic.
// Explain data partitions, model selection, timing, repetitions, aggregation,
// consequential evidence selection, transformations and uncertainty. Leave
// ordinary figure decoding to captions. Finish with a compression pass.
// The finished account must be understandable without code or reconstruction
// from appendices; each key equation must be understandable locally. Remove
// presentation narration and repeated captions, and flag missing evidence rather
// than inventing a procedure.
//
// PUBLIC ENTRY POINTS
//
// - `method-card(label, body)` creates one numbered scientific-operation card.
//   Pass `none` as the label when the opening sentence already carries the
//   subject.
// - `journal-methods(body:, orientation: none, status: none)` owns the complete
//   section and card container. Pass a step tuple or an authored content block
//   as `body`; a content block becomes one card.
// - Legacy `compute:`, `analysis:`/`analyse:` and `presentation:`/`present:`
//   arguments remain accepted for existing articles, concatenated in that order
//   without stage headings. Do not combine them with `body`.
// - `methods-heading(status: none)` supports separately wrapped section bodies.
// - Legacy `methods-stage(title, status: none)` emits no heading; existing
//   article content and explicit numbering remain in place.

#import "status.typ": component-title

// The section heading accepts `status: none` or "locked"; see status.typ.
#let method-card(label, body) = (
  label: label,
  body: body,
)

// Preserve existing articles' separately wrapped Methods bodies.
#let methods-heading(status: none) = heading(level: 2, component-title([Methods], status: status))
#let methods-stage(title, status: none) = {
  assert(title in ([Compute], [Analysis], [Presentation]),
    message: "methods-stage requires Compute, Analysis, or Presentation")
  assert(status in (none, "locked"), message: "status must be none or locked")
  if status == "locked" { text(size: .6em, fill: rgb("666666"))[locked] }
}

#let with-method-cards(body) = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-method-card { margin: .9rem 0; padding: 0 1.35rem; background: var(--paper); } "
      + ".pinglab-method-card > ol { margin: 0; padding-left: 1.5rem; } "
      + ".pinglab-method-card > ol > li > :first-child { margin-top: 0; } "
      + ".pinglab-method-card > ol > li > :last-child { margin-bottom: 0; } "
      + "@media (max-width: 520px) { .pinglab-method-card { margin: .8rem 0; padding: 0 1rem; } }",
    )
    html.elem("section", attrs: (class: "pinglab-method-cards"), body)
  } else {
    body
  }
}

#let render-method-card(body) = context {
  if target() == "html" {
    html.elem("article", attrs: (class: "pinglab-method-card"), body)
  } else {
    body
  }
}

#let render-methods(body, start: 1) = {
  if type(body) != array { return render-method-card(body) }
  let cards = body
  if cards.len() == 0 { return [] }
  for (offset, card) in cards.enumerate() {
    assert("label" in card and "body" in card and card.body != none,
      message: "each Methods item must be created with method-card")
    let item = if card.label == none { card.body } else { [*#card.label.* #card.body] }
    render-method-card(enum(start: start + offset, item))
  }
}

#let journal-methods(
  body: none,
  compute: none,
  analysis: none,
  presentation: none,
  analyse: none,
  present: none,
  orientation: none,
  status: none,
) = {
  assert(analysis == none or analyse == none,
    message: "use analysis, not both analysis and analyse")
  assert(presentation == none or present == none,
    message: "use presentation, not both presentation and present")
  let legacy = (
    compute,
    if analysis != none { analysis } else { analyse },
    if presentation != none { presentation } else { present },
  ).filter(part => part != none)
  assert(body == none or legacy.len() == 0,
    message: "use a Methods body or legacy arguments, not both")
  assert(body != none or legacy.len() > 0,
    message: "journal-methods requires Methods content")
  let parts = if body != none { (body,) } else { legacy }
  if parts.all(part => type(part) == array) { parts = (parts.flatten(),) }
  let step-count = parts.filter(part => type(part) == array).map(part => part.len()).sum(default: 0)
  assert(step-count <= 20, message: "journal-methods supports at most twenty steps")
  let start = 1

  [
    #methods-heading(status: status)

    #if orientation != none { orientation }

    #with-method-cards([
      #for part in parts {
        render-methods(part, start: start)
        if type(part) == array { start += part.len() }
      }
    ])
  ]
}
