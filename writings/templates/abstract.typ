// Reusable Abstract writing template.
//
// The four structured arguments are source-level prompts only; they are
// rendered as two unlabelled paragraphs. `body` instead wraps an existing
// authored abstract without changing its wording or paragraph structure.
// Supply complete authored sentences, including their punctuation. Keep the
// complete abstract concise (normally around 65 words), qualitative, and
// independent of repository or run bookkeeping.
//
// `question`
//   The experiment's question or purpose.
//
// `approach`
//   What was reused, changed, compared, or tested.
//
// `finding`
//   The main qualitative observation. Never substitute an expectation or plan.
//
// `scope`
//   What the finding establishes, its reuse value, or its main limitation.
//
// GROUNDING AND PROSE
//
// Before writing, read the experiment entry point and follow its execution into
// the relevant recipe, compute, analysis, presentation and helper code. Check
// completed-run provenance, retained configurations and results: code describes
// intended behaviour, while retained evidence establishes what happened. Resolve
// discrepancies and distinguish new execution from reused evidence. Never run an
// experiment merely to fill an Abstract.
//
// Use plain, direct language for a reader familiar with the wider project but
// not this experiment. Omit numerical results, sample sizes, parameter grids,
// units, citations, general background and implementation detail. Never replace
// an observation with an expectation. Apply the Writing Guide's global rules for
// evidence, provenance, tense, person and repository-independent prose.
//
// PUBLIC ENTRY POINT
//
// `journal-abstract(body: none, question: none, approach: none, finding: none,
// scope: none, status: none)` accepts either an existing authored `body` or all
// four structured roles, never a mixture. It owns the level-2 Abstract heading, two-paragraph
// structured rendering and the post-Abstract contents marker.

#import "contents.typ": contents-here
#import "status.typ": component-title

// `status` follows status.typ: none or author-assigned "locked".

#let journal-abstract(
  body: none,
  question: none,
  approach: none,
  finding: none,
  scope: none,
  status: none,
) = {
  let structured = (question, approach, finding, scope)
  assert(body != none or structured.all(value => value != none),
    message: "journal-abstract requires body or all four structured roles")
  assert(body == none or structured.all(value => value == none),
    message: "journal-abstract accepts body or structured roles, not both")

  [
    #heading(level: 2, component-title([Abstract], status: status))

    #if body != none {
      body
    } else [
      #question #approach

      #finding #scope
    ]

    #contents-here()
  ]
}
