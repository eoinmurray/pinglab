// Reusable lab-journal Results-card template (var2).
//
// CARD CONTRACT
//
// One invocation represents one direct Results subsection. It combines a
// self-contained title, exactly one `figure(...)`, and up to three short prose
// paragraphs. Cards render the first needed prose paragraph before the figure,
// then the figure and caption, followed only by any additional prose paragraphs
// and notes. A one-paragraph card therefore has no prose below its figure. The
// surrounding Results section still belongs inside
// `with-result-sections[...]`, which owns Results numbering. `contents.typ`
// independently discovers those headings for the article-scoped contents.
// The role names below are source-level prompts only; they are not rendered as
// labels. Omit unused roles instead of adding empty or generic prose.
//
// Each card must work in isolation: a reader entering at the card should be
// able to recover what was tested, what the figure displays, what happened,
// and the main evidential limit. It should also advance the article's argument.
// Integrate figure references into the claims they support, using a stable
// numbered cross-reference and the relevant panel when applicable, for example
// "the inhibitory population remained silent
// (#result-figure-ref(<fig:example>, panel: "C"))". Do not add a generic
// standalone sentence such as "The figure shows..." merely to satisfy this
// rule, and do not use layout-dependent "above" or "below".
//
// MINIMUM SUFFICIENT PROSE
//
// Default to one compact observation paragraph. Add orientation only when the
// title and caption do not establish the local question, and add expectation
// only when a genuine prediction or diagnostic distinction changes how the
// evidence should be read. Normally use no more than two prose paragraphs and
// 100 words across `orientation`, `expectation` and `observation`; this is a
// working ceiling, not a target. Exceed it only when additional prose is needed
// to interpret the evidence or state its limitation accurately. Delete scene
// setting, procedural recap, transitions, and details already carried by the
// title, figure, caption or Methods.
//
// `title`
//   A plain string containing a concise, context-free description of the
//   subject, comparison,
//   measurement, distinctive displayed condition, or supported outcome. Avoid
//   generic titles such as "Overview" or "Results plot". Do not add a manual
//   number. After whitespace is collapsed, the authored title must contain no
//   more than 38 characters, including spaces and punctuation, and no more than
//   7 words; the generated Results number is not counted. These limits keep the
//   heading on one line at the standard article width and default zoom. Use a
//   shorter title when unusually wide glyphs or multi-digit numbering require
//   it; never shrink the heading or force unbreakable overflow. Read together in
//   order, the card titles should form a compressed account of the experiment's
//   reasoning and findings.
//
// `lead` (optional, exceptional)
//   One brief orientation sentence that takes the pre-figure position ahead of
//   the ordinary prose roles, permitted only when a schematic or unfamiliar
//   diagnostic cannot be interpreted cold. Do not use it merely to announce,
//   summarize or duplicate the figure. Ordinary cards omit it; the first used
//   role among orientation, expectation and observation then occupies the
//   pre-figure position automatically.
//
// `visual`
//   Exactly one `figure(...)`, containing a retained scientific image, video,
//   or table. Its caption explains how to read the evidence: panel mappings,
//   variables, conditions, visual encodings, sampling unit, aggregation,
//   uncertainty, and illustrative, reused, or newly measured status. The caption
//   must not state or repeat the scientific finding from `observation`.
//
// `orientation` (optional)
//   Essential local context: what was tested, on what system, and why this
//   figure exists. It may connect to the preceding card by naming the specific
//   unresolved question, but never rely on "the previous result" without
//   restating that result or question. Include only the minimum Methods detail
//   needed to understand this card.
//
// `expectation` (optional)
//   An evidence-grounded prediction or the diagnostic distinction being tested.
//   State what observable pattern would support or distinguish the proposed
//   explanation. Omit this paragraph for exploratory displays rather than
//   inventing a retrospective hypothesis. Keep expectations distinct from
//   observations. For a schematic or theory diagram, describe the candidate
//   mechanism or comparison and explicitly identify the diagram as non-evidence.
//
// `observation` (optional)
//   What the displayed evidence establishes: the decisive measurement or
//   qualitative pattern, the scientific consequence when useful, and the main
//   limitation. Prefer a few discriminating values over exhaustive reporting.
//   Separate observation from interpretation; label post-hoc interpretations.
//   Match causal strength to the intervention and evidence. Omit this paragraph
//   when a diagram or exploratory figure supports no finding beyond its display.
//
// `notes` (optional)
//   A bullet list limited to selection provenance, caveats, or secondary audit
//   details. Notes must not carry the primary result or become extra Methods.
//
// DIVISION OF LABOUR
//
// - The title advertises the card's subject or supported takeaway.
// - The first needed prose paragraph supplies the context or claim before the
//   figure; an exceptional lead takes this position when indispensable.
// - The figure supplies the evidence immediately after that first paragraph.
// - The caption supplies the information needed to decode that evidence.
// - Orientation explains why the evidence is being shown.
// - Expectation records what would discriminate the possibilities.
// - Observation states what happened and what can safely be concluded.
// - Notes preserve optional audit information.
//
// Apply the deletion test. Without the prose, the figure and caption must remain
// readable. Without the caption, the prose must not become a substitute legend
// listing panels, variables, encodings, aggregation, or uncertainty. Never state
// the same finding in both prose and caption.
//
// STORY ACROSS CARDS
//
// Result-card headings are presented together in the article contents. Read only
// that heading sequence as an acceptance test: it must give a fast, progressive
// account of the experiment rather than a set of interchangeable topic labels.
// Each heading must add the next necessary step, so a reader can recover the
// system or question, baseline, tested contrast, main result, and any mechanism,
// robustness result or limit without opening the cards.
//
// Prefer this investigation sequence when the experiment supports it:
//
//   1. system or question
//   2. baseline
//   3. intervention or comparison
//   4. discriminating response or parameter range
//   5. main outcome
//   6. mechanism, robustness result, failure mode or evidential limit
//
// Omit stages that the experiment does not contain; do not manufacture figures
// to fill the sequence. A later card should name the concrete uncertainty left
// by the earlier evidence and then answer it with its own figure. Reword or
// reorder headings when the contents sequence does not progress, while keeping
// every card independently understandable. Do not add generic Results introductions,
// transitions, summaries, or placeholders merely to make the sequence explicit.
//
// FIGURE GEOMETRY
//
// Design every figure at final publication size. Unless a target journal
// overrides it, use approximately 90 mm for a single-column figure or 180 mm for
// a double-column figure, with a maximum height near 170 mm. Use 4:3 for an
// ordinary plot panel, 1:1 where equal scaling or matrix geometry matters, 3:2
// for dense time series, and wider panels for rasters or long temporal axes when
// scientifically justified. Do not impose one aspect ratio on a compound figure.
//
// Arrange compound panels in balanced scientifically meaningful rows or grids.
// Avoid empty cells and misleading hierarchy. Never stretch plots; change the
// canvas, source layout or decomposition instead. Verify final-size labels,
// legends, annotations, uncertainty marks and raster detail, using approximately
// 5–7 pt as the minimum final text range unless the target journal says otherwise.
// In HTML, static card images are capped at 48% of the viewport height so one
// figure does not consume the reading space on a laptop. Authored widths remain
// upper bounds, and the browser preserves each image's intrinsic aspect ratio.
//
// PUBLIC ENTRY POINTS
//
// - `with-result-sections(body, number-subsections: false)` owns automatic
//   Results numbering and the HTML/PDF Results container. Enable nested numbering
//   only for an explicitly designed hierarchy.
// - `result-card(body)` wraps an existing complete level-3 Results subsection
//   without changing its authored structure.
// - `result-figure-ref(target, panel: none)` renders a stable compact reference
//   such as "Fig. 1" or "Fig. 1C" from a figure label.
// - `journal-result-card(title:, visual:, lead: none, orientation: none,
//   expectation: none, observation: none, notes: none, status: none)` owns one structured
//   level-3 subsection, renders its first supplied prose paragraph before the
//   figure and only additional prose paragraphs after it, and enforces the title
//   limits above.

#import "status.typ": component-title

// Structured cards accept `status: none` or "locked"; see status.typ.
#let result-title-limits = (characters: 38, words: 7)

#let result-figure-ref(target, panel: none) = {
  let numbered = ref(target, supplement: [Fig.])
  if panel == none { numbered } else { [#numbered#panel] }
}

#let checked-result-title(title) = {
  assert(type(title) == str, message: "result-card title must be a plain string")
  let text = title.trim().replace(regex("\\s+"), " ")
  let words = if text == "" { 0 } else { text.split(" ").len() }
  assert(
    text.len() <= result-title-limits.characters,
    message: "result-card title exceeds the 38-character limit: " + repr(text),
  )
  assert(
    words <= result-title-limits.words,
    message: "result-card title exceeds the 7-word limit: " + repr(text),
  )
  title
}

#let result-numbering(number-subsections: false, ..numbers) = {
  let values = numbers.pos()
  if values.len() == 3 { numbering("1.", values.last()) }
  else if number-subsections and values.len() == 4 {
    numbering("1.1", values.at(2), values.at(3))
  }
}

#let with-result-sections(body, number-subsections: false) = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-result-sections { counter-reset: pinglab-result; } "
      + ".pinglab-result-sections > h4, .pinglab-result-sections > article > h4:first-child { counter-increment: pinglab-result; counter-reset: pinglab-result-subsection; } "
      + ".pinglab-result-sections > h4::before, .pinglab-result-sections > article > h4:first-child::before { content: counter(pinglab-result) \". \"; } "
      + if number-subsections {
        (
          ".pinglab-result-sections > h5 { counter-increment: pinglab-result-subsection; } "
          + ".pinglab-result-sections > h5::before { content: counter(pinglab-result) \".\" counter(pinglab-result-subsection) \" \"; } "
        )
      } else { "" }
      + ".pinglab-result-card { margin: 1.25rem 0; padding: 1.2rem 1.35rem 1.3rem; background: var(--paper); } "
      + ".pinglab-result-card > h4:first-child { margin-top: 0; } "
      + ".pinglab-result-card figure img { max-height: 48vh; object-fit: contain; } "
      + ".pinglab-result-card > :last-child { margin-bottom: 0; } "
      + "@media (max-width: 520px) { .pinglab-result-card { margin: 1rem 0; padding: .95rem 1rem 1.05rem; } }",
    )
    html.elem("section", attrs: (class: "pinglab-result-sections"), body)
  } else {
    [
      #set heading(numbering: result-numbering.with(number-subsections: number-subsections))
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

#let journal-result-card(
  title: none,
  visual: none,
  lead: none,
  orientation: none,
  expectation: none,
  observation: none,
  notes: none,
  status: none,
) = {
  assert(title != none, message: "journal-result-card requires a title")
  assert(visual != none, message: "journal-result-card requires one figure as visual")
  let title = checked-result-title(title)

  result-card([
    #heading(level: 3, component-title(title, status: status))

    #if lead != none {
      lead
      visual
      if orientation != none { orientation }
      if expectation != none { expectation }
      if observation != none { observation }
    } else if orientation != none {
      orientation
      visual
      if expectation != none { expectation }
      if observation != none { observation }
    } else if expectation != none {
      expectation
      visual
      if observation != none { observation }
    } else if observation != none {
      observation
      visual
    } else {
      visual
    }

    #if notes != none [
      *Notes.*
      #notes
    ]
  ])
}
