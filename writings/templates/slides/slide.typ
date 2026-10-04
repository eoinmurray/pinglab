// Standalone presentation slide.
// Public entry point: slide(header: ..., body: ...).
// Both arguments accept Typst content or strings; body is required. The header is a
// 22pt bold title at the top left; the body occupies the slide's content area.
// Omit header for a title-free slide with balanced top and bottom margins.
// Every slide displays its document page number / total at the bottom right.
// Each call creates one 16:9 page. Keep the body within the available area;
// overflowing content may create additional pages. Construct images in the
// calling document so relative asset paths resolve against that document.
#let slide(header: none, body: none) = {
  assert(body != none, message: "slide requires a body")
  set text(font: "Helvetica", size: 15pt, fill: rgb("242424"))
  set par(leading: 0.4em, spacing: 0pt)
  set heading(numbering: none)
  show heading.where(level: 1): set text(size: 32pt)
  show heading.where(level: 2): it => block(above: 0pt, below: 16pt)[
    #text(size: 15pt, weight: "bold")[#it.body]
  ]
  set enum(indent: 0pt, body-indent: 12pt, spacing: 24pt)
  show link: set text(fill: rgb("242424"))
  page(
    width: 320mm,
    height: 180mm,
    margin: (x: 14mm, top: if header == none { 16mm } else { 34mm }, bottom: 16mm),
    header: if header != none { align(left, text(size: 22pt, weight: "bold", header)) },
    header-ascent: 14mm,
    footer: context align(right, text(size: 11pt, counter(page).display("1/1", both: true))),
    footer-descent: 8mm,
    body,
  )
}
