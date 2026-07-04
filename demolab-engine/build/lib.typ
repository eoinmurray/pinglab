// Shared publishing helpers for the demolab Typst bundle.
// Imported (root-relative) by main.typ and by each writings/<id>.typ.
// The bundle emits three targets from one compile: web HTML, per-entry PDFs, and a book.

// --- brand defaults: overridden by the optional root demolab.yaml (see main.typ) ---
// The engine never hard-codes the site name; the resolved `brand` is threaded in from
// main.typ (which merges demolab.yaml over these defaults). Page functions default to
// these so they still render if called without a brand.
#let default-brand = (
  name: "demolab",
  book-title: "demolab — the book",
  contents-title: "demolab — contents",
  description: none, // one-line site tagline shown under the homepage title (set in demolab.yaml)
)

// --- human-date: render an ISO "YYYY-MM-DD" as a reader-friendly "16 June 2026" ---
// Dates are authored ISO (sortable, unambiguous); this is the form shown on the page.
// Falls back to the raw string if it isn't a well-formed ISO date.
#let human-date(iso) = {
  let p = str(iso).split("-")
  if p.len() == 3 {
    datetime(year: int(p.at(0)), month: int(p.at(1)), day: int(p.at(2))).display(
      "[day padding:none] [month repr:long] [year]",
    )
  } else { iso }
}

// --- web-styles: inject the stylesheet into HTML pages (ignored in the PDF pass) ---
#let web-styles = context {
  if target() == "html" {
    html.elem("link", attrs: (rel: "icon", type: "image/svg+xml", href: "favicon.svg"))
    html.elem("style", read("/demolab-engine/build/style.css"))
  }
}

// --- video: plays in HTML, omitted from PDF (a note points to the web edition) ---
// The mp4 is emitted as a bundle asset by build.py, referenced here by basename.
#let video(src, caption: none) = context {
  if target() == "html" {
    html.elem("video", attrs: (src: src, controls: "", style: "max-width:100%;width:640px"))[]
    if caption != none { text(size: 9pt, fill: gray)[#caption] }
  } else {
    text(
      size: 9pt,
      style: "italic",
      fill: gray,
    )[[ Video#if caption != none [ — #caption] · view the web edition to play. ]]
  }
}

// --- numbers-table: a parameter/metric table straight from a numbers.json entry ---
// so a writing's numbers come from the run and cannot drift.
#let numbers-table(entry, title: none) = {
  let cfg = entry.at("config", default: (:))
  let params = cfg.pairs().filter(p => not (p.at(0) in ("_provenance", "command")))
  let metrics = entry.pairs().filter(p => p.at(0) != "config")
  let rows = params + metrics
  block(breakable: false)[
    #if title != none [#strong(title)]
    #table(
      columns: (auto, auto),
      align: (left, right),
      table.header([*Parameter*], [*Value*]),
      ..rows.map(p => ([#raw(p.at(0))], [#p.at(1)])).flatten(),
    )
  ]
}

// --- provenance-footer: the git commit stamp the run wrote into numbers.json ---
#let provenance-footer(cfg) = {
  let prov = cfg.at("_provenance", default: none)
  if prov != none and prov.at("commit", default: none) != none {
    // separator: a CSS-styled <hr> on the web, a drawn rule in the PDF (v()/line()
    // are paged-only — Typst warns if they run during HTML export)
    context {
      if target() == "html" { html.elem("hr") } else {
        v(1.2em)
        line(length: 100%, stroke: 0.5pt + gray)
      }
    }
    text(size: 8pt, fill: gray)[
      Generated from commit #raw(prov.commit.slice(0, 7))#if prov.dirty [ (uncommitted changes)] · #human-date(prov.at("generated_at", default: "").slice(0, 10))
    ]
  }
}

// --- collections: entries are grouped on the homepage by their meta.collection ---
// A slug title-cases by default; an optional `collections` map + `collection-order` list
// in demolab.yaml (threaded in from main.typ) override the label / description / order.
#let title-case(slug) = slug.split("-").map(w => if w.len() > 0 { upper(w.slice(0, 1)) + w.slice(1) } else { w }).join(" ")
#let collection-label(slug, meta) = meta.at(slug, default: (:)).at("label", default: title-case(slug))
#let collection-description(slug, meta) = meta.at(slug, default: (:)).at("description", default: none)
#let collection-rank(slug, order) = {
  let i = order.position(s => s == slug)
  if i == none { order.len() } else { i }
}

// Flatten entries + decks into one uniform list of link rows. Decks (paged-only) link to
// their PDF and are forced into the `slides` collection; entries link to their HTML page.
#let collect-items(entries, decks) = {
  entries.map(e => (
    id: e.id,
    kind: e.kind, // "experiment" | "article"
    title: e.meta.title,
    date: e.meta.date,
    coll: e.meta.at("collection", default: "uncategorized"),
    href: e.id + ".html",
    pdf: "pdfs/" + e.id + ".pdf",
    deck: false,
  )) + decks.map(d => (
    id: d.id,
    kind: "deck",
    title: d.meta.title,
    date: d.meta.date,
    coll: "slides",
    href: "pdfs/" + d.id + ".pdf",
    pdf: "pdfs/" + d.id + ".pdf",
    deck: true,
  ))
}

// A list of link rows, shared by the homepage and the all-entries index. On the web each
// row is a flex grid — mono entry-id · title (fills) · date (right) · pdf — so ids and
// dates line up in columns (the "position of information" borrowed from pinglab). In the
// PDF the same rows stay as a plain prose bullet list. Every row ends in a linked pdf.
#let entry-list(items, show-collection: false, collection-meta: (:)) = context {
  if target() == "html" {
    html.elem("ul", attrs: (class: "entry-list"), {
      for it in items {
        html.elem("li", attrs: (class: "entry-row"), {
          html.elem("span", attrs: (class: "row-id"), it.id)
          html.elem("a", attrs: (class: "row-title", href: it.href), it.title)
          html.elem("span", attrs: (class: "row-meta"), {
            human-date(it.date)
            if show-collection [ · #collection-label(it.coll, collection-meta)]
          })
          html.elem("a", attrs: (class: "row-pdf", href: it.pdf), "pdf")
        })
      }
    })
  } else {
    for it in items {
      [- #link(it.href, it.title) #text(fill: gray, size: 9pt)[· #human-date(it.date)#if show-collection [ · #collection-label(it.coll, collection-meta)]] #link(it.pdf, text(size: 8pt)[[pdf]])]
    }
  }
}

// Render items grouped by kind — Articles, then Experiments, then Slides — each a level-2
// section (empty groups dropped), rows newest-first. Shared by the all-entries page and
// each collection page so both organise the same way.
#let grouped-entry-lists(items, show-collection: false, collection-meta: (:)) = {
  let groups = (("article", "Articles"), ("experiment", "Experiments"), ("deck", "Slides"))
  for (k, title) in groups {
    let g = items.filter(x => x.kind == k).sorted(key: x => x.date).rev()
    if g.len() > 0 {
      heading(level: 2, title)
      entry-list(g, show-collection: show-collection, collection-meta: collection-meta)
    }
  }
}

// The homepage directory: one row per collection — label (links to its page) · entry
// count · description underneath. The entries themselves live on the per-collection
// pages, mirroring pinglab's home → collection → entry drill-down.
#let collection-index(colls, collection-meta) = context {
  if target() == "html" {
    html.elem("ul", attrs: (class: "coll-list"), {
      for c in colls {
        let desc = collection-description(c, collection-meta)
        html.elem("li", attrs: (class: "coll-row"), {
          html.elem("a", attrs: (class: "coll-title", href: c + ".html"), collection-label(c, collection-meta))
          if desc != none { html.elem("p", attrs: (class: "coll-desc"), desc) }
        })
      }
    })
  } else {
    for c in colls {
      let desc = collection-description(c, collection-meta)
      [- #link(c + ".html", collection-label(c, collection-meta))]
      if desc != none { block(inset: (left: 1em), below: 0.6em, text(size: 9pt, fill: gray, desc)) }
    }
  }
}

// --- page templates (one per output document) ---

#let entry-page(meta, body, id: none, brand: default-brand) = {
  web-styles
  set text(font: "New Computer Modern", size: 11pt)
  set par(justify: true)
  // Left-align figure captions. In the PDF, align() does it; in HTML, style.css's
  // figcaption rule does — so the align (a paged-only fn) never runs during HTML export.
  show figure.caption: it => context { if target() == "html" { it } else { align(left, it) } }
  // outline() queries headings across the whole bundle; keep per-entry docs out of
  // the book's table of contents.
  set heading(outlined: false)
  heading(level: 1, meta.title)
  // the metadata strip under the title — id · date on the left, a pdf link pushed to the
  // right (web only; the PDF pass shows the plain gray meta line, since it *is* the pdf).
  let meta-bits = (
    id,
    human-date(meta.date),
  ).filter(x => x != none)
  let meta-line = meta-bits.join(" · ")
  let pdf-href = if id != none { "pdfs/" + id + ".pdf" } else { none }
  context {
    if target() == "html" {
      html.elem("div", attrs: (class: "entry-meta entry-bar"), {
        html.elem("span", meta-line)
        if pdf-href != none { html.elem("a", attrs: (class: "entry-pdf", href: pdf-href), "pdf") }
      })
    } else {
      text(size: 9pt, fill: gray, meta-line)
    }
  }
  parbreak()
  body
  // a little breathing room below the last line on the web (the PDF has page margins)
  context { if target() == "html" { html.elem("div", attrs: (class: "entry-tail")) } }
}

// The homepage: a directory of collections (decks fall under `slides`), each a link to its
// own page. Order follows demolab.yaml's `collection-order`; unlisted collections sort
// after, by first appearance. Entry rows live on the per-collection pages.
#let index-page(entries, decks: (), brand: default-brand, collection-order: (), collection-meta: (:)) = {
  web-styles
  set text(font: "New Computer Modern", size: 11pt)
  set heading(outlined: false) // keep the homepage out of the book's TOC
  let items = collect-items(entries, decks)
  let colls = items.map(it => it.coll).dedup().sorted(key: c => collection-rank(c, collection-order))
  // .listing scopes the pinglab treatment: nav/index links unadorned, underline on hover
  // only (entry-body prose keeps the default underline). The homepage leads with the same
  // title + description header a collection page uses, so the two read as siblings.
  html.elem("div", attrs: (class: "listing"), {
    heading(level: 1, brand.name)
    if brand.at("description", default: none) != none {
      html.elem("p", attrs: (class: "entry-meta"), brand.description)
    }
    collection-index(colls, collection-meta)
    html.elem("p", attrs: (class: "page-foot"), {
      link("all.html", "Browse all entries")
      [ · also available as a ]
      link("pdfs/book.pdf", "single pdf")
      [.]
    })
  })
}

// A per-collection page: the collection's label + description, then its entries grouped by
// kind (Articles / Experiments / Slides), the same organisation as the all-entries page.
// Reached from the homepage directory; the foot link returns there.
#let collection-page(coll, items, brand: default-brand, collection-meta: (:)) = {
  web-styles
  set text(font: "New Computer Modern", size: 11pt)
  set heading(outlined: false)
  let desc = collection-description(coll, collection-meta)
  html.elem("div", attrs: (class: "listing"), {
    heading(level: 1, collection-label(coll, collection-meta))
    if desc != none { html.elem("p", attrs: (class: "entry-meta"), desc) }
    grouped-entry-lists(items)
    html.elem("p", attrs: (class: "page-foot"), link("index.html", "← all collections"))
  })
}

// The flat everything index — every entry + deck, newest first, each tagged with its
// collection. Linked from the homepage.
#let all-page(entries, decks: (), brand: default-brand, collection-meta: (:)) = {
  web-styles
  set text(font: "New Computer Modern", size: 11pt)
  set heading(outlined: false)
  let items = collect-items(entries, decks)
  html.elem("div", attrs: (class: "listing"), {
    heading(level: 1, [All entries])
    grouped-entry-lists(items, show-collection: true, collection-meta: collection-meta)
    html.elem("p", attrs: (class: "page-foot"), link("index.html", "← grouped by collection"))
  })
}

#let book-page(entries, brand: default-brand) = {
  set text(font: "New Computer Modern", size: 11pt)
  set par(justify: true)
  show figure.caption: set align(left) // left-align captions (book is PDF-only)
  // Table of contents (page numbers auto-resolved from each entry's heading), no cover.
  outline(title: [#brand.contents-title], depth: 1)
  for e in entries {
    pagebreak()
    heading(level: 1, e.meta.title)
    text(size: 9pt, fill: gray)[#human-date(e.meta.date)]
    parbreak()
    e.body
  }
}
