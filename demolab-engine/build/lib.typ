// Shared publishing helpers for the demolab Typst bundle.
// Imported (root-relative) by main.typ and by each writings/<id>.typ.
// The bundle emits three targets from one compile: web HTML, per-entry PDFs, and a book.

// --- brand defaults: overridden by the optional root demolab.yaml (see main.typ) ---
// The engine never hard-codes the site name; the resolved `brand` is threaded in from
// main.typ (which merges demolab.yaml over these defaults). Page functions default to
// these so they still render if called without a brand.
#let default-brand = (
  name: "Demolab",
  book-title: "Demolab — the book",
  contents-title: "Demolab — contents",
  description: none, // one-line site tagline shown under the homepage title (set in demolab.yaml)
  author: none,      // the lab's owner — shown as a byline on the homepage + <meta name="author">
  contact: none,     // optional email/url; if set, the byline links to it (mailto for an @, else href)
)

// Root-relative path to a run artifact under artifacts/data/. In a normal lab content-prefix is
// empty and this is `/artifacts/data/<rel>`; dev:demo-site passes content-prefix so the demo can
// live under demolab-engine/scaffold/demo/ while Typst --root stays at the repo checkout.
#let content-prefix = sys.inputs.at("content-prefix", default: "")
#let data-file(rel) = content-prefix + "/artifacts/data/" + rel

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

// --- web-styles: inject the stylesheet + head meta into HTML pages (ignored in the PDF pass) ---
#let web-styles(brand: default-brand) = context {
  if target() == "html" {
    html.elem("link", attrs: (rel: "icon", type: "image/svg+xml", href: "favicon.svg"))
    html.elem("style", read("/demolab-engine/build/style.css"))
    // provenance: which engine built this page (invisible, machine-readable)
    html.elem("meta", attrs: (name: "generator", content: "demolab " + read("/demolab-engine/VERSION").trim()))
    if brand.at("author", default: none) != none {
      html.elem("meta", attrs: (name: "author", content: brand.author))
    }
    // hover popovers for inline citations (no-op on pages without cites)
    html.elem("script", attrs: (src: "cite-popover.js", defer: ""))[]
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

// --- pending: a placeholder for a figure whose asset isn't ready yet (a re-run in flight, data
// withheld). Drops into a #figure in place of the image, so the figure still numbers and captions
// normally, and reserves the figure's footprint (default 16:9, H12) so the page doesn't reflow when
// the real plot lands. A tinted dashed panel with a small framed-image mark over the muted reason.
// `pending-figure(...)` is the one-call convenience that guarantees continuous "Figure N" numbering.
#let pending(body, ratio: 16 / 9) = context {
  if target() == "html" {
    html.elem(
      "div",
      attrs: (class: "fig-pending", style: "aspect-ratio:" + str(ratio)),
      {
        html.elem("span", attrs: (class: "fig-pending-mark"))[]
        html.elem("span", attrs: (class: "fig-pending-note"), body)
      },
    )
  } else {
    layout(size => block(
      width: 100%,
      height: size.width / ratio,
      fill: luma(249),
      radius: 5pt,
      stroke: (thickness: 0.75pt, paint: luma(203), dash: "dashed"),
      inset: 1.1em,
      align(center + horizon, grid(
        rows: (auto, auto),
        row-gutter: 0.7em,
        align: center,
        box(width: 1.7em, height: 1.2em, radius: 1.5pt, stroke: 0.7pt + luma(178)),
        text(size: 9pt, fill: luma(140), style: "italic", body),
      )),
    ))
  }
}

// A whole pending figure in one call: numbers as a "Figure N" (kind: image) alongside real figures.
#let pending-figure(caption: none, note: [figure pending], ratio: 16 / 9) = figure(
  pending(note, ratio: ratio),
  caption: caption,
  kind: image,
  supplement: [Figure],
)

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

// --- citations: inline numbered cites + a DOI reference list ---
// Author-managed numbering (you pass the numbers), so it's dependency-free and works the same
// in the bundle's HTML and PDF. `#cite(1, 2)` renders "[1, 2]" (links to the refs on the web);
// `#reference-list((( text: "...", doi: "..." ), …))` renders the numbered References section,
// each entry linking out to https://doi.org/<doi>. On the web the inline cites jump to the entry.
#let cite(..ns) = {
  let nums = ns.pos()
  context {
    if target() == "html" {
      html.elem("span", attrs: (class: "cite"), {
        [\[]
        nums.map(n => html.elem("a", attrs: (href: "#ref-" + str(n)), str(n))).join(", ")
        [\]]
      })
    } else [#h(0.15em, weak: true)#text(weight: 600)[\[#nums.map(n => str(n)).join(", ")\]]]
  }
}

#let reference-list(items) = {
  heading(level: 2, "References")
  context {
    if target() == "html" {
      html.elem("ol", attrs: (class: "refs"), {
        for (i, r) in items.enumerate() {
          html.elem("li", attrs: (id: "ref-" + str(i + 1)), {
            r.text
            if r.at("doi", default: none) != none {
              [ ]
              html.elem("a", attrs: (class: "doi", href: "https://doi.org/" + r.doi, target: "_blank", rel: "noopener"), "doi:" + r.doi)
            }
          })
        }
      })
    } else {
      enum(..items.map(r => [
        #r.text#if r.at("doi", default: none) != none [ #link("https://doi.org/" + r.doi)[doi:#r.doi]]
      ]))
    }
  }
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

// --- status-badge: an entry's lifecycle marker, as plain text ---
// Set `status:` in a writing's meta. "final" (the default) shows nothing — a clean line is
// "done" — so only work-in-progress is flagged. Just the word, no styling; it sits in the
// meta line and inherits its look.
#let status-badge(status) = {
  let s = if status == none { "final" } else { status }
  if s != "final" {
    let disp = upper(s.slice(0, 1)) + s.slice(1)  // capitalised for display; sorting uses the raw value
    context {
      if target() == "html" { html.elem("span", attrs: (class: "status"), disp) } else { text(disp) }
    }
  }
}

// Lifecycle order for sorting: final → revising → building → draft. Unknown (free-form) values
// sort mid-lifecycle. Used by grouped-entry-lists so settled work leads and work-in-progress trails.
#let status-rank(s) = {
  let i = ("final", "revising", "building", "draft").position(x => x == s)
  if i == none { 2 } else { i }
}

// Flatten entries + decks into one uniform list of link rows. Decks (paged-only) link to
// their PDF and are forced into the `slides` collection; entries link to their HTML page.
#let collect-items(entries, decks) = {
  entries.map(e => (
    id: e.id,
    kind: e.kind, // "experiment" | "article"
    title: e.meta.title,
    date: e.meta.date,
    status: e.meta.at("status", default: "final"),
    coll: e.meta.at("collection", default: "uncategorized"),
    href: e.id + ".html",
    pdf: "pdfs/" + e.id + ".pdf",
    deck: false,
  )) + decks.map(d => (
    id: d.id,
    kind: "deck",
    title: d.meta.title,
    date: d.meta.date,
    status: d.meta.at("status", default: "final"),
    coll: "slides",
    href: "pdfs/" + d.id + ".pdf",
    pdf: "pdfs/" + d.id + ".pdf",
    deck: true,
  ))
}

// A list of link rows, shared by the homepage and the all-entries index. On the web each row
// stacks: mono entry-id + title on top, then a quiet meta sub-line (date · collection? · status? ·
// pdf) beneath the title — so long titles wrap cleanly without orphaning the meta. In the PDF the
// same rows stay as a plain prose bullet list.
#let entry-list(items, show-collection: false, collection-meta: (:)) = context {
  if target() == "html" {
    html.elem("ul", attrs: (class: "entry-list"), {
      for it in items {
        html.elem("li", attrs: (class: "entry-row"), {
          html.elem("span", attrs: (class: "row-id"), it.id)
          html.elem("div", attrs: (class: "row-main"), {
            html.elem("a", attrs: (class: "row-title", href: it.href), it.title)
            html.elem("div", attrs: (class: "row-meta"), {
              human-date(it.date)
              if show-collection [ · #collection-label(it.coll, collection-meta)]
              if it.status != "final" [ · #status-badge(it.status)]
              [ · ]
              html.elem("a", attrs: (class: "row-pdf", href: it.pdf), "pdf")
            })
          })
        })
      }
    })
  } else {
    for it in items {
      [- #link(it.href, it.title) \
        #text(fill: gray, size: 9pt)[#human-date(it.date)#if show-collection [ · #collection-label(it.coll, collection-meta)]#if it.status != "final" [ · #status-badge(it.status)] · #link(it.pdf)[pdf]]]
    }
  }
}

// Render items grouped by kind — Articles, then Experiments, then Slides — each a level-2
// section (empty groups dropped). Shared by the all-entries page and each collection page.
// Group order: Articles, then Experiments, then Slides. Within a group, rows sort by **status**
// (lifecycle: final first, draft last — settled work leads) then by **id** descending
// (newest first). A stable two-pass gives the id-desc tiebreak within each status.
#let grouped-entry-lists(items, show-collection: false, collection-meta: (:)) = {
  let groups = (("article", "Articles"), ("experiment", "Experiments"), ("deck", "Slides"))
  for (k, title) in groups {
    let g = items.filter(x => x.kind == k)
      .sorted(key: x => x.id).rev()               // id descending
      .sorted(key: x => status-rank(x.status))     // status ascending, stable → id-desc kept within a status
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

// --- heading anchors: a slug id on every heading so any section is deep-linkable (page.html#slug) ---
// to-string flattens a heading's content to plain text; slugify lowercases it to a-z0-9-hyphens.
#let to-string(c) = {
  if c == none { "" }
  else if type(c) == str { c }
  else if c.has("text") { c.text }
  else if c.has("children") { c.children.map(to-string).join() }
  else if c.has("body") { to-string(c.body) }
  else if c == [ ] { " " }
  else { "" }
}
#let slugify(s) = lower(s).replace(regex("[^a-z0-9]+"), "-").trim("-")

// --- numbered-pages: centered footer page numbers for the paged (PDF) documents ---
// main.typ wraps each entry PDF and the book in this. Not applied to the web pages
// (nothing to number) or to decks (slides stay unnumbered).
#let numbered-pages(body) = {
  set page(numbering: "1", number-align: center)
  body
}

// --- page templates (one per output document) ---

// --- broken-entry-page: a visible stub for an entry that failed to build (a missing figure, a
// Typst error). build.py flags it after a failed compile and main.typ renders this instead of
// importing the entry, so one bad page fails on its own rather than taking down the whole site.
// HTML only (main.typ emits no PDF for a stub). ---
#let broken-entry-page(id, error, brand: default-brand) = {
  web-styles(brand: brand)
  set text(font: "New Computer Modern", size: 11pt)
  heading(level: 1, id)
  html.elem("p", attrs: (class: "entry-meta"), [This entry failed to build, so it's a stub. The rest of the site built normally; fix the error below and rebuild to bring it back.])
  html.elem("pre", attrs: (class: "build-error"), error)
  html.elem("p", attrs: (class: "page-foot"), html.elem("a", attrs: (href: "index.html"), [← back to all entries]))
}

#let entry-page(meta, body, id: none, brand: default-brand) = {
  web-styles(brand: brand)
  set text(font: "New Computer Modern", size: 11pt)
  set par(justify: true)
  // Restart figure numbering per entry: the whole bundle is one compile, so Typst's global
  // figure counter would otherwise carry across every document. Each entry (its page + its
  // standalone PDF) numbers its own figures from 1.
  counter(figure.where(kind: image)).update(0)
  // Left-align figure captions. In the PDF, align() does it; in HTML, style.css's
  // figcaption rule does — so the align (a paged-only fn) never runs during HTML export.
  show figure.caption: it => context { if target() == "html" { it } else { align(left, it) } }
  // outline() queries headings across the whole bundle; keep per-entry docs out of
  // the book's table of contents.
  set heading(outlined: false)
  // Heading anchors (web only): re-emit each heading with a slug id + a quiet permalink that
  // appears on hover, so any section is directly linkable (page.html#slug). The PDF/book keep
  // native headings. Typst maps a level-N heading to <h(N+1)>, so match that tag.
  show heading: it => context {
    if target() != "html" { it } else {
      let id = slugify(to-string(it.body))
      if id == "" { it } else {
        html.elem("h" + str(calc.min(it.level + 1, 6)), attrs: (id: id, class: "hx"), {
          it.body
          html.elem("a", attrs: (class: "permalink", href: "#" + id, "aria-label": "Link to this section"), "#")
        })
      }
    }
  }
  heading(level: 1, meta.title)
  // the metadata strip under the title — id · date · status · pdf, all inline on the left (web
  // only; the PDF pass shows the plain gray meta line without the pdf link, since it *is* the pdf).
  let meta-bits = (
    id,
    human-date(meta.date),
  ).filter(x => x != none)
  let meta-line = meta-bits.join(" · ")
  let status = meta.at("status", default: "final")
  let pdf-href = if id != none { "pdfs/" + id + ".pdf" } else { none }
  context {
    if target() == "html" {
      html.elem("div", attrs: (class: "entry-meta"), {
        meta-line
        if status != "final" [ · #status-badge(status)]
        if pdf-href != none {
          [ · ]
          html.elem("a", attrs: (class: "entry-pdf", href: pdf-href), "pdf")
        }
      })
    } else {
      text(size: 9pt, fill: gray, { meta-line; if status != "final" [ · #status-badge(status)] })
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
// An optional `welcome` block in demolab.yaml renders a hero above the directory (pitch,
// install commands, links) — used by the upstream demo site; absent on a normal lab.
// `welcome.hide-directory: true` stops the homepage after the welcome block (no collection
// index or page foot) — an upstream-only landing-page mode, not documented in the skeleton.
#let welcome-block(welcome) = {
  if welcome == none { return }
  html.elem("div", attrs: (class: "welcome"), {
    if welcome.at("body", default: none) != none {
      for para in welcome.body.trim().split("\n\n") {
        let t = para.trim()
        if t.len() > 0 {
          html.elem("p", attrs: (class: "welcome-body"), t)
        }
      }
    }
    if welcome.at("links", default: none) != none {
      html.elem("p", attrs: (class: "welcome-links"), {
        for (i, l) in welcome.links.enumerate() {
          if i > 0 { [ · ] }
          link(l.at("href", default: ""), l.at("label", default: ""))
        }
      })
    }
    let inst = welcome.at("install", default: none)
    if inst != none {
      if inst.at("label", default: none) != none {
        html.elem("p", attrs: (class: "welcome-kicker"), inst.label)
      }
      if inst.at("unix", default: none) != none {
        html.elem("div", attrs: (class: "welcome-cmd"), {
          html.elem("div", attrs: (class: "welcome-os"), [macOS / Linux])
          html.elem("pre", inst.unix)
        })
      }
      if inst.at("windows", default: none) != none {
        html.elem("div", attrs: (class: "welcome-cmd"), {
          html.elem("div", attrs: (class: "welcome-os"), [Windows])
          html.elem("pre", inst.windows)
        })
      }
    }
    let prompt = welcome.at("prompt", default: none)
    if prompt != none {
      if prompt.at("label", default: none) != none {
        html.elem("p", attrs: (class: "welcome-kicker"), prompt.label)
      }
      if prompt.at("text", default: none) != none {
        html.elem("div", attrs: (class: "welcome-cmd"), {
          html.elem("pre", prompt.text)
        })
      }
    }
    let foot = welcome.at("footer", default: none)
    if foot != none {
      html.elem("p", attrs: (class: "welcome-foot"), {
        if type(foot) == dictionary and foot.at("href", default: none) != none {
          link(foot.href, foot.at("text", default: foot.href))
        } else {
          foot
        }
      })
    }
  })
}

#let index-page(entries, decks: (), brand: default-brand, collection-order: (), collection-meta: (:), welcome: none) = {
  web-styles(brand: brand)
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
    // Byline: the lab's owner under the title. Links to contact if given (mailto for an email).
    if brand.at("author", default: none) != none {
      let c = brand.at("contact", default: none)
      html.elem("p", attrs: (class: "byline"), {
        [by ]
        if c != none {
          link(if "@" in c and not c.starts-with("http") { "mailto:" + c } else { c }, brand.author)
        } else { brand.author }
      })
    }
    if items.len() == 0 {
      // Freshly-scaffolded repo — no writings yet. This is the first thing a (often non-technical)
      // user sees, so keep it warm and jargon-free: point them at their coding agent, not at file
      // paths and task commands (those live in the README for anyone doing it by hand).
      html.elem("div", attrs: (class: "empty-state"), {
        html.elem("p", attrs: (class: "empty-lead"), [Your lab is ready.])
        html.elem("p", [
          Nothing is published here yet. This is where your experiments and writeups will appear.
          Ask your coding agent to help you set up your first one — or to load a worked example, so
          you can see how it all fits together.
        ])
      })
    } else {
      welcome-block(welcome)
      let hide-dir = welcome != none and welcome.at("hide-directory", default: false)
      if not hide-dir {
        collection-index(colls, collection-meta)
        html.elem("p", attrs: (class: "page-foot"), {
          link("all.html", "Browse all entries")
          [ · also available as a ]
          link("pdfs/book.pdf", "single pdf")
          [.]
        })
      }
    }
  })
}

// A per-collection page: the collection's label + description, then its entries grouped by
// kind (Articles / Experiments / Slides), the same organisation as the all-entries page.
// Reached from the homepage directory; the foot link returns there.
#let collection-page(coll, items, brand: default-brand, collection-meta: (:)) = {
  web-styles(brand: brand)
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
  web-styles(brand: brand)
  set text(font: "New Computer Modern", size: 11pt)
  set heading(outlined: false)
  let items = collect-items(entries, decks)
  html.elem("div", attrs: (class: "listing"), {
    heading(level: 1, [All entries])
    grouped-entry-lists(items)
    html.elem("p", attrs: (class: "page-foot"), link("index.html", "← grouped by collection"))
  })
}

#let book-page(entries, brand: default-brand) = {
  set text(font: "New Computer Modern", size: 11pt)
  set par(justify: true)
  show figure.caption: set align(left) // left-align captions (book is PDF-only)
  // The book is emitted after every entry document in the same compile, so reset the global
  // figure counter here too — the book then numbers figures continuously 1…N across all chapters.
  counter(figure.where(kind: image)).update(0)
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
