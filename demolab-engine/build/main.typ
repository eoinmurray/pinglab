// The demolab bundle root — the single source compiled to all three targets with
//   typst compile --format bundle --features bundle,html
//
// This file is COMMITTED and static: it holds no per-entry knowledge. build.py globs
// the filesystem (Typst can't list directories) and writes the discovered id/asset
// lists to temp/bundle/index.json; this file reads that manifest and does everything
// else — importing each writing, emitting every document, embedding every asset —
// in plain Typst. No generated source.
//
// Compiled with `--root` at the repo root, so `/writings/...`, `/artifacts/...`, and
// `/temp/bundle/...` all resolve. Run it by hand to debug:
//   uv run python demolab-engine/build/build.py --generate-only   # writes the manifest + decks
//   typst compile --format bundle --features bundle,html --root . demolab-engine/build/main.typ artifacts/site/
#import "/demolab-engine/build/lib.typ": *

// The manifest build.py wrote: { entries: [{id, kind, videos}], decks: [{id}],
// has_brand_config }.
#let manifest = json("/temp/bundle/index.json")

// The optional root demolab.yaml (build.py sets has_brand_config after checking it exists
// — Typst can't stat). Branding merges over engine defaults; collection label/order are
// read from it too. Absent ⇒ engine defaults + derivation-only collections.
#let config = if manifest.has_brand_config { yaml("/demolab.yaml") } else { (:) }
#let brand = default-brand + config
#let collection-order = config.at("collection-order", default: ())
#let collection-meta = config.at("collections", default: (:))

// Import each *good* writing dynamically (import paths may be computed — no literal codegen).
// An entry contributes meta + body; a deck contributes only meta (touying is paged-only, so decks
// are compiled to standalone PDFs by build.py and embedded as assets below). An entry build.py
// flagged with an `error` (a missing figure, a Typst error) is NOT imported — it would fail the
// whole compile — but rendered as a stub page below, so one bad page fails on its own.
#let entries = manifest.entries.filter(e => "error" not in e).map(e => {
  import "/writings/" + e.id + ".typ": meta, body
  (id: e.id, kind: e.kind, meta: meta, body: body)
})
#let broken = manifest.entries.filter(e => "error" in e)
#let decks = manifest.decks.map(d => {
  import "/writings/" + d.id + ".slide.typ": meta
  (id: d.id, meta: meta)
})

// --- bundle assets ---
// every mp4 an experiment produced (filenames discovered by build.py, carried in the
// manifest), referenced by basename from the writing's #video(...)
#for e in manifest.entries {
  for v in e.videos { asset(v, read("/artifacts/data/" + e.id + "/" + v, encoding: none)) }
}
// deck PDFs, embedded at pdfs/<id>.pdf so `typst watch` serves them too
#for d in manifest.decks {
  asset("pdfs/" + d.id + ".pdf", read("/temp/bundle/decks/" + d.id + ".pdf", encoding: none))
}
// site favicon (a lab-notebook mark), linked from every page's <head> by lib.typ
#asset("favicon.svg", read("/demolab-engine/build/favicon.svg", encoding: none))
// hover popovers for inline citations (web-only), referenced by lib.typ's web-styles
#asset("cite-popover.js", read("/demolab-engine/build/cite-popover.js", encoding: none))

// --- documents (one compile emits them all into artifacts/site/) ---
// The homepage always exists; on a freshly-scaffolded repo (no entries) it shows a
// friendly empty state. Everything else is emitted only when there's content.
#let all-items = collect-items(entries, decks)
#document("index.html", title: [#brand.name])[#index-page(entries, decks: decks, brand: brand, collection-order: collection-order, collection-meta: collection-meta)]
#if all-items.len() > 0 {
  [#document("all.html", title: [#brand.name — all entries])[#all-page(entries, decks: decks, brand: brand, collection-meta: collection-meta)]]
  // one page per collection (web only — the book/PDFs don't have collection pages)
  for c in all-items.map(it => it.coll).dedup() {
    [#document(c + ".html", title: [#brand.name — #collection-label(c, collection-meta)])[#collection-page(c, all-items.filter(x => x.coll == c), brand: brand, collection-meta: collection-meta)]]
  }
}
#for e in entries {
  [#document(e.id + ".html", title: [#e.meta.title])[#entry-page(e.meta, e.body, id: e.id, brand: brand)]]
  [#document("pdfs/" + e.id + ".pdf", title: [#e.meta.title])[#numbered-pages(entry-page(e.meta, e.body, id: e.id, brand: brand))]]
}
// stub pages for entries that failed to build — a visible "this page failed" placeholder at the
// entry's own URL (web only; excluded from listings + the book), so the rest of the site is fine.
#for e in broken {
  [#document(e.id + ".html", title: [#e.id])[#broken-entry-page(e.id, e.error, brand: brand)]]
}
#if entries.len() > 0 {
  [#document("pdfs/book.pdf", title: [#brand.book-title])[#numbered-pages(book-page(entries, brand: brand))]]
}
