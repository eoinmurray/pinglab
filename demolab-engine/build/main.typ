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

// Import each writing dynamically (import paths may be computed — no literal codegen).
// An entry contributes meta + body; a deck contributes only meta (touying is paged-only,
// so decks are compiled to standalone PDFs by build.py and embedded as assets below).
#let entries = manifest.entries.map(e => {
  import "/writings/" + e.id + ".typ": meta, body
  (id: e.id, kind: e.kind, meta: meta, body: body)
})
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

// --- documents (one compile emits them all into artifacts/site/) ---
#document("index.html", title: [#brand.name])[#index-page(entries, decks: decks, brand: brand, collection-order: collection-order, collection-meta: collection-meta)]
#document("all.html", title: [#brand.name — all entries])[#all-page(entries, decks: decks, brand: brand, collection-meta: collection-meta)]
// One page per collection (web only — the book/PDFs don't have collection pages).
#let all-items = collect-items(entries, decks)
#for c in all-items.map(it => it.coll).dedup() {
  [#document(c + ".html", title: [#brand.name — #collection-label(c, collection-meta)])[#collection-page(c, all-items.filter(x => x.coll == c), brand: brand, collection-meta: collection-meta)]]
}
#for e in entries {
  [#document(e.id + ".html", title: [#e.meta.title])[#entry-page(e.meta, e.body, id: e.id, brand: brand)]]
  [#document("pdfs/" + e.id + ".pdf", title: [#e.meta.title])[#entry-page(e.meta, e.body, id: e.id, brand: brand)]]
}
#document("pdfs/book.pdf", title: [#brand.book-title])[#book-page(entries, brand: brand)]
