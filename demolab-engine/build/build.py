"""Discover writings/*.typ, write a JSON manifest, and compile all three targets.

build.py does only what Typst can't: it globs the filesystem (Typst has no directory
listing) and orchestrates the compiler. It writes the discovered id/asset lists to
temp/bundle/index.json; the committed, static demolab-engine/build/main.typ reads that manifest
and does the rest (imports, documents, assets) in plain Typst — there is no generated
source. Keeping the logic in main.typ means you can read it, and even run it by hand.

One `typst compile --format bundle --features bundle,html demolab-engine/build/main.typ` emits,
into artifacts/site/:
  index.html            — homepage index of experiments + articles
  <id>.html             — per-entry web page (figures inline, video plays)
  <id>.mp4              — video assets
  pdfs/<id>.pdf         — per-entry individual PDF
  pdfs/book.pdf         — every entry concatenated into one PDF (book mode)

The site (artifacts/site/) is a self-contained build output (gitignored, deployed to
Pages). The PDFs are also mirrored to the committed artifacts/pdfs/ as shareable
deliverables.

Each writings/<id>.typ exposes `#let meta = (...)` and `#let body = [...]`.
Entries not yet in that convention are skipped (incremental migration).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (this file is demolab-engine/build/)
WRITINGS = ROOT / "writings"
ENGINE = ROOT / "demolab-engine" / "build"  # the Typst engine (main.typ, lib.typ, style.css)
MAIN = ENGINE / "main.typ"                 # committed bundle root (reads the manifest)
BUILD = ROOT / "temp" / "bundle"          # scratch: the generated manifest + deck PDFs
MANIFEST = BUILD / "index.json"            # scratch: id/asset lists main.typ reads
DECKS = BUILD / "decks"                     # scratch: compiled deck PDFs, embedded as assets
SITE = ROOT / "artifacts" / "site"         # bundle output (HTML + mp4 + pdfs/), gitignored
PDFS = ROOT / "artifacts" / "pdfs"         # committed copy of the PDFs (shareable)
TYPST = "typst"  # system CLI — needs --features bundle,html (experimental)


def discover():
    """Entry ids (exp*/ar*) that follow the meta+body convention, sorted.

    Match real top-level definitions (`#let meta` / `#let body` at line start), not
    prose or comments that merely mention them. Slide decks (`*.slide.typ`) are a
    separate category — see discover_decks — so they're skipped here."""
    ids = []
    for p in sorted(WRITINGS.glob("*.typ")):
        if p.name.endswith(".slide.typ"):
            continue
        lines = p.read_text().splitlines()
        has_meta = any(ln.startswith("#let meta") for ln in lines)
        has_body = any(ln.startswith("#let body") for ln in lines)
        if has_meta and has_body:
            ids.append(p.stem)
    return ids


def discover_decks():
    """Deck ids from `writings/<id>.slide.typ` — standalone touying slide decks, sorted.

    Touying decks are paged-only (they don't survive HTML export, see the deck header
    comment), so they aren't bundle entries. Instead they're compiled to standalone PDFs
    and linked from the homepage. Each deck declares `#let meta` (title/date) but no
    `#let body`; the meta is imported to label the link."""
    return [p.name.removesuffix(".slide.typ") for p in sorted(WRITINGS.glob("*.slide.typ"))]


def write_manifest(ids: list[str], deck_ids: list[str]) -> None:
    """Write temp/bundle/index.json — the id/asset lists demolab-engine/build/main.typ reads.

    This is the only place per-entry knowledge is assembled, and it's pure data (no Typst
    source): the entry ids + kind, each entry's mp4 filenames (globbed here because Typst
    can't list a directory), and the deck ids."""
    entries = [
        {
            "id": i,
            "kind": "experiment" if i.startswith("exp") else "article",
            "videos": [v.name for v in sorted((ROOT / "artifacts" / "data" / i).glob("*.mp4"))],
        }
        for i in ids
    ]
    # Signal whether the optional root demolab.yaml exists — Typst can't stat files, so
    # main.typ only reads it (for branding) when this flag says it's there.
    manifest = {
        "entries": entries,
        "decks": [{"id": d} for d in deck_ids],
        "has_brand_config": (ROOT / "demolab.yaml").exists(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def compile_decks(deck_ids: list[str]) -> None:
    """Compile each standalone deck to a scratch PDF (temp/bundle/decks/<id>.pdf).

    main.typ embeds these as bundle assets at pdfs/<id>.pdf (so both `task build` and
    `typst watch` (dev) emit + serve them). Must run before the bundle compile so the
    asset `read(...)` finds the files. Decks don't live-reload in dev — re-run
    `task dev`/`task build` to pick up deck edits."""
    DECKS.mkdir(parents=True, exist_ok=True)
    for d in deck_ids:
        subprocess.run(
            [TYPST, "compile", "--root", str(ROOT),
             str(WRITINGS / f"{d}.slide.typ"), str(DECKS / f"{d}.pdf")],
            check=True,
        )


def main() -> None:
    # --generate-only writes the manifest + deck PDFs without compiling the bundle — used
    # by `task dev`, which then runs `typst watch` on the committed main.typ (its own HTTP
    # server + live reload; main.typ re-reads the manifest, so writing edits hot-reload).
    generate_only = "--generate-only" in sys.argv
    ids = discover()
    deck_ids = discover_decks()
    if not ids:
        print("no converted writings (need `#let meta` + `#let body`)", file=sys.stderr)
        sys.exit(1)
    BUILD.mkdir(parents=True, exist_ok=True)
    # Compile decks first so their PDFs exist for the asset embeds in main.typ.
    compile_decks(deck_ids)
    write_manifest(ids, deck_ids)
    SITE.mkdir(parents=True, exist_ok=True)
    if generate_only:
        print(f"wrote manifest for {len(ids)} entries: {', '.join(ids)}"
              + (f" + {len(deck_ids)} decks: {', '.join(deck_ids)}" if deck_ids else ""))
        return
    subprocess.run(
        [TYPST, "compile", "--format", "bundle", "--features", "bundle,html",
         "--root", str(ROOT), str(MAIN), str(SITE) + "/"],
        check=True,
    )
    # mirror the compiled PDFs (entries, book, and decks) to the committed artifacts/pdfs/
    PDFS.mkdir(parents=True, exist_ok=True)
    for pdf in sorted((SITE / "pdfs").glob("*.pdf")):
        shutil.copy(pdf, PDFS / pdf.name)
    built = f"built {len(ids)} entries" + (f" + {len(deck_ids)} decks" if deck_ids else "")
    print(f"{built} -> {SITE}/ (pdfs mirrored -> {PDFS}/)  entries: {', '.join(ids)}"
          + (f"  decks: {', '.join(deck_ids)}" if deck_ids else ""))


if __name__ == "__main__":
    main()
