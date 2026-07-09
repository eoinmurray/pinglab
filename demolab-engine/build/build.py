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
from __future__ import annotations  # keep type hints lazy so `X | None` works on Python 3.9 too

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Content root. Normally the repo root (this file is demolab-engine/build/); override with
# DEMOLAB_ROOT (e.g. demolab-engine/scaffold/demo for `task dev:demo-site`). When the root is the
# shipped demo, writings/data/config are read from there directly; Typst --root stays at the real
# repo checkout so engine paths resolve, with content-prefix/build-root passed as --input.
REPO = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO / "demolab-engine" / "scaffold" / "demo"
ROOT = Path(os.environ.get("DEMOLAB_ROOT") or REPO)
DEMO = os.environ.get("DEMOLAB_DEMO") == "1" or ROOT.resolve() == DEMO_ROOT.resolve()
CONTENT = DEMO_ROOT if DEMO else ROOT
WRITINGS = CONTENT / "writings"
_local_engine = ROOT / "demolab-engine" / "build"
ENGINE = _local_engine if (_local_engine / "main.typ").exists() else REPO / "demolab-engine" / "build"
MAIN = ENGINE / "main.typ"                 # committed bundle root (reads the manifest)
BUILD = ROOT / "temp" / "bundle"          # scratch: the generated manifest + deck PDFs
MANIFEST = BUILD / "index.json"            # scratch: id/asset lists main.typ reads
DECKS = BUILD / "decks"                     # scratch: compiled deck PDFs, embedded as assets
SITE = ROOT / "artifacts" / "site"         # bundle output (HTML + mp4 + pdfs/), gitignored
PDFS = ROOT / "artifacts" / "pdfs"         # committed copy of the PDFs (shareable)
TYPST = "typst"  # system CLI — needs --features bundle,html (experimental)


def typst_root_and_inputs() -> tuple[Path, list[str]]:
    """Typst --root and --input flags. Demo mode keeps --root at the real repo (engine + prefixed
    content paths) while scratch/output live under DEMOLAB_ROOT (usually scaffold/demo/)."""
    prefix = content_prefix()
    if prefix:
        inputs = ["--input", f"content-prefix={prefix}"]
        if ROOT != REPO:
            inputs.extend(["--input", f"build-root=/{ROOT.relative_to(REPO).as_posix()}"])
        return REPO, inputs
    return ROOT, []


def content_prefix() -> str:
    """Root-relative prefix for content paths in main.typ ('' at repo root, else '/demolab-engine/scaffold/demo')."""
    if not DEMO:
        return ""
    return "/" + CONTENT.relative_to(REPO).as_posix()


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


def write_manifest(ids: list[str], deck_ids: list[str], broken: dict | None = None) -> None:
    """Write temp/bundle/index.json — the id/asset lists demolab-engine/build/main.typ reads.

    This is the only place per-entry knowledge is assembled, and it's pure data (no Typst
    source): the entry ids + kind, each entry's mp4 filenames (globbed here because Typst
    can't list a directory), and the deck ids. An entry in `broken` (id -> error text) carries an
    `error` field and loads no assets — main.typ renders it as a stub instead of importing it."""
    broken = broken or {}
    entries = []
    for i in ids:
        entry = {
            "id": i,
            "kind": "experiment" if i.startswith("exp") else "article",
            "videos": [] if i in broken else [v.name for v in sorted((CONTENT / "artifacts" / "data" / i).glob("*.mp4"))],
        }
        if i in broken:
            entry["error"] = broken[i]
        entries.append(entry)
    # Signal whether the optional root demolab.yaml exists — Typst can't stat files, so
    # main.typ only reads it (for branding) when this flag says it's there.
    manifest = {
        "entries": entries,
        "decks": [{"id": d} for d in deck_ids],
        "has_brand_config": (CONTENT / "demolab.yaml").exists(),
        "content_prefix": content_prefix(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def compile_decks(deck_ids: list[str]) -> list[str]:
    """Compile each standalone deck to a scratch PDF (temp/bundle/decks/<id>.pdf); return the ones
    that built. A deck that fails to compile is skipped with a warning rather than failing the whole
    build (main.typ only embeds decks that produced a PDF).

    main.typ embeds these as bundle assets at pdfs/<id>.pdf. Must run before the bundle compile so
    the asset `read(...)` finds the files. The dev server (devserver.py) reruns this on every
    change, so deck edits and new decks live-reload like any entry."""
    DECKS.mkdir(parents=True, exist_ok=True)
    good = []
    typst_root, inputs = typst_root_and_inputs()
    for d in deck_ids:
        proc = subprocess.run(
            [TYPST, "compile", "--root", str(typst_root), *inputs,
             str(WRITINGS / f"{d}.slide.typ"), str(DECKS / f"{d}.pdf")],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            good.append(d)
        else:
            print(f"  ⚠ deck {d} failed to build — skipping it: "
                  + _error_excerpt(proc.stdout + proc.stderr).splitlines()[0], flush=True)
    return good


def _entry_from_error(err: str, candidates: set) -> str | None:
    """Which entry did a bundle-compile error come from? Parsed from a `/writings/<id>.typ` mention
    in the message (only ids we can still drop)."""
    for m in re.finditer(r"/writings/([A-Za-z0-9_-]+)\.typ", err):
        if m.group(1) in candidates:
            return m.group(1)
    return None


def _error_excerpt(err: str, lines: int = 8) -> str:
    """The first `error:` block from Typst's output, for the stub page and the warning."""
    rows = err.splitlines()
    for i, row in enumerate(rows):
        if row.lstrip().startswith("error:"):
            return "\n".join(rows[i:i + lines]).strip()
    return err.strip() or "build failed"


def compile_bundle(ids: list[str], deck_ids: list[str]) -> dict:
    """Compile the whole bundle. If an entry fails (a missing figure, a Typst error), flag it and
    retry, so it renders as a stub page instead of taking the rest of the site down with it. Returns
    the {id: error} map of entries that were stubbed."""
    broken: dict = {}
    while True:
        write_manifest(ids, deck_ids, broken=broken)
        typst_root, inputs = typst_root_and_inputs()
        proc = subprocess.run(
            [TYPST, "compile", "--format", "bundle", "--features", "bundle,html",
             "--root", str(typst_root), *inputs, str(MAIN), str(SITE) + "/"],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return broken
        err = proc.stdout + proc.stderr
        bad = _entry_from_error(err, set(ids) - broken.keys())
        if bad is None:
            # Not attributable to one entry (an engine, asset, or deck error): surface the real
            # failure rather than looping.
            sys.stderr.write(err)
            raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)
        broken[bad] = _error_excerpt(err)
        print(f"  ⚠ {bad} failed to build — stubbing it, keeping the rest: "
              + broken[bad].splitlines()[0], flush=True)


def main() -> None:
    # --generate-only writes the manifest + deck PDFs without compiling the bundle: a hand
    # tool for inspecting what the compiler will see. (Dev serving is devserver.py, which runs
    # a full build on each change; it doesn't use this flag.)
    generate_only = "--generate-only" in sys.argv
    # --skip-decks reuses the deck PDFs already in temp/bundle/decks/ instead of recompiling
    # them. The dev server passes it when a change touched no deck source or data asset, so a
    # prose/CSS/lib edit doesn't pay for deck compilation it can't have affected. Safe only when
    # those PDFs exist (a full build ran first); a bare `task build` never skips.
    skip_decks = "--skip-decks" in sys.argv
    ids = discover()
    deck_ids = discover_decks()
    # Zero writings is a valid state (a freshly `task scaffold`-ed repo): main.typ renders
    # a friendly empty-state homepage, so we build rather than error.
    BUILD.mkdir(parents=True, exist_ok=True)
    # Compile decks first so their PDFs exist for the asset embeds in main.typ (skip reuses the
    # PDFs already on disk). Either way, only decks that actually have a PDF are referenced.
    if skip_decks:
        good_decks = [d for d in deck_ids if (DECKS / f"{d}.pdf").exists()]
    else:
        good_decks = compile_decks(deck_ids)
    write_manifest(ids, good_decks)
    SITE.mkdir(parents=True, exist_ok=True)
    if generate_only:
        print(f"wrote manifest for {len(ids)} entries: {', '.join(ids)}"
              + (f" + {len(good_decks)} decks: {', '.join(good_decks)}" if good_decks else ""))
        return
    # One bad entry (a missing figure, a Typst error) becomes a stub page instead of failing the
    # whole site — compile_bundle flags it and retries.
    broken = compile_bundle(ids, good_decks)
    good = [i for i in ids if i not in broken]
    # mirror the compiled PDFs (entries, book, and decks) to the committed artifacts/pdfs/
    PDFS.mkdir(parents=True, exist_ok=True)
    for pdf in sorted((SITE / "pdfs").glob("*.pdf")):
        shutil.copy(pdf, PDFS / pdf.name)
    # The verbose detail (which ids built / stubbed, where the PDFs mirror) goes on its own line;
    # the CONCISE summary is printed LAST, because the dev-server watch loop echoes only build.py's
    # final stdout line on each rebuild. So a `task dev` session shows a terse one-liner, while a
    # one-shot `task build` still prints the full id list above it.
    print(f"  entries: {', '.join(good)}"
          + (f"  ·  decks: {', '.join(good_decks)}" if good_decks else "")
          + (f"  ·  ⚠ stubbed: {', '.join(sorted(broken))}" if broken else "")
          + f"  ·  pdfs -> {PDFS.relative_to(ROOT)}/")
    summary = f"built {len(good)} entries" + (f" + {len(good_decks)} decks" if good_decks else "")
    if broken:
        summary += f", {len(broken)} stubbed"
    print(f"{summary} -> {SITE.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    main()
