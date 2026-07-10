"""Compile standalone Typst decks to artifacts/pdfs/ — the portable engine behind
`task slides`.

A "standalone" writing is any writings/*.typ that does NOT declare the bundle contract
(`#let meta` + `#let body`); bundle entries are built by build.py instead. This logic
lived as a shell loop in the Taskfile, but it leaned on grep and basename, which Windows
doesn't have — so it's stdlib Python now, sharing build.py's typst resolution.
"""
import re
import subprocess
import sys

import build

_META = re.compile(r"^#let meta", re.M)
_BODY = re.compile(r"^#let body", re.M)


def is_bundle_entry(source: str) -> bool:
    return bool(_META.search(source) and _BODY.search(source))


def main() -> None:
    build.PDFS.mkdir(parents=True, exist_ok=True)
    found = failed = False
    for f in sorted(build.WRITINGS.glob("*.typ")):
        if is_bundle_entry(f.read_text(encoding="utf-8")):
            continue
        found = True
        out = build.PDFS / (f.name.removesuffix(".typ") + ".pdf")
        proc = subprocess.run([build.TYPST, "compile", "--root", str(build.ROOT), str(f), str(out)])
        if proc.returncode == 0:
            print(f"wrote {out.relative_to(build.ROOT)}", flush=True)
        else:
            failed = True  # typst already printed the error; keep compiling the rest
    if not found:
        print("no standalone decks in writings/", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
