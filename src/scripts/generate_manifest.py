"""Query .typ file metadata via typst and generate pdfs/manifest.json."""

import json
import subprocess
import sys
from pathlib import Path

TYPST_DIR = Path("src/typst")


def query_metadata(typ_file: Path) -> dict:
    name = typ_file.stem
    root = "src" if name == "report.1-ping-bptt" else "src/typst"
    result = subprocess.run(
        ["typst", "query", str(typ_file), "<meta>",
         "--root", root, "--one", "--field", "value"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: failed to query {name}: {result.stderr.strip()}", file=sys.stderr)
        return {"title": name, "date": "", "description": ""}
    return json.loads(result.stdout)


def main():
    entries = []
    for f in sorted(TYPST_DIR.glob("*.typ")):
        meta = query_metadata(f)
        entries.append({
            "file": f"{f.stem}.pdf",
            "title": meta.get("title", f.stem),
            "description": meta.get("description", ""),
            "date": meta.get("date", ""),
        })

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/vite/public/manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Generated manifest with {len(entries)} entries → {out}")


if __name__ == "__main__":
    main()
