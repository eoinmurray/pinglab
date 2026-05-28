"""033 — Wilson-Cowan mean-field bifurcation analysis of PING.

Theory page only at present. Numerics deferred — see the doc's
"Next steps" section for the planned analysis pipeline.

This runner produces no figures yet. It writes a minimal numbers.json
so the notebook index can find the entry.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SLUG = "nb033"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = {
        "slug": SLUG,
        "status": "theory-only",
        "config": {"tier": "n/a"},
        "results": {},
        "success_criteria": [
            {
                "label": "theory page exists",
                "passed": True,
                "detail": "src/docs/src/pages/notebooks/nb033.mdx",
            }
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print("nb033 is theory-only at present — see the doc for the planned numerics.")


if __name__ == "__main__":
    main()
