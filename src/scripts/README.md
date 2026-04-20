# scripts

One-off utilities that support the repo but aren't part of the core Python package. Not imported by `src/pinglab/`; invoke directly.

## validate-notebook.py

Checks every notebook entry against the conventions in `src/docs/src/pages/styleguide.md`.

```sh
uv run python src/scripts/validate-notebook.py
```

Four checks:

1. **Triple-existence** — each slug has an entry (`src/docs/src/pages/notebook/<slug>.{md,mdx}`), a repro script (`src/pinglab/notebook/<slug>.py`), and a figure directory (`src/docs/public/figures/notebook/<slug>/`).
2. **Figure references resolve** — every `/figures/notebook/<slug>/<name>` URL in an entry exists on disk, no entry references another entry's figures, and every file in a figure dir is referenced by its entry (orphan warning).
3. **H2 skeleton** (convention 1) — first five H2s are *Introduction / Method / Findings / Implications / Next steps*.
4. **Caption after image** (convention 6) — every markdown image and `<video>` is followed by an italic caption line.

Frontmatter `structure: paper` exempts an entry from checks 3 and 4 — used for paper-style drafts with custom sectioning.

Exits 0 if no errors (warnings allowed), 1 otherwise. Run before committing entry changes.
