# Runbook: Embedding as a docs subfolder

Use demolab as a docs subfolder inside another project. The tree is path-portable — every tool/writing resolves paths relative to the repo root (`typst --root`), and the built site uses **relative links**, so it works under any URL path with no base config.

**Triggers** — say any of these, or just `EMBED-DOCS`: **"embed demolab as a docs site"**, "use this as a wiki subfolder", "drop this into my project as docs".

1. **Place it.** Copy demolab degitted (no `.git`) into the host project, e.g. `docs/`. Delete the copy's bundled `.github/workflows/deploy.yml` (it assumes demolab is the repo root).
2. **Run from inside the subfolder** so `uv`/`typst`/`task` resolve demolab's manifests: `cd docs && task install && task build` (or `task dev`). Output lands in `docs/artifacts/site/`.
3. **Deploy from the HOST repo.** Add a Pages workflow to the *host* repo that installs `typst`, runs `python3 docs/demolab-engine/build/build.py`, and uploads `docs/artifacts/site/` — model it on demolab's own `.github/workflows/deploy.yml`. Enable Pages on the host (*Settings → Pages → Source: GitHub Actions*).

Notes: one Pages site per repo — if the host already uses Pages, deploy the docs from a dedicated repo.
