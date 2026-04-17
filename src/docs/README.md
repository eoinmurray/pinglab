# docs

Astro site for pinglab. Built with bun, deployed via GitHub Pages.

```sh
bun install
bun run dev      # localhost:4321
bun run build    # static build to dist/
```

See `src/pages/llm-context.md` for the docs conventions (notebook structure, figure namespacing, date format, captions, etc.).

## Figure freeze flow

Figures produced by `src/pinglab/oscilloscope.py` or by a notebook repro script at `src/pinglab/notebook/<slug>.py` land under `src/artifacts/` (gitignored). To publish, freeze into the docs tree:

```sh
uv run python src/scripts/freeze-figure.py <source> src/docs/public/figures/notebook/<entry-slug>/<name>.png
```

The script copies the PNG and writes a sidecar JSON with the git SHA at freeze time and the run config. Every frozen figure belongs to exactly one notebook entry — reference from markdown via the web path `/figures/notebook/<entry-slug>/<name>.png`.
