# docs

Astro site for pinglab. Built with bun, deployed via GitHub Pages.

```sh
bun install
bun run dev      # localhost:4321
bun run build    # static build to dist/
```

See `src/pages/llm-context.md` for the docs conventions (journal structure, figure namespacing, date format, captions, etc.).

## Figure freeze flow

Figures produced by `src/pinglab/oscilloscope.py` or by a journal repro script at `src/pinglab/journal/<slug>.py` land under `src/artifacts/` (gitignored). To publish, freeze into the docs tree:

```sh
uv run python src/scripts/freeze-figure.py <source> src/docs/public/figures/journal/<entry-slug>/<name>.png
```

The script copies the PNG and writes a sidecar JSON with the git SHA at freeze time and the run config. Every frozen figure belongs to exactly one journal entry — reference from markdown via the web path `/figures/journal/<entry-slug>/<name>.png`.
