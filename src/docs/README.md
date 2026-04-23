# docs

Astro site for pinglab. Built with bun, deployed via GitHub Pages to [pl.eoinmurray.info](https://pl.eoinmurray.info).

```sh
bun install
bun run dev      # localhost:3000
bun run build    # static build to dist/
```

See `src/pages/styleguide.md` for the docs conventions (notebook structure, figure namespacing, date format, captions, etc.).

## Figures

Raw outputs from `src/pinglab/oscilloscope.py` land under `src/artifacts/` (gitignored). Figures shown in a notebook entry are written by that entry's repro script at `src/pinglab/notebooks/<slug>.py` directly into `src/docs/public/figures/notebooks/<slug>/`. The repro script is the promotion gate — running it regenerates every figure and `numbers.json` the entry cites. Every published figure belongs to exactly one entry; reference from markdown via the web path `/figures/notebooks/<slug>/<name>.png`.
