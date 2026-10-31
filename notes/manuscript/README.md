# Manuscript

Article aggregating existing PING experiments. `index.mdx` contains the main
text; `appendix.mdx` and `references.mdx` are rendered on the same page.
`figures/` holds presentation code, illustrations and composition tests.

The public URL is `/notes/manuscript`; `/exp110` redirects here. The
`presentation: exp110` declaration preserves the historical Pingstore namespace
for compound figures. Existing run identities, payloads and source pins remain
unchanged. `inputs` declares evidence dependencies for selection and retention.

Run from the repository root:

1. Compose figures: `uv run python notes/manuscript/figures/present.py --source EXP054_ANALYSIS --theory-source EXP117_ANALYSIS` (optional `--run-id`).
2. Check presentation contracts: `uv run pytest notes/manuscript/figures/test.py`.
3. Regenerate illustrations: `uv run python notes/manuscript/figures/present.py --draw-schematics`.

Composition consumes explicitly validated source runs; it launches no simulations.
Illustrations do not reconstruct missing historical bundles.
