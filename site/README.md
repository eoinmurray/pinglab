# Pinglab Astro publishing

Run the site with Bun from `site/`:

```sh
bun install
bun run dev --background     # localhost:3000, toolbar disabled
bun run astro dev stop
bun run build                # disposable dist/
```

Articles live beside experiment code in `experiments/expXXX/report.mdx`.
The homepage is text-only and ordered by ID, highest first. Distill's official
v2 runtime and styles are self-hosted; the article shell supplies the byline,
sticky contents, and appendix. Optional `parameters.mdx`, `appendix.mdx`, and
`references.mdx` files supply footer content. Original Typst files are unchanged.
Native MathML is retained in `math.json`; complex tables live in `tables.json`.
The Astro build has no Typst or Demolab dependency.

Collections use the configured display labels and order in
`src/lib/collections.ts`. Source tags remain internal metadata for data availability
and author-assigned review status. Citation previews appear on hover or keyboard focus; clicks still
navigate to references. Report pages embed Hypothes.is with highlights visible
and the sidebar closed by default. Readers can open it from the Hypothesis
toolbar or by clicking a highlight. The homepage and collection pages do not
load the annotation client. Interactive run selectors are absent.

## Data and values

The simple TypeScript Pingstore reader resolves a latest presentation once per
build/dev session, or uses an article's explicit frontmatter `runs` selections.
It does not perform schema/checksum/ancestry validation. Restart the dev server
to pick up newly completed runs. Figures/media come directly from selected
exports; only disposable builds copy them.

`LiveValue` evaluates the migrated authored calculations in
`src/lib/report-values.ts`, using the same selected JSON as the figures. Bound
numbers and complex-table values update during rendering when selections change.
No arbitrary JavaScript is evaluated. Missing values or nonfinite results fail
rather than becoming fabricated numbers. Existing direct MDX calculations also
remain active. Authored literal numbers and interpretations remain authored;
changing the dataset does not automatically rewrite scientific conclusions.

Dataset footers show completed run history, producing execution timing for
imported runs, and upstream/downstream article dependencies. Bold run IDs mark
presentations used by the article. Text-only dependent articles retain dependency
links without inventing unavailable data.

Seven articles have no local presentation: exp022, exp024, exp047, exp054,
exp080, exp081, and exp117. Their existing unavailable-data notices remain.

## Publication

See `../tools/publishing/README.md`. Build a frozen local Astro publication with:

```sh
uv run python tools/publishing/publish.py build
python -m http.server 3001 --directory .publication/site
uv run python tools/publishing/publish.py publish
```

Run those commands from the repository root. The publisher retains a build
receipt, checks the exact static output before pushing it to `gh-pages`, and
preserves the remote CNAME and existing PR previews.
