# Local Astro publication

GitHub Pages serves the `gh-pages` branch from `/`. Build locally with Bun,
Python, Git, and the local Pingstore; Typst and Demolab are not required.

```sh
uv run python tools/publishing/publish.py build
python -m http.server 3001 --directory .publication/site
uv run python tools/publishing/publish.py publish
```

The builder freezes latest presentation selections for reports using latest,
while frontmatter `runs` pins remain authoritative for pinned reports. Astro
reads the selected exports directly and calculates bound numerical prose in
TypeScript. The Pingstore lookup remains intentionally simple, without schema,
payload-checksum or ancestry validation. No scientific outputs are copied to a
persistent article directory.

Authored-source fingerprints before and after the build detect concurrent
edits. Static output and a receipt live in disposable `.publication/`; the
receipt records latest selections, source fingerprints/status and output
checksums. Failed builds preserve the previous publication. The finished site
contains neither the store nor private build receipts. Videos preload nothing.

Inspect the publication on port 3001 before publishing. `publish` verifies the
exact built files against the receipt and pushes them to `gh-pages` without
rebuilding or committing source changes. It retains remote `CNAME` and
`pr-preview/`, and uses a normal push to reject concurrent updates. The old
Demolab production/preview workflow templates remain disabled.
