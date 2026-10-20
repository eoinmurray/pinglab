# Local Astro publication

GitHub Pages serves the `gh-pages` branch from `/`. Build locally with Bun,
Python, Git, and the local Pingstore; Typst and Demolab are not required.

```sh
uv run python tools/publishing/publish.py build
(cd site && bun run astro preview --outDir ../.publication/site --port 3001)
uv run python tools/publishing/publish.py publish
```

The builder freezes latest presentation selections for reports using latest,
while frontmatter `runs` pins remain authoritative for pinned reports. Astro
reads the selected exports directly and calculates bound numerical prose in
TypeScript. Ordinary number and figure lookups remain simple, without full payload
validation. Parameter tables read original native snapshots through the pinned
presentation ancestry; the snapshot loader checks record identities and input
digest references but does not audit complete payloads. No scientific outputs are copied to a
persistent article directory.

Authored-source fingerprints cover `site/`, each experiment's `report/` and
`js/` directories, `md/notes/`, `md/slides/`, and `papers/`, including supporting assets.
Fingerprints before and after the build detect concurrent
edits. Static output and a receipt live in disposable `.publication/`; the
receipt records latest selections, source fingerprints/status and output
checksums. Failed builds preserve the previous publication. The finished site
contains neither the store nor private build receipts. Videos preload nothing.

Inspect the publication on port 3001 before publishing. `publish` verifies the
exact built files against the receipt and pushes them to `gh-pages` without
rebuilding or committing source changes. It retains remote `CNAME` and
`pr-preview/`, and uses a normal push to reject concurrent updates. The old
Demolab production/preview workflow templates remain disabled.

## Run assets

Development and static builds prefer `https://assets.eoinmurray.info` for each
selected presentation asset. Selection downloads the remote file with a ten-second
timeout and checks its SHA-256 against the local file, plus its content type.
Missing files, network failures, incompatible responses and checksum differences
fall back to the same local run and filename. JSON animation data also requires
CORS permission for the requesting origin; otherwise it stays local.

During development, `/run-images/` redirects verified assets to R2 and serves
fallbacks from local Pingstore. Choices are cached for thirty seconds and are
invalidated by local file size or modification-time changes. Redirects are not
browser-cached. Local development still needs the selected run's local exports.

At build time, choices are frozen. HTML uses absolute R2 URLs for verified
assets; only fallback assets are emitted under `/run-images/` for GitHub Pages.
The private build receipt records the source, URL, size and SHA-256 of every
considered asset, and the publisher checks bundled fallbacks against those
records. A later backup switches assets to R2 on the next build. A later R2
outage does not cause a published static page to switch to your local store.

Set `PINGLAB_ASSET_BASE_URL` to override the asset origin, or set it to an empty
string for an entirely local/offline build. Selection never uploads files or
changes bucket configuration. Test selection with `cd site && bun run test` and
the publisher with `.venv/bin/python -m pytest tools/publishing/tests -q`.

## Hostnames

GitHub Pages uses `pinglab.eoinmurray.info` as its custom domain, with a
DNS-only CNAME to `eoinmurray.github.io`. The publisher preserves the remote
`CNAME` file.

`pl.eoinmurray.info` is proxied through Cloudflare solely for a Single Redirect
rule: hostname equals `pl.eoinmurray.info`, permanent 301 destination
`concat("https://pinglab.eoinmurray.info", http.request.uri.path)`, with query
strings preserved. No Worker is involved.
