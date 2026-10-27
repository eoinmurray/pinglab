# Pinglab Astro publishing

The reporting resolver also requires Python 3.10+ and PyYAML. A normal
`uv sync --dev` supplies them. The site uses `.venv/bin/python` when available,
otherwise `python3`; `PINGLAB_PYTHON` can select an explicit interpreter.
Pages installs the resolver/S3 dependencies and rclone without installing scientific packages.

Run the site with Bun from `site/`:

```sh
bun install
bun run dev --background     # localhost:3000, toolbar disabled
bun run astro dev stop
bun run build                # disposable dist/
```

Pixi is prebundled when development starts, including animation scripts outside
`site/`. Development and builds use separate Vite caches under
`node_modules/.vite/dev` and `node_modules/.vite/build`, so production SSR optimization also stays out of the development cache and cannot replace
the dependency URLs used by a running preview.

Distill's runtime is already self-hosted at `/vendor/distill/template.v2.js`.
Only report and note pages load it, with `defer` so it does not block HTML
parsing. Listing pages use the same prerendered styles without the runtime.
Search renders its button on the server and downloads React, cmdk and Radix only
on the first click or Cmd/Ctrl+K activation; that first activation opens the dialog.

Page rendering errors are isolated by `src/middleware.ts`. The affected URL
shows “An error occurred.” and the error message with a link back to the articles;
other pages continue rendering. HTML is buffered so component errors cannot
leave a partially rendered page. Development returns HTTP 500; static builds
fail on rendering errors so incomplete evidence cannot be deployed as a success
page. Compiler, import-resolution,
and content-schema errors can still prevent a build before rendering begins.
Development report routes read current collection metadata on each request so
edited evidence metadata reaches the layout and Dataset footer even when Astro retains
older static-path props.

Exp124 uses the latest completed presentation, with one frozen selection per
production build or development page request for figures, numbers and native parameter snapshots.
Its report-local snapshot loader follows that presentation's recorded ancestry;
missing native snapshots remain explicitly unavailable and invalid evidence fails.
The earlier published HTML is separately labelled at
`/recovered/exp124/published.html`, with assets under
`public/recovered/exp124/`. It is an archive, not a recovered Pingstore run;
its historical Dataset inventory is not a current local inventory.

Development run assets use the exact local presentation when available, otherwise the verified R2 catalog;
they do not scan every article's runs. Missing assets return HTTP 404. During a
static build, unavailable run exports are logged and skipped in asset enumeration,
while data or figure errors fail the build. Shared article
appendices load only for the current report. Listings also exclude cached reports
whose source file has been deleted.

The header theme button sits to the right of GitHub and cycles through system,
light, and dark modes, with monitor, sun, and moon icons. It follows the system
by default, including live changes; explicit choices persist in local storage.
Cycling back to system clears the override. The theme applies before first paint.
Printed articles always use light colours, and scientific images keep their
original palettes.

The header's Search button opens a site-wide command palette, also available
with `Cmd+K` on macOS or `Ctrl+K` elsewhere. Type to filter, use Up/Down to
select, Enter to navigate, and Escape to close. Home is the first item, so
`Cmd+K` then Enter opens the homepage. It includes every article,
note, slide deck, and collection that has a page, using the same content data
as the article lists. Titles, descriptions, article IDs, and collection names
are searchable. Experiment numbers also work without the `exp` prefix, with or
without leading zeros (for example, `082` or `82`). New content appears
automatically on the next content sync/build.
The unlisted `/pingstore` page is available as Pingstore in the command palette,
but has no homepage or sidebar entry. It shows the shared Dataset table for each
experiment in the normal report appendix layout, newest experiment first,
preserving core’s evidence classifications and retention reasons. The report body contains only a title
and a short introduction, deduplicated Current/Required/Unresolved counts, and the
all-run eviction dry-run/confirmation commands. Unresolved holdings are never
classified as unused.
Experiments without a selected presentation show any completed available history.
The page requests search-engine exclusion with `noindex`. It is publicly
accessible by URL; it is not access-controlled. Missing selected evidence is
shown as unavailable for that experiment without blocking the other tables.

Individual papers appear in a Papers group with titles and authors, linking
to publisher/repository sources in production and local PDFs in development. Paper titles, authors, permanent IDs, filenames, years,
venues, DOIs, and collection names are searchable; unindexed PDFs use filenames.

`SiteSearch.astro` supplies the navigation data to a React island. The local
`ui/command.tsx` adapts shadcn/ui's Command composition with `cmdk` and Radix
Dialog; `styles/command.css` supplies Pinglab styling without Tailwind.

Articles live in `experiments/expXXX/report/report.mdx`, with supporting MDX,
CSS, and article components in the same `report/` directory. Experiment-specific
browser JavaScript lives in `experiments/expXXX/report/js/`; article components import
it from `./js/`. Shared site components and scripts remain under `site/src/`.
For experiment reports and notes, set `created_at` once at creation and refresh
`updated_at` when prose or figures change. Use quoted UTC datetimes with seconds
(`YYYY-MM-DDTHH:mm:ssZ`); see Writing Rules' “Editing and dates”.
Existing date-only `created_at` values remain supported.
Notes live in `md/notes/` as MDX and belong to the Notes collection
at `/collections/notes/`. Each note requires `title`, `author`, `description`,
`created_at`, `updated_at`, and `collection` frontmatter including `notes`.
The filename determines the URL, for example `/notes/compute-options/`.
Notes and experiments both use `Report.astro`, including its byline, contents
sidebar, article spacing, and annotations. Optional `abstract` frontmatter is
supported for both; experiment IDs and Dataset footers require experiment metadata.
The shared page layout declares each page's canonical URL on
`https://pinglab.eoinmurray.info`, without a trailing slash (except the root),
`.html` extension, query, or fragment. Astro uses `trailingSlash: 'never'` and
`build.format: 'file'`, producing `exp130.html` for the public URL `/exp130`.
GitHub Pages resolves these extensionless URLs. Use Astro preview for local
static previews; Python's basic HTTP server does not resolve `/exp130` to
`exp130.html`. Existing bookmarks ending in `/` need host redirects to remain
compatible with the new file layout.
Hypothes.is uses this identity to share article annotations between production
and localhost. Local annotations therefore belong to the production article too.
The homepage lists reports, notes, and slides with the same article previews.
Report and note bylines include a final Report column, stacking word, figure,
and (for experiments) run counts. Listings show the same counts beside collection
badges; slide previews retain their presentation metadata. Counts are computed
from rendered main content, including the abstract and resolved data bindings.
Words exclude headings, captions, tables, code, mathematics, editorial Meta,
references, and appendices. Figures count main-content figure elements once,
including multipanel and unavailable figures, but exclude tables. Run counts
include completed v4 compute/analyse/present records owned by the experiment in
the local/R2 union in development and the R2 catalog in production, including historical runs, without upstream or hidden runs.
These are available-run inventory counts, not a claim that every run informed the report.
Static builds freeze counts; development caches rendered word/figure counts until
an experiment, note, slide or shared site source changes, while refreshing run counts
on every request. Presentation selections refresh with each HTML request. A report that
cannot render shows unavailable word/figure counts in listings rather than taking
the listing offline. No authored count fields or article date changes are needed.
All article lists, including every collection, show experiments first, sorted by
descending numeric experiment ID, then other articles by decreasing authored
`updated_at`, with article ID and URL ascending to break ties. This order applies
in development and static builds.
The homepage has a compact Sort by control above the articles. Default order
uses that ordering; Recently updated sorts all articles by descending authored
`updated_at`. The selection persists across refreshes in browser local storage.
Press `s` to switch orders; the `[s]` hint beside Sort by can be clicked to
disable or re-enable this shortcut, and that preference also persists. The
shortcut ignores typing in editable fields, native selects, open dialogs,
modified keys, composition and held-key repeats.
If storage is unavailable, sorting still works for the current page. Without
JavaScript, the default order remains available and the control is hidden.
The homepage, `/notes/`, and `/collections/<collection>/` share the same
collection sidebar and article layout, including aligned navigation. Collection
pages show only that collection's articles and highlight its sidebar link.
`/collections/` lists
clickable collection names, including Notes, without article previews. Distill's official
v2 runtime and styles are self-hosted; the article shell supplies the byline,
sticky contents, and appendix. Optional `appendix.mdx` and `references.mdx` files
supply footer content. Put network diagrams first in `appendix.mdx`, followed by
the main parameter table and supporting detail. The appendix's own section headings
appear to the left of their content on desktop, aligned with the top of each
section, and above it on smaller screens. They are included in the table of
contents. Existing `parameters.mdx` files
remain supported by the shared layout under Parameters immediately above Dataset,
with a Parameters entry in the table of contents. The canonical scientific-writing
rules are preserved in [Writing Rules](../md/notes/writing-rules.md).
Write mathematics directly in MDX as `$...$` for inline expressions and `$$`
blocks for displayed equations. The existing remark-math and rehype-katex
plugins render these with KaTeX. Preserve displayed equation numbers with
`\tag{1}` and cross-reference anchors with `<span id="equation-id" />`.
Data-bound numbers use native MDX exports and expressions:

```mdx
import { reportData, display } from '../../../site/src/lib/report-data';

export const run = await reportData('exp044');
export const epochs = display(run.config.training_contract.common.epochs);

We trained for {epochs} epochs.
```

`reportData` uses the current request or build’s resolved presentation.
Report frontmatter must not contain `runs` or `thumbnailRun` pins. Pass `{ experiment: 'exp082' }`
for a declared upstream input, or `{ allowUnavailable: true }` for an article
whose data may be absent. The latter returns `undefined` only when there is no
presentation; missing or invalid selected JSON still fails. Guard calculations
with `run ? ... : 'unavailable'` in those articles. `display`, `round`, `mean`,
`percent`, `nullableRound`, and `joinLast` provide small shared helpers;
calculations and named bindings remain ordinary JavaScript in the article.

All article tables use `Table` with explicit `headers` and `rows` arrays:

```mdx
<Table headers={["Parameter", "Value"]} rows={[["Epochs", epochs]]} />
```

For grouped tables, pass `groups` instead of `rows`. Each group has a `label`
and nonempty `rows`; `headers` describes the data columns. Each group label
appears in a full-width heading row above its data, without adding a column:

```mdx
<Table headers={["Parameter", "Value"]} groups={[
  { label: "Network", rows: [["E / I neurons", "1,024 / 256"]] },
  { label: "Input and simulation", rows: [
    ["Input encoding", "Poisson pixel encoding"],
    ["Input encoding rate", "Maximum 25 Hz per input channel"],
    ["Timestep / presentation", "0.1 / 200 ms"],
  ] },
  { label: "Analysis", rows: [["Analysis window", "50–200 ms"]] },
]} />
```

Use custom groups and rows as needed. Additional data columns (for example,
`["Parameter", "COBA", "PING"]`) work in either format. Each group renders as a separate table body with an
accessible row-group heading; group labels support the same rich cells as data.

Cells accept strings, finite numbers, `{ math: "..." }` for KaTeX, or
`{ markdown: "..." }` for authored rich text and mixed text/mathematics.
Optional `caption`, `id`, and `label` props supply a plain caption, anchor,
and accessible region name. For Markdown captions, put the caption in the default
slot instead of the `caption` prop. Captioned tables use the same `figcaption`
styling as figures, outside the horizontal scroll region; an optional `number`
prop adds the bold `Table N.` prefix (including appendix labels such as `A1`).
Pass existing table captions through this API rather than separate paragraphs
or article-owned `figure`/`figcaption` wrappers. Tables without captions can
remain self-closing. Header and row widths must match. Plain cells are
escaped; Markdown cells are trusted authored content, not arbitrary run HTML.
There is no string-expression parser or per-value rendering component.
Do not create `math.json` sidecars or embed exported MathML in article sources.
The Astro build has no Typst or Demolab dependency.

Collections use the configured display labels and order in
`src/lib/collections.ts`. Reports, notes, and slides accept a single collection
slug or a nonempty list of unique slugs in frontmatter:

```yaml
collection: [neuromodulation, literature]
```

An article appears in every listed collection, and previews and article bylines
link to each membership in the authored order. Existing scalar values remain
valid. Notes must include `notes`; slides default to `slides` when the field is
omitted. For experiment runs, the first collection supplies Pingstore's single
provenance value; completed runs remain unchanged.

The collections index shows the description
configured in `src/lib/collections.ts`, a count of distinct article URLs, and
the latest member article's `updated_at` date. Articles in multiple collections
count once in each; latest activity describes article edits, not experiment execution.

Source tags remain internal metadata for data availability
and author-assigned review status. Citation previews appear on hover or keyboard focus; clicks still
navigate to references. Report and note pages embed Hypothes.is with highlights visible
and the sidebar closed by default. Readers can open it from the Hypothesis
toolbar or by clicking a highlight. The homepage and collection pages do not
load the annotation client. Interactive run selectors are absent.

## Papers

`/papers` lists the citation index in production, linking to each entry's DOI or
optional HTTPS `url` (publisher or authorised repository). Entries without a
public source remain visible without a download link. Production does not read
the R2 paper catalog, download PDFs, or emit `/papers/*.pdf` compatibility copies.

Development continues to list and serve local PDFs from `papers/`, including
subfolders, with unindexed PDFs shown by filename. Hidden files and symlinks are
excluded. Listings sort by newest publication year, then title. The left column
shows publication year and permanent paper ID. Development reads the directory
on refresh; no public PDF archive is needed.

`papers/index.yaml` holds citation metadata keyed by relative PDF filename.
Each entry requires a unique `id` such as `P01`, `title`, a nonempty `authors`
list, and integer `year`; `venue`, `doi`, and an HTTPS source `url` are optional.
The page displays this metadata with the PDF link, falling back to the filename
for unindexed PDFs. Invalid metadata fails rendering/building; development entries without
matching PDFs produce warnings. Production needs only the index. An absent index is allowed. The index is tracked
in Git; PDFs remain local ignored files. No extraction or external lookups run
during page rendering. Preprint metadata identifies the saved version.

Paper IDs are permanent, authored identifiers, never generated from list position.
The initial 16 papers were assigned `P01`–`P16` by increasing publication year,
then title alphabetically within each year. Give the next addition `P17`, then
increment on each addition regardless of publication year. Never renumber or
reuse an assigned ID, including after removing a paper. IDs use at least two
digits (`P01` through `P99`, then `P100` onwards). The page still sorts newest
publication years first.

Paper collections use the same sidebar layout and collection badges as article
lists, with separate pages at `/papers/collections/<slug>`. Set `collection` on
each index entry to a lowercase hyphenated slug or a nonempty list of unique
slugs, for example `collection: [probabilistic-coding, gamma-oscillations]`.
Labels are derived from the slugs. New collections appear automatically on
refresh in development or after rebuilding the static site. Missing membership
defaults to `uncategorized`, including PDFs without metadata. Current topic
collections include `probabilistic-coding`, `ping` (displayed as PING), and
`spiking`; papers without a matching topic retain `[uncategorized]`.

## Slides

Each deck has a folder under `md/slides/`, containing `slides.mdx` and an
`images/` folder for its images. For example,
`md/slides/hello-slides/slides.mdx` opens at `/slides/hello-slides`.
Flat `.md` and `.mdx` decks also remain supported, using the file stem as their
URL. Every deck gets the normal homepage entry. Required frontmatter is `title`,
`author`, `description`, `created_at`, and `updated_at`; dates are quoted
`YYYY-MM-DD` strings. Do not put supporting Markdown files in this folder:
every Markdown/MDX file is a deck.

Start from `../md/slides/hello-slides/slides.mdx`. MDX uses our native Astro components,
with no presentation library:

```mdx
---
title: "My talk"
author: "Eoin Murray"
description: "A short description for the homepage."
created_at: "2026-10-13"
updated_at: "2026-10-13"
---

import Slide from '../../../site/src/components/slides/Slide.astro';
import SlideColumns from '../../../site/src/components/slides/SlideColumns.astro';

<Slide title="A clear opening" variant="title" eyebrow="Pinglab">

Write Markdown here, leaving blank lines inside the component.

</Slide>

<Slide title="Two columns">
  <SlideColumns ratio="wide-left">
    <div slot="left">The main explanation.</div>
    <div slot="right">Supporting evidence.</div>
  </SlideColumns>
</Slide>
```

Adjust relative imports for nested source folders. `Slide` accepts an optional
`title`, `eyebrow`, and `variant` (`default`, `title`, or `contrast`).
Use `density="compact"` for existing decks with longer text or figure legends.
`SlideColumns` accepts `equal`, `wide-left`, or `wide-right` for `ratio`, with
`left` and `right` slots.

Six reusable templates live beside `Slide` in `src/components/slides/`:

1. `TitleSlide`: `title`, optional `eyebrow`, `subtitle`, `presenter`, `date`, and cover
   `image`, `alt`, and `imageLink`. The default slot holds extra introduction.
2. `SectionSlide`: `title`, optional `eyebrow`, and a framing sentence in the
   default slot. Uses the dark green background.
3. `TextSlide`: `title`, optional `eyebrow` and `density`, Markdown in the default slot, and
   an optional named `source` slot.
   Short prose/lists automatically get larger body type and roomier list spacing;
   medium-length text gets a smaller increase, while dense content retains its
   existing size. This uses visible text length and block count, not just bullet
   count. Slides containing tables or embedded media keep their existing layout.
   Headers and source notes do not scale. Small/short screens and print retain
   the original sizing; no deck markup changes are needed.
4. `FigureSlide`: optional `title` and `eyebrow`, required `image` and `alt`, and optional
   `caption` and `source` strings. The figure fits the available slide area,
   aligned left and vertically centred, including cropped figures.
5. `FigureTextSlide`: `title`, optional `eyebrow`, `image`, `alt`, and explanatory Markdown in the
   default slot. `figureSide` is `left` or `right`; `balance` is `figure-wide`
   (default), `equal`, or `text-wide`. Also accepts `density`, `caption`, and
   `source`.

6. `HeroTextSlide`: required `title` and optional `eyebrow`. A text-only slide
   with no body, using the normal cream background and shared top-aligned header.

The optional `eyebrow` displays a small section heading above the slide title;
use it to repeat the current numbered section throughout a deck.
All templates share the figure-slide header position, title size, eyebrow size,
and spacing, independently of body density. Title, section, and hero-text slides
also use this top-aligned header rather than centering it. Contrast slides retain
their light header text on the dark background.

Figure images retain their proportions and all labels. Import PNG/JPEG assets
normally; import SVG image URLs with `?url`. Templates accept the imported
image metadata or a URL string. A metadata image can use
`crop={{ top: 5, bottom: 10 }}` to hide those percentages at display time; use
this only for an intentional crop. Source files remain intact. Figure-and-text
slides stack on mobile; long explanations can scroll on desktop.

```mdx
import FigureSlide from '../../../site/src/components/slides/FigureSlide.astro';
import figure from './images/figure.png';

<FigureSlide title="A result" image={figure} alt="Describe what the figure shows." />
```

`md/slides/journal-club-oct-5th/slides.mdx` includes text, hero-text, and figure slides.
Keep slides at the top level; put deck content inside them.
Plain Markdown or MDX without slide components starts a new slide at
each `##` heading; introductory content becomes its own slide. This grouping
happens in the browser. Without JavaScript the complete content stays readable.

The viewer provides buttons, arrow keys, Space/Shift+Space, Home/End,
Page Up/Down, horizontal touch swipes, and fullscreen (`F`).
The header shows only the deck title. Navigation and the slide counter sit at
the bottom right, with an icon-only fullscreen toggle and accessible state labels.
The current slide is linked in the URL as `#slide-3`; browser history restores
slide selection. Hidden slides cannot receive keyboard focus. Large content
scrolls within the presentation rather than being clipped; columns stack on
mobile. Browser printing shows all slides with page breaks. Existing KaTeX
support handles `$...$` and `$$` mathematics in decks too.

## Data and values

SNNLab parameter tables use `reportSnapshots` from `src/lib/report-data.ts` to
read the original saved native snapshots at render time. Supply explicit
snapshot paths; the loader uses the same selected presentation as figures and numbers:

```mdx
import { reportSnapshots } from '../../../site/src/lib/report-data';

export const modelEvidence = await reportSnapshots('exp124', {
  snapshots: {
    coba: 'coba/train/end/snapshot.json',
    ping: 'ping/train/end/snapshot.json',
  },
  allowUnavailable: true,
});
```

The loader follows the selected presentation's analysis and compute references,
checks recorded identities and input digests (including chains of reanalysis),
and uses Pingstore's `native_paths`
mapping to read original snapshot JSON. It returns unmodified snapshot metadata,
the compute configuration and dataset provenance, and the compute reference.
When an analysis combines multiple compute runs, set `computeInput` to the
recorded input role supplying the native snapshots; ambiguous selections fail.
It does not audit the complete payload or read weight arrays. Row selection,
units, labels and display-only derivations belong in the report layer, using the
shared `Table` component. Do not export table rows or separately serialized snapshot metadata through
`numbers.json`. Missing ancestors or snapshots may be shown as
unavailable; malformed records, mismatched pins and broken declared files fail.
There is no fallback to another run, current recipe defaults or old table rows.


The shared TypeScript run reader uses the public `pingstore/catalog.json` on
`https://assets.eoinmurray.info`. The catalog embeds completed run records, exact
input pins, and per-file sizes and SHA-256 checksums. JSON and media are fetched
on demand and verified; full scientific arrays are not restored for rendering.
Development uses local copies when present and otherwise reads R2. A
report uses its latest local presentation if any, otherwise the latest remote
presentation. Core owns this policy and validates digest-matched ancestry. The
TypeScript readers consume `pingstore.index/v1` through `pingstore status --json`.
Selections freeze per production build or development request. On localhost, a
reload refreshes MDX module bindings, numbers, figures, snapshots and inventory
together, without restarting the server. An existing but broken local run
fails rather than being silently replaced by R2.

Production builds use R2 exclusively and freeze catalog/selection reads for the
build. Set `PINGLAB_RUN_SOURCE=remote` to test without local evidence, `auto` for
the development union, or `local` for an offline local build. Set
`PINGLAB_RUNS_ROOT` to override the local runs directory and
`PINGLAB_ASSET_BASE_URL` to use another catalog/asset origin. Empty origin is
supported only in local mode. Browser animation JSON uses public GET/HEAD CORS.
Figures and videos use direct R2 URLs in production. Local development routes
serve local exports, proxy verified remote animation JSON, and redirect remote images/video to R2.

`RunFigure` and `PublishedFigure` resolve a figure by its filename stem within
the selected presentation: prefer `.svg`, then fall back to `.png`, regardless
of which extension the article supplies. Clicking a linked figure opens that
same resolved image in a near-fullscreen overlay with a translucent dark
backdrop. Click again anywhere or press Escape to close it and return focus
to the figure. Modified clicks retain the raw-image link behaviour.
If neither exists, rendering fails; video filenames are used unchanged. Older
PNG-only runs therefore remain usable when new presentations emit SVG.

Native MDX bindings read the selected JSON through `src/lib/report-data.ts`,
using the same frozen presentation selection as the figures. Bound
numbers and complex-table values update during rendering when selections change.
Authored calculations use ordinary JavaScript. Missing values or nonfinite results fail
rather than becoming fabricated numbers. A data report whose declared input has
no completed runset in the selected source shows an explicit Evidence unavailable
page before evaluating its MDX bindings or appendix. Its Dataset table still
explains the available inventory. If a presentation exists, missing or invalid
fields still fail. Existing direct MDX calculations remain active when evidence
is available. Authored literal numbers and interpretations remain authored;
changing the dataset does not automatically rewrite scientific conclusions.

Dataset footers and `/pingstore` use core's shared evidence classifications.
Current identifies each current runset's presentation and exact ancestry; Required
names report/protection/unfinished-work consumers. Historical holdings remain
Unresolved while active and incoming deployment evidence is unknown. Storage is
Local, R2 or Both, independently of Ran on. Verification distinguishes cached
local byte validation, metadata-only local validation, and verified R2 catalog evidence. An R2 outage stays
explicitly unresolved. Local validation caches are derived sidecars outside runs;
changed file metadata makes evidence unresolved until `pingstore status --verify`
authenticates it. Page rendering never hashes scientific arrays. Deleting caches
returns local verification to metadata-only.
Text-only dependent articles retain dependency
links without inventing unavailable data.

Experiment reports select the latest completed presentation. Their
appendices read original native snapshots and recorded configurations; historical
runs without snapshots show unavailable model values. Analytical and history-only
studies identify their different evidence requirements. No current recipe defaults
or copied parameter tables replace missing execution evidence.

Each report declares its scientific `inputs` in frontmatter. Python execution
reads its `collection` there. Pruning derives global report usage from the
latest completed presentations for declared inputs and their complete ancestry, including
cross-experiment dependencies. Unified `prune --storage` defaults to both stores
and retains current runsets, manual protection, unfinished work, shared inputs and
active/incoming deployment evidence. Status does not authorize deletion; each
cleanup requires its inspected, unchanged plan hash.
See [Pingstore Rules](../md/notes/pingstore-rules.md) for the hash-bound dry-run
and confirmation workflow. Neither operation depends on historical `writings/`.

## Publication

`.github/workflows/pages.yml` runs on pushes to `main`, manual dispatch and
`repository_dispatch` of type `pingstore-updated`. It tests the readers, registers
incoming evidence in the shared R2 coordination record and freezes the exact
remote catalog/index for rendering. `PINGLAB_CATALOG_PATH`,
`PINGLAB_REPORT_INDEX_PATH` and `PINGLAB_EVIDENCE_PATH` bind the build to those
prepared files. The rendered manifest must match the incoming registration.
Every deployment attempt registers that frozen manifest again before deploying,
including a retry of the deploy job alone. A deployment failure keeps its
incoming evidence protected because Pages may already have received the artifact.
Activation runs in a following job after the Pages environment job succeeds;
GitHub cannot report that deployment successful while its own job is still running.
A verified replacement releases completed incoming attempts while keeping the
inputs of other pending workflows.

The private `PINGLAB_RCLONE_CONFIG` repository Actions secret supplies the existing
R2 remote. The workflow installs rclone, PyYAML and boto3 without the scientific
runtime. `actions/upload-pages-artifact` and `actions/deploy-pages` deploy `site/dist`
directly. After deployment, GitHub's successful active deployment and the deployed
manifest must agree before previous evidence is released. Failed/incomplete work
keeps the previous deployment protected. An unresolved outcome keeps both roots.
There is no manual `gh-pages` branch or separate publishing framework.

```sh
cd site
bun install --frozen-lockfile
bun run build
bun run preview --port 3001
```

Report HTML, bound values and inventory tables are build-time snapshots. New
backups need another Actions run to appear; backup does not publish code or
trigger deployment automatically. Assets already referenced by deployed pages
load directly from immutable R2 run paths.

`pingstore backup` publishes a new checksummed catalog only after verifying run
bytes and complete ancestry. Bootstrap also verifies existing remote backups;
remote-only historical runs remain in the catalog. Use its separate dry-run and
hash-confirmation invocations. Do not remove local runs until they are cataloged
and a remote-only build succeeds. Scientific execution still requires local
inputs; restore selected runs and their ancestry before running new stages.

The Git-ignored PDF library stays local; do not upload the library to the public assets bucket.
Production paper links come from the tracked `papers/index.yaml` citation index.

Every build also includes `/pingstore-evidence.json`, containing the build and
revision identity, index generation, selections and exact payload/input digests.
It ships through the existing Pages workflow. The manifest records build evidence;
it does not itself prove successful deployment. Core checks the actual active
Pages deployment and its manifest before pruning. Legacy bootstrap uses the exact
authenticated successful Pages artifact; the self-contained recovered exp124 text
archive does not promote its historical Dataset labels into live run roots.

A data report with no completed runset shows “Evidence unavailable” and skips its
MDX results, parameter tables and appendices. Its missing inputs are explicit in
the build manifest. Invalid selected evidence still fails rendering; it is never
hidden by this absence handling. Listings show unavailable metrics until evidence
is available. No scientific results are invented or replaced with recipe defaults.
