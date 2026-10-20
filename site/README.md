# Pinglab Astro publishing

Run the site with Bun from `site/`:

```sh
bun install
bun run dev --background     # localhost:3000, toolbar disabled
bun run astro dev stop
bun run build                # disposable dist/
```

Pixi is prebundled when development starts, including animation scripts outside
`site/`. Development and builds use separate Vite caches under
`node_modules/.vite/dev` and `node_modules/.vite/build`, so a build cannot replace
the dependency URLs used by a running preview.

Page rendering errors are isolated by `src/middleware.ts`. The affected URL
shows “An error occurred.” and the error message with a link back to the articles;
other pages continue rendering. HTML is buffered so component errors cannot
leave a partially rendered page. Development returns HTTP 500; static builds
write the error page at the affected URL and log the failure. Static hosting
serves that generated file with its normal success status. Compiler, import-resolution,
and content-schema errors can still prevent a build before rendering begins.

Exp124 renders its current authored report even when its historical pinned
payload is missing. Its report-local evidence loader returns unavailable only
for an absent pinned payload; it never selects a newer diagnostic run, and invalid
existing JSON still fails. Figures depend on the original presentation payload.
Parameter tables independently follow that same pin's ancestry to the original
compute snapshots; absent native snapshots remain explicitly unavailable.
The earlier published HTML is separately
labelled at `/recovered/exp124/published.html`, with assets under
`public/recovered/exp124/`. It is an archive, not a recovered Pingstore run;
its historical Dataset inventory is not a current local inventory.

Development run assets are read directly from the requested presentation export;
they do not scan every article's runs. Missing assets return HTTP 404. During a
static build, unavailable run exports are logged and skipped in asset enumeration,
while data or figure errors remain visible on their own article. Shared article
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
local Pingstore, including historical runs, without upstream or hidden runs.
These are local inventory counts, not a claim that every run informed the report.
Static builds freeze counts; development recalculates on refresh. A report that
cannot render shows unavailable word/figure counts in listings rather than taking
the listing offline. No authored count fields or article date changes are needed.
All article types sort together by decreasing authored `updated_at`, with article
ID and then URL ascending to break ties, in both development and static builds.
The homepage, `/notes/`, and `/collections/<collection>/` share the same
collection sidebar and article layout, including aligned navigation. Collection
pages show only that collection's articles and highlight its sidebar link.
`/collections/` lists
clickable collection names, including Notes, without article previews. Distill's official
v2 runtime and styles are self-hosted; the article shell supplies the byline,
sticky contents, and appendix. Optional `appendix.mdx` and `references.mdx` files
supply footer content. Put network diagrams first in `appendix.mdx`, followed by
the main parameter table and supporting detail. The appendix's own section headings
appear to the left of their content on desktop and above it on smaller screens,
and are included in the table of contents. Existing `parameters.mdx` files
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

`reportData` honors the article's frontmatter `runs` pins and otherwise uses
this build/session's frozen latest selection. Pass `{ experiment: 'exp082' }`
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

`/papers` lists PDFs from the repository's `papers/` directory, including
subfolders, newest publication year first, then alphabetically by title, with
unindexed PDFs last and direct PDF links. The left column
shows the publication year in the article list's experiment-ID style, followed
by the permanent paper ID in the smaller date style underneath on desktop.
Hidden files and symlinks are excluded. The homepage and paper pages share
Articles and Literature sidebar lists; the search palette also links to
the library. Development reads the directory on refresh; static builds emit
the PDFs under `/papers/`, so additions and removals need a rebuild to publish.

`papers/index.yaml` holds citation metadata keyed by relative PDF filename.
Each entry requires a unique `id` such as `P01`, `title`, a nonempty `authors`
list, and integer `year`; `venue` and `doi` are optional.
The page displays this metadata with the PDF link, falling back to the filename
for unindexed PDFs. Invalid metadata fails rendering/building; entries without
matching PDFs produce warnings. An absent index is allowed. The index is tracked
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
snapshot paths and an article frontmatter presentation pin:

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

The loader follows the pinned presentation's analysis and compute references,
checks recorded identities and input digests, and uses Pingstore's `native_paths`
mapping to read original snapshot JSON. It returns unmodified snapshot metadata,
the compute configuration and dataset provenance, and the compute reference.
It does not audit the complete payload or read weight arrays. Row selection,
units, labels and display-only derivations belong in the report layer, using the
shared `Table` component. Do not export table rows or separately serialized snapshot metadata through
`numbers.json`. Missing ancestors or snapshots may be shown as
unavailable; malformed records, mismatched pins and broken declared files fail.
There is no fallback to another run, current recipe defaults or old table rows.


The simple TypeScript Pingstore reader resolves a latest presentation once per
build/dev session, or uses an article's explicit frontmatter `runs` selections.
The ordinary numbers and figure readers do not audit schema, checksums or
ancestry; the snapshot loader performs the metadata checks described above. Restart the dev server
to pick up newly completed runs. Figures/media come directly from selected
exports; only disposable builds copy them.

`RunFigure` and `PublishedFigure` resolve a figure by its filename stem within
the selected presentation: prefer `.svg`, then fall back to `.png`, regardless
of which extension the article supplies. Links open the same resolved file.
If neither exists, rendering fails; video filenames are used unchanged. Older
PNG-only runs therefore remain usable when new presentations emit SVG.

Native MDX bindings read the selected JSON through `src/lib/report-data.ts`,
using the same run pins as the figures. Bound
numbers and complex-table values update during rendering when selections change.
Authored calculations use ordinary JavaScript. Missing values or nonfinite results fail
rather than becoming fabricated numbers. Migrated articles without a local
presentation retain their authored text and calculations, with explicit
unavailable-value and unavailable-figure markers. If a presentation exists,
missing or invalid fields still fail. Existing direct MDX calculations also
remain active. Authored literal numbers and interpretations remain authored;
changing the dataset does not automatically rewrite scientific conclusions.

Dataset footers show completed run history, producing execution timing for
imported runs, and upstream/downstream article dependencies. Bold run IDs mark
presentations used by the article and every run in their recorded input chains,
including compute, analyse and upstream-experiment runs. All these rows are
highlighted and marked “Used”; unrelated historical runs remain unmarked.
Only selected presentations list report asset counts in Source.
Text-only dependent articles retain dependency
links without inventing unavailable data.

Seven articles have no local presentation: exp022, exp024, exp047, exp054,
exp080, exp081, and exp117. Their complete authored accounts, including
equations, captions, specification tables and appendices, are preserved;
an availability notice distinguishes the account from locally checkable data.

Each report declares its scientific `inputs` in frontmatter. Python execution
reads its `collection` there. Pruning derives global report usage from pinned
or latest unpinned declared inputs and their complete ancestry, including
cross-experiment dependencies. Unused runs require a manual `.protected` file
to retain them and their ancestry, apart from incomplete-input and scope
safeguards. HPC origin and unselected latest runs do not confer protection.
See [Pingstore Rules](../md/notes/pingstore-rules.md) for the hash-bound dry-run
and confirmation workflow. Neither operation depends on historical `writings/`.

## Publication

See `../tools/publishing/README.md`. Build a frozen local Astro publication with:

```sh
uv run python tools/publishing/publish.py build
(cd site && bun run astro preview --outDir ../.publication/site --port 3001)
uv run python tools/publishing/publish.py publish
```

Run those commands from the repository root. The publisher retains a build
receipt, checks the exact static output before pushing it to `gh-pages`, and
preserves the remote CNAME and existing PR previews.
