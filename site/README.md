# Pinglab Astro publishing

Run the site with Bun from `site/`:

```sh
bun install
bun run dev --background     # localhost:3000, toolbar disabled
bun run astro dev stop
bun run build                # disposable dist/
```

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
browser JavaScript lives in `experiments/expXXX/js/`; article components import
it from `../js/`. Shared site components and scripts remain under `site/src/`.
Notes live in `md/notes/` as MDX and belong to the Notes collection
at `/collections/notes/`. Each note requires `title`, `author`, `description`,
`created_at`, `updated_at`, and `collection` frontmatter including `notes`; dates use quoted `YYYY-MM-DD` strings.
The filename determines the URL, for example `/notes/compute-options/`.
Notes and experiments both use `Report.astro`, including its byline, contents
sidebar, article spacing, and annotations. Optional `abstract` frontmatter is
supported for both; experiment IDs and Dataset footers require experiment metadata.
The shared page layout declares each page's canonical URL on
`https://pinglab.eoinmurray.info`, with a trailing slash and no query or fragment.
Hypothes.is uses this identity to share article annotations between production
and localhost. Local annotations therefore belong to the production article too.
The homepage lists reports, notes, and slides with the same article previews.
Notes and slides appear first, ordered by update date; reports follow in descending
ID order. `/notes/` and `/collections/<collection>/` use the homepage list with a
collection header and only that collection's articles. `/collections/` lists
clickable collection names, including Notes, without article previews. Distill's official
v2 runtime and styles are self-hosted; the article shell supplies the byline,
sticky contents, and appendix. Optional `parameters.mdx`, `appendix.mdx`, and
`references.mdx` files supply footer content. The canonical scientific-writing
rules are preserved in `../md/notes/writing-guide.md`.
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

Cells accept strings, finite numbers, `{ math: "..." }` for KaTeX, or
`{ markdown: "..." }` for authored rich text and mixed text/mathematics.
Optional `caption`, `id`, and `label` props supply a plain caption, anchor,
and accessible region name. Header and row widths must match. Plain cells are
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

The collections index and individual collection headers show the description
configured in `src/lib/collections.ts`, a count of distinct article URLs, and
the latest member article's `updated_at` date. Articles in multiple collections
count once in each; latest activity describes article edits, not experiment execution.

Source tags remain internal metadata for data availability
and author-assigned review status. Citation previews appear on hover or keyboard focus; clicks still
navigate to references. Report and note pages embed Hypothes.is with highlights visible
and the sidebar closed by default. Readers can open it from the Hypothesis
toolbar or by clicking a highlight. The homepage and collection pages do not
load the annotation client. Interactive run selectors are absent.

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

Five reusable templates live beside `Slide` in `src/components/slides/`:

1. `TitleSlide`: `title`, optional `subtitle`, `presenter`, `date`, and cover
   `image`, `alt`, and `imageLink`. The default slot holds extra introduction.
2. `SectionSlide`: `title`, optional `eyebrow`, and a framing sentence in the
   default slot. Uses the dark green background.
3. `TextSlide`: `title`, optional `density`, Markdown in the default slot, and
   an optional named `source` slot.
4. `FigureSlide`: optional `title`, required `image` and `alt`, and optional
   `caption` and `source` strings. The figure fills the available slide area.
5. `FigureTextSlide`: `title`, `image`, `alt`, and explanatory Markdown in the
   default slot. `figureSide` is `left` or `right`; `balance` is `figure-wide`
   (default), `equal`, or `text-wide`. Also accepts `density`, `caption`, and
   `source`.

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

`md/slides/journal-club-oct-5th/slides.mdx` uses all five templates.
Keep slides at the top level; put deck content inside them.
Plain Markdown or MDX without slide components starts a new slide at
each `##` heading; introductory content becomes its own slide. This grouping
happens in the browser. Without JavaScript the complete content stays readable.

The viewer provides buttons, arrow keys, Space/Shift+Space, Home/End,
Page Up/Down, horizontal touch swipes, and fullscreen (`F`).
The current slide is linked in the URL as `#slide-3`; browser history restores
slide selection. Hidden slides cannot receive keyboard focus. Large content
scrolls within the presentation rather than being clipped; columns stack on
mobile. Browser printing shows all slides with page breaks. Existing KaTeX
support handles `$...$` and `$$` mathematics in decks too.

## Data and values

The simple TypeScript Pingstore reader resolves a latest presentation once per
build/dev session, or uses an article's explicit frontmatter `runs` selections.
It does not perform schema/checksum/ancestry validation. Restart the dev server
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
presentations used by the article. Text-only dependent articles retain dependency
links without inventing unavailable data.

Seven articles have no local presentation: exp022, exp024, exp047, exp054,
exp080, exp081, and exp117. Their complete authored accounts, including
equations, captions, specification tables and appendices, are preserved;
an availability notice distinguishes the account from locally checkable data.

Each report declares its scientific `inputs` in frontmatter. Python execution
reads its `collection` there; pruning protects explicit frontmatter `runs` pins
and treats an existing `report.mdx` as an active article. Neither operation
depends on the historical `writings/` directory.

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
