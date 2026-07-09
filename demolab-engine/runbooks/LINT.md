# Runbook: Lint against the house style

> Check a repo's writings **and the figures they embed** against
> [`HOUSESTYLE.md`](../guides/HOUSESTYLE.md) (H1–H24), report each violation with the H-rule it
> breaks and a `file:line` (or figure path), and fix only what the user approves.

## When to use
When you want the prose and plots to read like a person — not an LLM — wrote them, and to
conform to the house style before publishing. This is the **style** pass — one of three:
*Doctor* audits the structural RULES, *Lint* (here) audits the prose **and the figures**, and
*Red-team* audits whether the result is true and defensible. Run all three for a full check.
`HOUSESTYLE.md` is the source of truth for every rule: this runbook cites it, it does not
restate it. Scope is `writings/*.typ` (the published prose) **and the rendered figures it
embeds** (`artifacts/data/<id>/*.svg`, `*.png`) — the plots are half of a writing and are lint
targets, not just decoration. The framework docs are reference material and out of scope unless
the user asks.

**Don't lint the prose and stop.** The figure rules (H10–H15) are the most-skipped part of this
pass, because a grep can't see a plot and an SVG *looks* like text but isn't (see §2b). A lint
that never opened a figure is half a lint — §2b is not optional.

**The `log` article is exempt from the prose rules.** In an autoresearch program (see
[`../guides/AUTORESEARCH-RULES.md`](../guides/AUTORESEARCH-RULES.md)) the `log` is a
chronological, append-only notebook — it deliberately doesn't lead with a claim, define every
symbol, or read like a polished article, so **skip H2–H5 and the voice/structure judgment checks
for `writings/log.typ`**. What still applies: it's a published page, so the can't-drift rules
hold — no hand-typed run numbers (H9), citations through the system (H24), and figures it embeds
obey H10–H15. The `plan` article is a normal article and gets the full pass.

**Lint against the *effective* style.** First check for a root `HOUSESTYLE.local.md`. If it's
absent, use these defaults. In `extend` mode (default) its rules override or add to the defaults
— apply the merged set, and where the two conflict the local file wins. In `replace` mode (its
first line is `<!-- mode: replace -->`) lint against **only** the local file and ignore the
H-rules entirely. Report against whichever set is in force. The checks below are for the default
style; drop or adjust any the local file overrides.

## What it does

0. **Build is green.** `task build` must compile first. Linting broken Typst is noise.

1. **Mechanical checks (grep — each hit is a candidate, not a verdict).**

   ```sh
   # H7 — no em-dashes in prose. Titles, headings, and labels are fine, so read each hit in
   # context before flagging it.
   grep -nE '—' writings/*.typ

   # H6 — tilde misuse. In Typst prose `~` is a non-breaking space; for "approximately" use ≈.
   grep -nE '~' writings/*.typ

   # Voice: LLM-tell vocabulary. A hit is a smell, not proof — read the sentence.
   grep -inE 'delve|leverage|utiliz|seamless|robust|comprehensive|nuanced|multifaceted|underscore|pivotal|tapestry|realm|landscape|in today.?s|it.?s (important|worth) (to note|noting|mentioning)|game.?chang|revolutioniz|unlock the' writings/*.typ

   # Voice: antithesis / false-balance scaffolding ("not just X but Y", "while … also").
   grep -inE 'not (just|only) .* but( also)?|while .*, .* (also|there)' writings/*.typ

   # H24 — references must go through the system (#cite + #reference-list), never hand-rolled.
   # (a) hand-typed inline cites: a bracketed number that isn't a #cite(...). Math intervals/indices
   #     ([0,1], matrix delims) are false positives — read each hit.
   grep -nE '\[[0-9]+( *, *[0-9]+)*\]' writings/*.typ
   # (b) a manually written References section (should be #reference-list(...)).
   grep -inE '^\s*=+ +references\b' writings/*.typ
   # (c) DOI links typed by hand (reference-list builds the doi.org URL from a `doi:` field, so a
   #     literal doi.org in a writing means the reference list was hand-rolled).
   grep -in 'doi\.org' writings/*.typ
   # (d) parenthetical author–year cites, e.g. "(Smith 2020)" — a citation style that bypasses #cite.
   grep -nE '\([A-Z][a-zA-Z]+ (et al\.?,? )?[0-9]{4}[a-z]?\)' writings/*.typ

   # --- Figures (H10–H15): the mechanical, greppable slice. The *visual* checks are §2b — these only
   # catch what lives in the markup. Plots live in artifacts/data/<id>/.

   # H10 — a line plot (traces, curves, sparse scatter) belongs in SVG; a heatmap / dense scatter /
   # raster in PNG. List both, then judge each PNG in §2b: genuinely dense (fine) or a soft line plot
   # that should be vector?
   ls artifacts/data/*/*.svg artifacts/data/*/*.png 2>/dev/null

   # H13 — palette. Each plot is near-black ink + at most ONE accent colour. Dump the distinct hexes per
   # SVG; ignoring #000000 / greys / #ffffff, more than one chromatic hue is a candidate.
   for f in artifacts/data/*/*.svg; do echo "== $f =="; grep -oiE '#[0-9a-f]{6}' "$f" | sort -u; done

   # H14 — white background. Every SVG should paint a #ffffff page rect; list any that don't mention it.
   grep -L '#ffffff' artifacts/data/*/*.svg 2>/dev/null

   # H14 — alt text. Each embedded #image needs an `alt:`. List the embeds, then confirm each call
   # carries an alt: argument (grep can't tell across a multi-line call — read the hits).
   grep -nE '#?image\("' writings/*.typ
   ```

2. **Judgment checks (read each writing — no grep suffices).**
   - **§H1 titles.** `meta.title` is a bare sentence, no id prefix.
   - **§H2–H4 prose.** Reports partial/negative results honestly; leads with the claim; no repo
     bookkeeping in the body.
   - **§H5 define every symbol.** After each equation, every symbol is defined (a list for dense
     blocks; the Hodgkin–Huxley sheet is the model).
   - **§H7 em-dashes in prose.** For each grep hit from §1, confirm it is running prose (a
     violation) rather than a title / heading / label (fine).
   - **§H16–H20 captions.** Each caption is standalone, follows the lead → axes-with-units +
     panels → takeaway anatomy, uses present tense, and pulls its numbers from the run (`#run…`)
     rather than a hand-typed literal.
   - **§H9 numbers (read for these — no grep).** Every number a run *produced* — in prose,
     captions, and tables — traces to `numbers.json` via `#run…` / `#numbers-table(...)`,
     **never** a typed literal that can drift. Hardcoded run figures are almost always a
     violation to fix. The only literals allowed are values that are *not* run outputs: a
     model's fixed coefficients in an equation, a structural count, or a paper's target quoted
     for comparison (§6.6).
   - **§H24 citations.** If a writing cites anything at all, it **must go through the system** —
     `#cite(...)` inline and `#reference-list(...)` for the list (RULES §6.6). Any hand-rolled
     alternative is a violation to fix: hand-typed `[1]` brackets, a manually written
     `== References` section, directly-typed `doi.org` links, or author–year cites like
     "(Smith 2020)" — each bypasses the numbering, linking, popover, and new-tab DOI the helpers
     provide. Each reference carries a DOI where one exists.
   - **Voice (the real one).** Sentence rhythm varies (not every sentence the same medium
     length); no rule-of-three-everything; it takes a position rather than hedging both sides.
     Does it read like a person or a model?

2b. **Figures — open every rendered figure (H10–H15).**

   The plots are the other half of a writing, and neither a grep nor the caption can see them —
   **you must open each rendered figure and look at it.** For every asset the writings embed
   (`artifacts/data/<id>/*.svg`, `*.png`), read the image and check it against the list below.

   > **The SVG trap.** An SVG *looks* greppable but isn't: matplotlib writes axis labels as glyph
   > `<use>` paths, not `<text>` nodes, so the label text is invisible to grep — "is the y-axis
   > labelled mV?" cannot be answered from the source, only from the picture. A PNG is opaque to
   > grep entirely. The §1 greps catch the palette and background; **everything below needs your
   > eyes on the rendered figure**, both formats. If you skipped this because "it's an image,"
   > that is exactly the miss this section exists to stop.

   For each figure:
   - **§H17 no plot title, §H11 labelled axes.** There is *no* title baked into the figure (the
     caption carries the takeaway); both axes are labelled **with units**.
   - **§H11 legibility at column width.** Text reads at the ~6.5 in column — labels and ticks are
     neither tiny nor huge (roughly `font.size` 10, `axes.labelsize` 11, ticks 9; H11's table).
     A figure whose fonts are clearly off was sized wrong.
   - **§H12 one aspect, sensibly wide.** Every figure in the writeup shares one aspect; none is
     portrait or gratuitously square, and there's no log axis unless the data demands it.
   - **§H13 ink + one accent, grayscale-safe.** Near-black by default; at most a single accent
     colour, and wherever colour separates traces the line-style/marker varies too, so it
     survives a grayscale print and colour-blind readers.
   - **§H14 white background.** Plot area and margins are white — not grey, not transparent.
   - **§H10 right format for the content.** A line plot is crisp SVG; a heatmap / dense scatter /
     raster is PNG at ~240 dpi. A soft or pixelated line plot is the tell it should have been
     vector.
   - **§H15 consistent, central style.** The figures look like one lab drew them (a shared
     matplotlib style) — divergent fonts, colours, or sizes between figures mean per-runner
     drift.

   A *rendering* (an mp4, e.g. `mujoco`) is the exception: it plays on the web and the PDF just
   notes "view the web edition" — no still to inspect (H14).

3. **Report.** Present one report grouped by severity:
   - **Prose tells** (em-dashes in prose, LLM vocabulary, false balance) — fix; they're the
     loudest giveaways.
   - **Can't-drift** (hand-typed run numbers, §H9/H16–H20) — fix before publishing.
   - **Figures** (§H10–H15 from §2b) — an unlabelled axis, a baked-in title, or a soft PNG line
     plot is a fix; aspect/palette/style drift is often advisory. A figure fix usually means
     editing the *runner*, not the writing.
   - **Advisory** (voice, structure, define-terms consistency) — the user's call.

   For each finding: name the rule (link `../guides/HOUSESTYLE.md`), the `file:line` **or figure
   path**, and the rewrite. Apply only what the user approves, then re-run the relevant checks to
   confirm. **State plainly whether §2b was done** — if you didn't open the figures, say so
   rather than implying the plots passed.

---

## Agent contract
- **Triggers** — `LINT`, "lint", "lint the writings", "lint the repo against the house style",
  "check the house style", "does this read like an LLM wrote it".
- **Gates** — §0 must hold (broken Typst is noise); resolve the *effective* style first
  (`HOUSESTYLE.local.md` in `extend`/`replace` mode, else the H-defaults) and report against
  whichever set is in force.
- **Report & apply** — drive it interactively: run the mechanical checks, read each writing for
  the judgment ones **and open every figure for §2b**, collect the hits, present one report
  grouped by rule, then offer to fix. Apply only what the user approves, then re-run to confirm.
  **State plainly whether §2b was done.**
