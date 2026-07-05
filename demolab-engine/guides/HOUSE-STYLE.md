# HOUSE-STYLE — how demolab writings read and look

Preferences for authoring a writing (§6): how prose, math, figures, captions, and
structure should read. These are *style*, softer than the invariants in
[`RULES.md`](RULES.md) — but follow them unless you have a reason not to, so the lab reads
as one voice. Rules are numbered `H<n>` (citable, like RULES' `§N.M`). Terms are in
[`GLOSSARY.md`](GLOSSARY.md).

## Prose

**H1 — Bare-sentence titles.** A `meta.title` is a plain sentence, *not* prefixed with the id (`"LIF voltage traces…"`, never `"exp000 — LIF…"`). The id, date, status, and collection already appear in the entry's meta strip — the title shouldn't repeat them.

**H2 — Report results honestly.** Include partial and negative results; fold the caveats into the prose or the figure caption rather than hiding them. Prefer one coherent writeup over several linked stubs.

**H3 — Say it, then show it.** Lead with the claim in plain language; let the figure and the numbers carry the evidence. Keep motivation short — a reader came for the result.

**H4 — Plain prose, no repo bookkeeping.** Write for a scientific reader, not a changelog. Don't narrate file moves, commits, or tooling in the body of a writing.

## Math & symbols

**H5 — Define every term.** After an equation, define *every* symbol in it in a list — `$tau_m$` is the membrane time constant, `$V_"th"$` the threshold, and so on. A reader should never have to guess what a symbol means. This is the single most important house rule for demolab's math-heavy pages.

**H6 — `≈`, never `~`, for "approximately".** In prose use `≈`; in math use `approx`. Never a tilde: in Typst markup `~` is a *non-breaking space*, so it silently won't render as a tilde at all. Reserve `∼` (`tilde`) in math for "distributed as" (`$u tilde "Uniform"(0,1)$`).

**H7 — En-dashes for ranges.** `30–90 Hz`, `draft–final` — an en-dash (`–`), not a hyphen. (In Typst, `--` renders as an en-dash.)

**H8 — Native Typst math, not images.** Write equations in Typst (`$…$` inline, block for display) so they render as selectable MathML on the web and typeset in the PDF — never paste an equation as an image.

## Numbers

**H9 — Never hand-type a run's numbers.** Any number produced by a run comes from `numbers.json` via `#numbers-table(...)` / `json(...)` (RULES §6.2, §5.4). If you cite a value in prose *or a caption* (H19), it must trace to the run too — a hand-typed literal can drift and is the thing demolab exists to prevent.

## Figures & plots

Figures are drawn by a runner from the tool's CSV (`.py` + matplotlib, §4.2), then embedded in *both* the HTML and the PDF from **one rendered asset**. That dual target drives these rules — size for the sharper of the two media, and keep the look in one place.

**H10 — Vector where you can, raster where you must.** Line plots (traces, curves, sparse scatter) → **SVG**: vector-sharp at any zoom on the web *and* in the PDF, and usually smaller. Dense or photographic content (heatmaps, big scatter, rasters, renderings) → **PNG** at ~240 dpi (H11). The trap: a 96-dpi PNG looks fine in the PDF and soft on a retina screen — one asset serves both, so size it for the sharper one.

**H11 — Design for the display width.** A figure shows at 100% of the column in both targets — **≈6.3 in** in the PDF, **≈7.3 in** on the web (the 46em column ≈ 700 CSS px). matplotlib font sizes are points *relative to the figsize*, so what the reader sees is:

> apparent pt = matplotlib pt × (display width ÷ figsize width)

Set the **figsize width ≈ 6.5 in** and that ratio ≈ 1 — matplotlib points map ~1:1 to what's seen, on both targets, so you can size fonts directly. Concretely: `font.size` 10, `axes.labelsize` 11, tick labels 9, legend 9, `linewidth` 1.6, and *no plot title* (the caption carries the takeaway, H17). Render PNG at **dpi 240** — 6.5 × 240 = 1560 px, ≥2× the web column (crisp on retina) and ~240 DPI in the print block; 200 is the floor, below that it's soft on retina. SVG ignores dpi. Always `savefig(bbox_inches="tight")` so no dead margin skews the aspect.

**H12 — One aspect, sensibly wide.** Keep a single aspect across a writeup; width fixed at the column (~6.5 in), height from the ratio:

| Ratio | figsize (in) | Use |
|--|--|--|
| **16:9** | 6.5 × 3.66 | time series, voltage traces, slides — **default** |
| 3:2 | 6.5 × 4.33 | general single plot |
| 2:1 | 6.5 × 3.25 | long / wide series |
| stacked panels | 6.5 × ≤5.5 | multi-panel (raster + current); cap the height so it fits a page/slide |
| 1:1 | ~5 × 5 | correlation matrices, phase portraits — *only when the data is square*, and drop the width so it isn't huge |

Never **portrait** — it reads badly on wide screens and slides. No **log axis** unless the data demands one. Use `≈` (not `~`) in labels too (H6).

**H13 — Black by default; earn every colour.** Use near-black ink for a trace; introduce a *single* accent colour only to tell two-or-more traces sharing one axis apart. Separate subplots stay black — no decorative per-panel colour, no ad-hoc hex. When you *do* use colour, also vary line-style/marker so the figure survives a **grayscale print** and colour-blind readers.

**H14 — White background, alt text.** Save figures on a **white** background (both targets are white — a transparent or grey box looks wrong), and give each `#image` an `alt:` line so the web is accessible and the meaning survives a failed load. (A *rendering* is the exception: its mp4 plays on the web and the PDF just notes "view the web edition" — that's fine, no still needed.)

**H15 — One plot style, centrally.** Every figure is drawn by a runner, so the look — fonts, sizes, colours, aspect, dpi, SVG-vs-PNG choice — belongs in *one* shared style the runners import, not re-set per runner (that's how a lab drifts). A minimal matplotlib block encoding H11–H14:

```python
import matplotlib as mpl
mpl.rcParams.update({
    "figure.figsize": (6.5, 3.66),   # 16:9 at column width  (H11–H12)
    "savefig.dpi": 240, "savefig.bbox": "tight",             # crisp retina + print (H10–H11)
    "font.size": 10, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",  # (H14)
})
# line plots → fig.savefig("fig.svg");  dense / raster → fig.savefig("fig.png")  (H10)
```

## Captions

**H16 — Standalone, cold-read.** Write each caption so a reader who jumps straight to the figure understands it without the body. The body never carries load the caption should.

**H17 — Caption anatomy: lead → body → takeaway.** Three parts, in order: (1) a **lead clause** — what the figure shows, sentence case, no result yet ("Membrane potential of a single LIF neuron under constant input."); (2) the **body** — name each axis *with units*, label and describe each panel (`(a)`/`(b)`, top/bottom), state the conditions/parameters needed to read it, and define every symbol/abbreviation inline (mirrors H5); (3) the **takeaway** — one clause on what to conclude, descriptive and honest (H2), not overclaiming.

**H18 — Present tense, describe the figure.** "The trace rises", "shows", "resets at threshold" — describe what's on screen, not a narrative past.

**H19 — Caption numbers come from the run.** Interpolate `#run.lif.firing_rate_hz`, never type "90 Hz". This is the one caption rule only demolab can enforce, and it closes H9's gap (which covered body + tables): a caption that hand-types a number can drift too. If the value you need isn't reported, extend the tool's `headline_metrics` to surface it — the right pressure, pushing numbers into the provenance-tracked record.

**H20 — Concise.** A few sentences: complete for a cold read, but not an essay.

## Structure

**H21 — Abstract → Methods → Results is a good default.** A short abstract (hypothesis → what you ran → verdict), Methods as an *itemised* list of steps, and Results carried by figures + captions. It's a default, not a mandate — an article (G1) that's a reference sheet or derivation can ignore it.

**H22 — The writeup is the recipe.** Reproducing a result shouldn't require remembering flags: the run's parameters live in the writeup (as a `numbers-table` from the run), and the exact CLI call is captured in the tool's `run.sh`. A reader should be able to rebuild the result from the page.

**H23 — Enumerate almost everything.** Reference and procedural material gets numbered, citable anchors — rules (`§N.M`), glossary terms (`G<n>`), style points (`H<n>`), runbook steps — so any item can be cited precisely and scanned at a glance. This applies to the guides and runbooks *and* to a writing's own Methods steps. The *"almost"*: genuinely non-sequential lists — categorical sets (the firewall zones, an update's tiers), either/or options, conjunctive checklists — stay bullets, because numbering them implies an order that isn't there.
