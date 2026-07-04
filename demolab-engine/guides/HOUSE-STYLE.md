# HOUSE-STYLE — how demolab writings read and look

Preferences for authoring a writing (§6): how prose, math, figures, and structure should
read. These are *style*, softer than the invariants in [`RULES.md`](RULES.md) — but follow
them unless you have a reason not to, so the lab reads as one voice. Rules are numbered
`H<n>` (citable, like RULES' `§N.M`). Terms are in [`GLOSSARY.md`](GLOSSARY.md).

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

## Numbers & figures

**H9 — Never hand-type a run's numbers.** Any number produced by a run comes from `numbers.json` via `#numbers-table(...)` / `json(...)` (RULES §6.2, §5.4). If you cite a value in prose, it must trace to the run too — a hand-typed literal can drift and is the thing demolab exists to prevent.

**H10 — Captions are standalone.** Write each figure caption for a cold read: name the axes/panels and conditions, define any term inline (raster, rate, SNR…), and end with the takeaway — so a reader who jumps straight to the figure understands it without the body.

**H11 — Plots are black by default.** Use near-black ink for a trace; introduce a *single* accent colour only to distinguish two-or-more traces sharing one axis. Separate subplots stay black — don't colour panels for decoration. No ad-hoc hex per condition.

**H12 — One aspect, no gratuitous log axes.** Pick a figure aspect and keep it across a writeup (16:9 is a sound default); a square only when the data is genuinely square. No log axis unless the data demands one. Use `≈` (not `~`) in figure labels too.

## Structure

**H13 — Abstract → Methods → Results is a good default.** A short abstract (hypothesis → what you ran → verdict), Methods as an *itemised* list of steps, and Results carried by figures + captions. It's a default, not a mandate — an article (G1) that's a reference sheet or derivation can ignore it.

**H14 — The writeup is the recipe.** Reproducing a result shouldn't require remembering flags: the run's parameters live in the writeup (as a `numbers-table` from the run), and the exact CLI call is captured in the tool's `run.sh`. A reader should be able to rebuild the result from the page.

**H15 — Enumerate almost everything.** Reference and procedural material gets numbered, citable anchors — rules (`§N.M`), glossary terms (`G<n>`), style points (`H<n>`), runbook steps — so any item can be cited precisely and scanned at a glance. This applies to the guides and runbooks *and* to a writing's own Methods steps. The *"almost"*: genuinely non-sequential lists — categorical sets (the firewall zones, an update's tiers), either/or options, conjunctive checklists — stay bullets, because numbering them implies an order that isn't there.
