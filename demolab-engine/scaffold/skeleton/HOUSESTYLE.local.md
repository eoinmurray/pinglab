<!-- mode: extend -->

# HOUSESTYLE.local — your lab's house style (optional)

This file is **yours**. Edit it to bend demolab's default house style
([`demolab-engine/guides/HOUSESTYLE.md`](demolab-engine/guides/HOUSESTYLE.md)) to your lab.
It survives `"update demolab"` (like `demolab.yaml`); delete it to use the default unchanged.

**How it combines** — set by the mode marker on line 1:

- `<!-- mode: extend -->` (default): your rules below **add** to demolab's and **override**
  any default they restate (cite it by H-number or topic). The defaults fill everything you
  leave unsaid.
- `<!-- mode: replace -->`: an agent ignores demolab's house style entirely and follows
  **only** this file. The [`RULES.md`](demolab-engine/guides/RULES.md) contract still
  applies; only *taste* is overridable.

Write rules however you like; number them `L1`, `L2`, … if you want them citable. A lab that
wanted the opposite of some demolab defaults might write:

    **L1 — Em-dashes are fine.** Overrides H7; use them freely.
    **L2 — The primary trace is brand teal (#008080), not black.** Overrides H13.
    **L3 — Cite in APA.** A rule demolab's default doesn't cover.

There are no active rules below, so this repo uses demolab's defaults as-is. Add yours here.
