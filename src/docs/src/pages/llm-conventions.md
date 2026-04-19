---
layout: ../layouts/MarkdownLayout.astro
title: "LLM Conventions"
---

# LLM Conventions

Hard rules for editing the docs and notebook. For repo layout, workflow narrative, and how to run things, see [Introduction](/introduction/).

## Invariants and gotchas

Facts about this repo that are load-bearing and easy to get wrong.

- **Trial length defaults to 600 ms** (*--t-ms 600*). Shorter trials emit a warning — the harness will let you run them but flags that the transient window is too short.
- **Raw run outputs go to src/artifacts/; published figures go to src/docs/public/figures/notebook/&lt;slug&gt;/.** *oscilloscope.py* writes only to its *--out-dir* (typically under *src/artifacts/*). The notebook notebook runner reads from those run dirs and writes published figures directly into *src/docs/public/figures/notebook/&lt;slug&gt;/*. There is no separate freeze step.
- **src/artifacts/ is gitignored.** Nothing under it is part of the repo; it is reproducible from code + config + seed.
- **Poisson input encoding is frozen across Δt-sweeps.** Each image is encoded once at the finest sweep Δt, then OR-pooled to the target Δt. This eliminates Poisson resampling as a confound.
- **uv for Python, bun for JavaScript.** No *pip install*, no *npm install*. Lock files are *uv.lock* and *bun.lock*.

## Conventions

These rules apply when editing the docs and notebook. They are the source of truth — if anything on the [Introduction](/introduction/) page conflicts, these win.

### 1. Notebook-entry H2 skeleton

Every notebook entry under *src/docs/src/pages/notebook/* uses the same five H2 headings, in this order:

1. **Introduction** — what was built, wired, or changed, and why
2. **Method** — how the experiment was run, including methodological notes
3. **Findings** — results (use H3 subsections for multiple results); mechanism analysis nests here, not as its own H2
4. **Implications** — what it means for the paper or project
5. **Next steps** — what is still to do

The H1 stays entry-specific (the entry title). The skeleton is for H2s only. *Why:* notebook-entry navigation should be predictable across the site.

Paper-style drafts with custom sectioning can opt out by setting *structure: paper* in frontmatter. That flag also exempts the entry from convention 6 (caption-after-image), since paper-style drafts often use surrounding prose rather than italic captions.

### 2. Entry numbering and date format

Every notebook entry has a **global sequential number**, zero-padded to 3 digits (*001*, *002*, …). It is the entry's primary identifier and appears in the filename, URL, frontmatter (*entry: &lt;N&gt;*), byline, and home-page list. Numbers never change once assigned. Reserve the next number by *ls src/pinglab/notebook/* and adding 1 to the highest.

Visible dates use the long-form, day-of-week-first format: *DayName, Month D YYYY*.

Three places per entry combine these:

1. Frontmatter title — *title: "001 — &lt;title&gt;"* (number only; date omitted here so tabs stay compact)
2. Italic byline under H1 — *001 · Thursday, April 16 2026*
3. Home-page Notebook list in *src/docs/src/pages/index.astro* — plain-weight *001 · &lt;link to title&gt; — Thursday, April 16 2026*, pointing at */notebook/&lt;slug&gt;/* (e.g. */notebook/nb001/*)

The word "Entry" is never written — the zero-padded number carries the meaning on its own.

Filenames and URL slugs are *nbNNN* (e.g. *nb001.mdx*). The descriptive name lives in the frontmatter *title*, not the slug. The canonical ISO date lives in frontmatter (*date: YYYY-MM-DD*) alongside *entry: &lt;N&gt;*. Day-name only appears in human-visible text. Compute on macOS with *date -j -f "%Y-%m-%d" &lt;YYYY-MM-DD&gt; "+%A"*; Linux: *date -d &lt;YYYY-MM-DD&gt; +%A*.

### 3. Figure aspect ratio is 16:9

All figures and plots in *src/docs/* and *src/artifacts/* use 16:9 (width:height ≈ 1.778:1). Use matplotlib *figsize=(8.0, 4.5)*, *(9.0, 5.0625)*, or *(10.0, 5.625)*. Multi-panel figures should respect 16:9 for the overall figure, even if each panel is squarer. GIFs and MP4s recorded by the harness should also be 16:9.

### 4. Figures are namespaced by notebook entry

Every frozen figure under *src/docs/public/figures/* belongs to exactly one notebook entry, under *src/docs/public/figures/notebook/&lt;entry-slug&gt;/*, where *&lt;entry-slug&gt;* matches the entry's markdown filename. There is no separate paper-level figures tier. If a later entry wants to reference an earlier entry's figure, it does so by URL — figures are not copied.

### 5. No inline backtick code in docs

In *src/docs/* markdown, do not use backtick inline-code formatting.

- **Identifier names** (models, files, functions) — plain text or italics
- **Equations** previously written as inline code — convert to math markup ($...$)
- **Flag/parameter names** like *reset_delay=True* or *--w-in 0.15* — write plain

Fenced code blocks for actual multi-line samples can stay. Doesn't apply to *README.md* files outside *src/docs/*, memory files, or commit messages.

### 6. Images need captions

Every image in docs — notebook entry or background page — carries a caption on the line directly beneath it, in italics:

```markdown
![short alt text](/figures/notebook/<entry-slug>/example.png)
*Caption interpreting the figure. Self-contained — a reader scrolling to the image should understand what it shows without reading the surrounding prose.*
```

The alt text describes the image for accessibility. The caption interprets it for the reader. An image without a caption beneath it is a bug — except in entries with *structure: paper* in frontmatter, where surrounding prose stands in for the italic caption.

### 7. Bug findings belong in PRs, not notebook entries

If a "finding" in a notebook entry turns out to be a bug about to be fixed, remove it from the notebook. The evidence, root cause, and reproducer belong in the PR that fixes it. The notebook records findings about how the system behaves; PRs record what changed and why.

When a notebook entry's narrative gets invalidated by a fix (e.g., an "accuracy gap" that was actually a device-specific bug), the entry needs rewriting against the post-fix baseline — not a new subsection documenting the debugging trail. Durable profilers added mid-investigation get removed with the notebook subsection unless they have standalone value beyond the fixed bug.

### 8. Run sizing tiers

Notebook entry runs pick from a fixed set of tiers. Tiers are size-named so the expected wall-clock cost is legible from the config alone. Notebooks come in two flavours — **training** (gradient descent over samples × epochs) and **video-render** (forward-pass sim per frame + matplotlib render). Both use the same tier names so an entry labeled *small* is an iteration-speed run regardless of flavour, but the underlying budgets differ.

**Training tiers.** ETA per model on MPS at dt 0.25 ms, post the MPS fast-path fix:

| tier | max-samples | epochs | t-ms | steps | $\sim$ETA/model |
| ---- | ----------: | -----: | ---: | ----: | ---------: |
| extra small |   200 |  3 | 600 | 2400 |  $\sim$15 s |
| small       |   500 |  5 | 600 | 2400 |  $\sim$50 s |
| medium      |  2000 | 10 | 600 | 2400 |  $\sim$7 min |
| large       |  5000 | 40 | 200 |  800 | $\sim$20 min |
| extra large | 10000 | 40 | 600 | 2400 |  $\sim$4 hr |

Training cost scales linearly in *max-samples × epochs × t-ms* ($\approx$3.3 × 10⁻⁵ s per unit per model on MPS). Multi-model comparisons multiply: an N-model head-to-head at *medium* is N × $\sim$7 min. *observe video* with per-epoch frames adds $\lesssim$10% at *medium* and above.

**Video-render tiers.** ETA per 3-scan nb002-style run (one 600 ms PING forward pass + full SCOPE_FRAME render per frame). Per-frame wall-clock is $\sim$1.3 s, dominated by matplotlib:

| tier | frames/scan | $\sim$ETA |
| ---- | ----------: | --------: |
| extra small |   5 | $\sim$20 s |
| small       |  15 | $\sim$1 min |
| medium      | 100 | $\sim$6 min |
| large       | 300 | $\sim$20 min |
| extra large | 600 | $\sim$40 min |

Each notebook entry names the tier it ran at — either in Method prose or implicitly via its notebook runner's *MAX_SAMPLES / EPOCHS / T_MS* constants (training) or *TIER_FRAMES* lookup (video). If a finding needs a tier larger than what produced the numbers currently on the page, say so in Next steps rather than quietly upgrading.

## Maintaining this page

When a convention changes, update this page and propagate to the project's persistent memory store. Conventions are enforced by reading and care, not by automation.
