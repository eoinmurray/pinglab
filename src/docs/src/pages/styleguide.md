---
layout: ../layouts/MarkdownLayout.astro
title: "Style guide"
---

# Style guide

Hard rules for editing the docs and notebook. For repo layout, workflow narrative, and how to run things, see [Introduction](/introduction/).

These rules apply when editing the docs and notebook. They are the source of truth — if anything on the [Introduction](/introduction/) page conflicts, these win.

### 1. Notebook-entry H2 skeleton

Every notebook entry under *src/docs/src/pages/notebook/* uses the same five H2 headings, in this order:

1. **Introduction** — what was built, wired, or changed, and why
2. **Method** — how the experiment was run, including methodological notes
3. **Findings** — results (use H3 subsections for multiple results); mechanism analysis nests here, not as its own H2
4. **Implications** — what it means for the paper or project
5. **Next steps** — what is still to do

The H1 stays entry-specific (the entry title). The skeleton is for H2s only. *Why:* notebook-entry navigation should be predictable across the site.

Every entry ends with a single *Appendix* H2 after *Next steps*, holding at minimum the two required H3 subsections below (plus an optional *Videos*):

1. **Reproduction** — the exact *uv run src/pinglab/notebook/nbNNN.py* command that regenerates every artifact in the entry
2. **Run parameters** — a table of the config used (dataset, samples / epochs, *dt*, hidden sizes, batch / lr, seed, tier, elapsed, run ID, git SHA). Use values pulled from *numbers.json* via the *cfg* / *numbers* exports rather than hard-coding
3. **Videos** *(optional)* — per-epoch or per-sweep MP4s that would be too noisy in the body. Use when the entry produces videos that are supporting material rather than headline findings

*Why:* every entry is a reproducible artefact, and the appendix is where the reader goes when they want to re-run the experiment or look up the numerical config — keeping this consistent across entries means that contract is visible at a predictable URL fragment (*#appendix*). Paper-style drafts with custom sectioning can opt out of the main-body skeleton by setting *structure: paper* in frontmatter, but the Appendix still applies.

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

In *src/docs/* markdown, do not use backtick inline-code formatting. Replace each case with the right alternative:

- **Identifier names** (models, files, functions) — plain text or italics. `oscilloscope.py` → *oscilloscope.py*.
- **Flag and parameter names** — italics, no backticks. `--w-in 0.15` → *--w-in 0.15*; `reset_delay=True` → *reset_delay=True*.
- **Equations written as inline code** — convert to math markup. `beta = exp(-dt/tau)` → $\beta = e^{-\Delta t / \tau}$.
- **Numeric values with units** — plain prose. `1.5 ms` stays as 1.5 ms.
- **Approximation tildes** — never write a bare `~` in front of a number in prose. The serif body font renders it as a raised spacing-tilde, not a math "approximately." Use `$\sim$10×` (math-mode tilde, what the rest of this repo uses) or spell it as *about 10×* / "roughly 10×". Same rule for `~±0.04` → `$\sim$±0.04`.

Fenced code blocks for actual multi-line samples (CLI invocations, config snippets) can stay. Doesn't apply to *README.md* files outside *src/docs/*, memory files, or commit messages.

### 6. Figures use the Figure component

Every image and video in docs — notebook entry or background page — is rendered with the shared `<Figure>` component in *src/docs/src/components/Figure.astro*, not raw markdown `![]()`. A figure carries an *id* (displayed label, e.g. *Figure 1*), *title* (short heading), *alt* (accessibility text for images), and a slotted caption body.

```mdx
import Figure from "../../components/Figure.astro";

<Figure id="Figure 1" title="Accuracy vs eval-dt" src="/figures/notebook/nb003/dt_sweep.png" alt="Accuracy vs eval-dt, one panel per training-dt regime">
  Caption interpreting the figure. Self-contained — a reader scrolling to the image should understand what it shows without reading the surrounding prose.
</Figure>
```

For MP4s, pass *type="video"*. The component handles borders, spacing, and caption typography uniformly. A raw `<img>` or markdown image without a surrounding `<Figure>` is a bug.

### 7. Code pointers and reproduction are pinned to commit SHA

Notebook entries describe a point-in-time run, so any link from an entry to code should resolve to the code *as it was at that run*, not whatever is on *main* today. Three shared pieces make this work:

- *src/docs/src/config/repo.ts* — central repo URL plus helpers *blobUrl*, *commitUrl*, *treeUrl*, *shortSha*, *isDirty*. All URL building flows through here; nothing else hard-codes *github.com/eoinmurray/pinglab*. The helpers strip a *(dirty)* suffix before building a link, so a dirty SHA still resolves to its parent commit.
- `<ReproBadge>` (*src/docs/src/components/ReproBadge.astro*) — renders the per-entry metadata strip (run id, commit link, finished timestamp). Every notebook entry places one at the top of the page, fed from the entry's *numbers.json*. The *(dirty)* flag renders visibly so a reader can tell when figures came from uncommitted code.
- `<Code>` (*src/docs/src/components/Code.astro*) — reusable code permalink; takes *path*, optional *line* or *range*, optional *sha*. Inline code references in notebook prose should use this rather than raw links, and should pass *sha={gitSha}* (the entry's *git_sha* export) so the pointer pins to the run.

Every notebook entry must export *gitSha* from its *numbers.json* and pass it into the *ReproBadge*. Background pages (models, metrics, the oscilloscope) don't have a run SHA, so they use `<Code>` without *sha* and the link defaults to *main*.

The notebook driver scripts (*src/pinglab/notebook/<slug>.py*) are responsible for lifting *git_sha* from the training *config.json* into the top-level of *numbers.json* — the *ReproBadge* reads it from there. Every layout also gets an "Edit this page on GitHub" footer for free via *MarkdownLayout.astro*, derived from the source file path; no per-page work.

### 8. Bug findings belong in PRs, not notebook entries

If a "finding" in a notebook entry turns out to be a bug about to be fixed, remove it from the notebook. The evidence, root cause, and reproducer belong in the PR that fixes it. The notebook records findings about how the system behaves; PRs record what changed and why.

When a notebook entry's narrative gets invalidated by a fix (e.g., an "accuracy gap" that was actually a device-specific bug), the entry needs rewriting against the post-fix baseline — not a new subsection documenting the debugging trail. Durable profilers added mid-investigation get removed with the notebook subsection unless they have standalone value beyond the fixed bug.

### 9. Run sizing tiers

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
