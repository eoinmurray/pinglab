---
layout: ../layouts/MarkdownLayout.astro
title: "LLM Context"
---

# LLM Context

Pinglab is a research project on spiking neural networks and PING dynamics. This page orients a new agent or collaborator to the repo layout, conventions, and how to run things.

## How this project is run

Pinglab treats the human–AI collaboration driving it as an experimental method, not as scaffolding. The position is between *AI runs the experiments* (aspirational) and *AI helps you write* (surface polish): the AI participates across code, analysis, framing, and writing, while a six-layer architecture with explicit promotion gates keeps contributions reviewable.

## Repo layout

The project is organised in six layers. Each layer has one job; they are connected by manual promotion gates.

- **Code** — *src/pinglab/*. Models, training/inference CLI (*oscilloscope.py*), metrics, plotting, journal repro scripts. Pure Python under [uv](https://docs.astral.sh/uv/).
- **Run outputs** — *src/artifacts/*. Raw figures, videos, logs produced by *oscilloscope.py* and journal repro scripts. Gitignored.
- **Frozen figures** — *src/docs/public/figures/journal/&lt;entry-slug&gt;/*. Published figures with sidecar JSONs carrying the git SHA and run config that produced them. One directory per journal entry.
- **Documentation** — *src/docs/src/pages/*. This Astro site. The substantive content lives under *journal/* — dated entries, newest first. Repro scripts for entries live at *src/pinglab/journal/&lt;entry-slug&gt;.py*.
- **Reference literature** — *src/papers/*. Bibliography for the project (PDFs themselves are not redistributed; see *src/papers/README.md* for citations).
- **Collaboration meta** — *CLAUDE.md* at the repo root, this LLM Context page, and a persistent memory directory under *~/.claude/projects/-Users-eoin-pinglab/memory/*. The agreed-upon rules for how the human and AI work on this project.

Code writes only to *src/artifacts/*. It never touches *src/docs/*. Figures enter the docs site via an explicit freeze step.

## Journal

All writeups are journal entries at *src/docs/src/pages/journal/&lt;slug&gt;.md*, listed chronologically on the home page at *src/docs/src/pages/index.astro*. There is no separate per-experiment page or paper layer — when a paper-shaped writeup is appropriate, it is a journal entry like any other.

**Slug.** Descriptive, no date prefix — e.g. *snntorch-calibration*, not *2026-04-17-1100-snntorch-parity-and-calibration*. The slug is the filename, the URL, and the key for the parallel repro and figures directories. The canonical date lives in the entry's frontmatter (*date: YYYY-MM-DD*) and in its visible long-form byline; the filename stays date-free so the slug can read as prose. Chronological ordering in the home page's Journal list is human-maintained.

Most entries follow a fixed structure: Introduction, Method, Findings, Implications, Next steps. Longer paper-style drafts keep their own section numbering.

Each entry that cites generated figures or numbers gets a repro script at *src/pinglab/journal/&lt;slug&gt;.py* — one command regenerates every figure and dumps a *numbers.json* next to them under *src/docs/public/figures/journal/&lt;slug&gt;/*.

## Journal entry review

Before a finding in a journal entry is considered load-bearing, run the entry past an adversarial reviewer in a fresh Claude session — no shared context with the session that wrote it. The reviewer reads the entry cold and challenges the framing. Its output is a critique to incorporate, not a veto: add the reviewer's findings (and how each was addressed or argued down) as a *Reviewer-agent critique* H3 subsection at the end of the entry's *Findings*.

The prompt to paste into the fresh session, followed by the entry text:

```text
You are an adversarial reviewer for a research journal entry. The entry below is a draft finding that may be promoted into a longer-form paper. Your job is to read the entry cold — assume nothing about the surrounding project, the author's intentions, or unstated context — and challenge the framing.

Look for:

1. Single-cause framings. Does the entry attribute an effect to one mechanism when multiple plausible mechanisms could explain it? Name each alternative.
2. Unstated assumptions. Are there premises being treated as obvious that a sceptical reader would question? List them.
3. Sample-size or statistical-power issues. If numbers are reported, are they robust to seed noise, repetition, or sample-size scaling? Would the conclusion survive a reasonable replication?
4. Methodology gaps. Is the experiment described in enough detail that a reader could reproduce it? Is the comparison fair? Are confounds controlled?
5. Overreach. Does the entry's headline claim exceed what the data shows? Is the claim worded more strongly than the evidence supports?
6. Missing alternatives. Are there obvious alternative experiments that would either falsify or strengthen the claim, and are they acknowledged in the next-steps?
7. Internal consistency. Do the Findings, Implications, and Next steps cohere? Does the Implications section match the strength of the Findings?

For each issue you raise, give:

- A one-line statement of the issue.
- The specific text in the entry it applies to (quote it).
- A concrete suggestion for what would address it (an experiment, a rewording, a caveat, an alternative explanation).

Be concrete and falsifiable. Avoid generic critiques ("could be more rigorous", "needs more data"). If the entry passes a particular check cleanly, say so explicitly — don't manufacture concerns.

End with a one-paragraph summary judgement: is this finding ready to promote into the paper as-is, ready with caveats (list them), or not ready (state why).

The entry follows.
```

## Figures

Figures produced by *oscilloscope.py* and journal repro scripts land under *src/artifacts/*. When one is ready to publish, freeze it:

```sh
uv run python src/scripts/freeze-figure.py <source> <dest>
```

The script copies the PNG and writes a sidecar JSON with the git SHA at freeze time and the run config (run_id, model, dt, samples, epochs). Every frozen figure belongs to exactly one journal entry:

- Destination — *src/docs/public/figures/journal/&lt;entry-slug&gt;/&lt;name&gt;.png*

Reference from markdown via the web path, e.g. */figures/journal/&lt;entry-slug&gt;/&lt;figure-name&gt;.png*.

Commit the PNG, the sidecar JSON, and the markdown edit together. Figures use 16:9 aspect ratio.

## Running

Install [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh/).

**Train or inspect a model:**

```sh
uv run python src/pinglab/oscilloscope.py --help
uv run python src/pinglab/oscilloscope.py train --model snntorch --dataset mnist --max-samples 1000 --epochs 3
```

**Reproduce a journal entry:**

```sh
uv run python src/pinglab/journal/<entry-slug>.py
```

Each script is argument-free and regenerates every figure and number its entry cites.

**Run the docs site:**

```sh
cd src/docs
bun install
bun dev
```

The site is served at *localhost:4321*.

**Run the tests:**

```sh
uv run pytest
```

Tests live in *src/pinglab/tests/unit/*. Markers *slow* and *regression* gate the slower subset.

## Glossary

Project-specific terms. Definitions here are load-bearing — if something elsewhere contradicts a definition, this page wins.

- **Oscilloscope** — the training / inference / inspection CLI at *src/pinglab/oscilloscope.py*. Training and evaluation runs go through its *train* and *infer* subcommands; all other invocation patterns drive it (journal repro scripts shell out to it via *sh*).
- **Ladder** — the feature-incremental set of models stepping from a vanilla SNN to full PING: *snntorch → cuba → cuba-exp → coba → ping*. Each rung adds one biophysical feature.
- **Promotion gate** — a manual step that moves content between layers: run output → frozen figure, journal entry → paper section, ad-hoc preference → persistent memory. Code does not cross these gates.
- **Calibration** — tuning hyperparameters (weight scales, thresholds, input drive) so models on the ladder are comparable before an experiment runs.
- **Frozen figure** — a published PNG under *src/docs/public/figures/…* copied from *src/artifacts/…* with a sidecar JSON carrying the git SHA at freeze time and the run config.
- **Trainable surface** — what optimisation actually updates. Across the ladder it is input + output weights; recurrent weights in *ping* are frozen at init, not trained.
- **Δt-stability** — the diagnostic of training at one integration timestep and evaluating at another. The shared lens across pinglab experiments.
- **PING** — Pyramidal-Interneuron Gamma. The E→I→E feedback loop that produces gamma oscillations (30–80 Hz).
- **CUBA / COBA** — current-based vs conductance-based synapses. The axis the Δt-stability experiment decomposes.

## Invariants and gotchas

Facts about this repo that are load-bearing and easy to get wrong.

- **Δt is in milliseconds.** Everywhere. Config fields carry the *_ms* suffix (*sim_ms*, *burn_in_ms*, *step_on_ms*); CLI flags follow the same convention (*--t-ms*). A "Δt of 1" means 1 ms, not 1 s.
- **Trial length defaults to 600 ms** (*--t-ms 600*). Shorter trials emit a warning — the harness will let you run them but flags that the transient window is too short.
- **Code writes only to src/artifacts/.** *oscilloscope.py* and journal repro scripts never touch *src/docs/*. Figures enter the docs site exclusively through *src/scripts/freeze-figure.py*.
- **src/artifacts/ is gitignored.** Nothing under it is part of the repo; it is reproducible from code + config + seed.
- **PING recurrent weights are frozen.** Set analytically at init, not trained. Only input and output weights are trainable across the ladder. This is a deliberate control — it isolates the effect of the E–I architecture from learning of internal weights.
- **Poisson input encoding is frozen across Δt-sweeps.** Each image is encoded once at the finest sweep Δt, then OR-pooled to the target Δt. This eliminates Poisson resampling as a confound.
- **uv for Python, bun for JavaScript.** No *pip install*, no *npm install*. Lock files are *uv.lock* and *bun.lock*.

## Conventions

These rules apply when editing the docs and journal. They are the source of truth — if anything above them conflicts, these win.

### 1. Journal-entry H2 skeleton

Every journal entry under *src/docs/src/pages/journal/* uses the same five H2 headings, in this order:

1. **Introduction** — what was built, wired, or changed, and why
2. **Method** — how the experiment was run, including methodological notes
3. **Findings** — results (use H3 subsections for multiple results); mechanism analysis nests here, not as its own H2
4. **Implications** — what it means for the paper or project
5. **Next steps** — what is still to do

The H1 stays entry-specific (the entry title). The skeleton is for H2s only. *Why:* journal-entry navigation should be predictable across the site.

### 2. Date format includes day of week

Visible journal dates start with the full day-of-week name. Format: *DayName, Month D YYYY*.

Three places per entry use this:

1. Frontmatter title — *title: "Thursday, April 16 2026 — &lt;slug&gt;"*
2. Italic byline under H1 — *Thursday, April 16 2026*
3. Home-page Journal list in *src/docs/src/pages/index.astro* — bold date prefix *Thursday, April 16 2026* followed by *— &lt;title&gt;*, pointing at */journal/&lt;entry-slug&gt;/*

Filenames and URL slugs are date-free (e.g. *snntorch-calibration.md*); the canonical date lives in frontmatter (*date: YYYY-MM-DD*). Day-name only appears in human-visible text. Compute on macOS with *date -j -f "%Y-%m-%d" &lt;YYYY-MM-DD&gt; "+%A"*; Linux: *date -d &lt;YYYY-MM-DD&gt; +%A*.

### 3. Figure aspect ratio is 16:9

All figures and plots in *src/docs/* and *src/artifacts/* use 16:9 (width:height ≈ 1.778:1). Use matplotlib *figsize=(8.0, 4.5)*, *(9.0, 5.0625)*, or *(10.0, 5.625)*. Multi-panel figures should respect 16:9 for the overall figure, even if each panel is squarer. GIFs and MP4s recorded by the harness should also be 16:9.

### 4. Figures are namespaced by journal entry

Every frozen figure under *src/docs/public/figures/* belongs to exactly one journal entry, under *src/docs/public/figures/journal/&lt;entry-slug&gt;/*, where *&lt;entry-slug&gt;* matches the entry's markdown filename. There is no separate paper-level figures tier. If a later entry wants to reference an earlier entry's figure, it does so by URL — figures are not copied.

### 5. No inline backtick code in docs

In *src/docs/* markdown, do not use backtick inline-code formatting.

- **Identifier names** (models, files, functions) — plain text or italics
- **Equations** previously written as inline code — convert to math markup ($...$)
- **Flag/parameter names** like *reset_delay=True* or *--w-in 0.15* — write plain

Fenced code blocks for actual multi-line samples can stay. Doesn't apply to *README.md* files outside *src/docs/*, memory files, or commit messages.

### 6. Images need captions

Every image in docs — journal entry or background page — carries a caption on the line directly beneath it, in italics:

```markdown
![short alt text](/figures/journal/<entry-slug>/example.png)
*Caption interpreting the figure. Self-contained — a reader scrolling to the image should understand what it shows without reading the surrounding prose.*
```

The alt text describes the image for accessibility. The caption interprets it for the reader. An image without a caption beneath it is a bug.

### 7. Bug findings belong in PRs, not journals

If a "finding" in a journal entry turns out to be a bug about to be fixed, remove it from the journal. The evidence, root cause, and reproducer belong in the PR that fixes it. The journal records findings about how the system behaves; PRs record what changed and why.

When a journal entry's narrative gets invalidated by a fix (e.g., an "accuracy gap" that was actually a device-specific bug), the entry needs rewriting against the post-fix baseline — not a new subsection documenting the debugging trail. Durable profilers added mid-investigation get removed with the journal subsection unless they have standalone value beyond the fixed bug.

## Maintaining this page

When a convention changes, update this page and propagate to persistent memory under *~/.claude/projects/-Users-eoin-pinglab/memory/*. Conventions are enforced by reading and care, not by automation.
