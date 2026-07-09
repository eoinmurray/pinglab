# Runbook: Grounding cited claims in their sources

> Build a **claim-verification companion**: for each work cited in a manuscript, point the reader
> to the exact sentence(s) in the source that back the claim. No scripts, no install — an agent
> follows this one file.

## When to use
When a manuscript leans on cited works and you want each citation traceable to the exact sentence
in the source that backs it. The output is a **pointer aid, not an authority**: quotes are
agent-located and self-checked for verbatim match, but the reader confirms against the source
before citing. Every page produced under this runbook must carry that disclaimer (see *Labeling*).

Have two things in place before you start:

1. **The manuscript**, with a numbered reference list and inline citation markers (e.g. `[26]`).
   (In pinglab: the ar009 article + its `ar009-references.ts` list.)
2. **The source papers as files** in one folder — you download them yourself (institutional
   access, open access, wherever). Name each file so the reference is identifiable, or let the
   agent match by title/DOI. Papers you don't provide are flagged as missing; the agent never
   invents their content.

The agent produces one section per reference, in the **Entry format** below (Summary / Claim it
supports / Quotes), assembled into the companion page in reference-number order.

## What it does

0. **Confirm the inputs.** The manuscript (numbered references + inline markers) and the source
   papers as files are provided. Papers not provided are flagged missing — never invented.

1. **Map claims to citations.** Read the manuscript. For every reference number, record which
   paragraph(s) cite it and copy the **verbatim** manuscript sentence carrying that citation. This
   is the "Claim it supports" content and it is pure copying — no interpretation.

2. **For each reference that has a source file:**
   - **Read the source in full.** Not the abstract, not from memory.
   - **Summary** — one paragraph, from the source: what it is and what it argues.
   - **Claim it supports** — paste the verbatim manuscript line(s) from step 1.
   - **Quotes** — find sentence(s) in the source that back that specific claim. Copy each
     **character-for-character**. Prefer 1–3 strong anchors over many weak ones.

3. **Self-verify every quote — the load-bearing step.** For each quote you are about to write,
   re-locate it in the source text and confirm ALL of the following. If any fails, **drop the
   quote** — never reword the source to fit.
   - It is a **single contiguous passage** actually present in the source.
   - It matches **character-for-character**, ignoring ONLY these typographic artifacts: line
     breaks, end-of-line hyphenation (`inhibi-\ntion` → `inhibition`), ligatures (`ﬁ`→`fi`), smart
     vs straight quotes/dashes, and runs of whitespace.
   - It differs in **no word, no number, and no negation**. If you had to add or drop a "not"/"no",
     change a number, or swap a word to make it fit the claim, it FAILS. ("was silent" must never
     stand in for "was not silent".)
   - It is a real clause, not a 2–3 word fragment (roughly 4+ words).

4. **Missing or abstract-only sources.** No source file, or only the abstract → say so explicitly
   in *Quotes*, e.g. *No verbatim anchor — source not provided*, and stop. Never fabricate a quote
   or infer one from the title.

5. **Assemble.** Write each section under its anchor in the Entry format, in reference-number
   order, and add the *Labeling* line to the page header.

---

## Entry format

One section per cited work, always these three fields in this order.

### Skeleton

```
<span id="ref-N"></span>
#### Authors (Year) — Title

[doi.org/DOI](https://doi.org/DOI)

**Summary.** One paragraph on what the paper is and what it argues, from the source.

**Claim it supports (§x.y).** *"Verbatim manuscript sentence carrying this citation, citation markers [N] intact."*

**Quotes.**

- *"Verbatim source quote"* (p. N) — one line: what this quote establishes for the claim.
- *"Another verbatim source quote"* (p. N) — its context line.
```

### Example (filled)

```
<span id="ref-1"></span>
#### Buzsáki & Wang (2012) — Mechanisms of Gamma Oscillations

[doi.org/10.1146/annurev-neuro-062111-150444](https://doi.org/10.1146/annurev-neuro-062111-150444)

**Summary.** A canonical review of the cellular and network mechanisms of gamma oscillations, which it defines as a synchronous network rhythm in the 30–90 Hz band. …

**Claim it supports (§1.1).** *"Gamma oscillations in the 30–80 Hz band have been associated with attention, binding, and gating in cortical activity [1, 2, 3]."*

**Quotes.**

- *"We refer to periodic events in the 30–90-Hz band as gamma oscillations"* (p. 3) — establishes the gamma frequency band the manuscript invokes.
- *"Several studies have examined the relationship between cross-frequency coupling of gamma oscillations and cognitive processes"* (p. 14) — supports the link between gamma and cognition.
```

### Rules

Structure
- Heading is `####`: `Authors (Year) — Title`, preceded by `<span id="ref-N"></span>`.
- Link line: the DOI link. (Site-specific extras — e.g. a local-PDF link — are optional
  presentation and not required by the format.)
- Three bold labels, always in order: **Summary.** / **Claim it supports (§x).** / **Quotes.**

Content
- Summary — descriptive, from the source, under 100 words (~2–4 sentences). No editorial.
- Claim it supports — the verbatim manuscript line(s), italic, citation markers intact.
  One italic quote per citing section if cited in more than one place.
- Quotes — verbatim source anchors (typically 1–3), each *italic quote* + `(p. N)` + a
  one-line context gloss.

Typography
- Quotes in *italics* with quotation marks; labels in **bold**.
- En-dashes for ranges (30–90 Hz); `≈` not `~`; no backticks / inline code.
- Page location as `(p. N)`; for sources with no page number use the section title,
  e.g. `(§ Introduction)`.

Honesty
- Descriptive, not editorial: state the claim, let the verbatim quotes carry it.
- Both sides quoted verbatim (manuscript line and source quotes); prose only in Summary
  and the one-line context glosses.
- A quote that fails self-verify is dropped, never reworded to pass.

---

## Labeling (required)

The companion page must state, near the top:

> Quotes are agent-located pointers to each source, self-checked for verbatim match but
> not independently verified — confirm against the source before citing.

This single line keeps the output honest: a lookup aid with a human verification step,
not a claim of correctness.

---

## Optional: fan out for large reference lists

For many references, run one agent per source paper in parallel — each does steps 2–3
for its paper and returns the finished section. Steps 1 (claim map) and 5 (assemble)
stay central. Keep the self-verify (step 3) inside each agent; it is not optional.

## What this runbook deliberately does NOT do

- **Fetch papers.** You provide them.
- **Guarantee correctness.** Self-verify catches most fabrications and dropped
  negations, but it is an agent check, not a deterministic one; the reader is the final
  verifier.

---

## Agent contract
- **Triggers** — `GROUND-CLAIMS`.
- **Gates** — the manuscript (numbered references + inline citation markers) and the source papers
  as files are provided; the runbook never fetches papers. Missing sources are flagged, never
  invented.
- **Report & apply** — every companion page must carry the *Labeling* disclaimer near the top; the
  output is a pointer aid, not an authority, and the reader is the final verifier. Self-verify
  (step 3) is not optional, including inside any fanned-out agent. A quote that fails self-verify
  is dropped, never reworded to pass.
