# Repository lexicon audit — 28 August 2026

**Scope:** every current tracked or non-ignored `.typ` source in the repository: **46 files**, comprising **42 articles and four shared helpers**. The deprecated exp048 article is included. This audits the working checkout, including its uncommitted changes, not just HEAD.

**Result:** **234 recurring candidate families**, plus **three supplementary local candidates**, with a proposed standard or explicit boundary for each. The families overlap: they are an inventory for review, not 234 mutually exclusive concepts or 234 confirmed naming defects. See `summary.json` for generated counts and `corpus.json` for per-file line counts and content hashes.

## Read the audit

| Document | Purpose |
| --- | --- |
| [Findings and decisions](findings.md) | The important distinctions, source disagreements and recommended order of work. Start here. |
| [Complete candidate catalogue](catalogue.md) | Every curated term family, proposed definition/naming boundary, file count and occurrence count. |
| [Mathematical notation](notation.md) | Colliding symbols, units, meanings and suggested qualified names. |
| [Repeated symbols and identifiers](identifiers.md) | Every repeated extracted math token, inline-code string and local declaration, with all locations. |
| [Every matched location](evidence.md) | Cross-references from each candidate to every matching file and line. |
| [File-by-file inventory](inventory.md) | All 46 sources, with the candidate entries found in each. |

The highest-value decisions are **readout identity**, **rate versus count**, **checkpoint selection**, **participation versus fitted slope**, **rhythm estimator identity**, **weight orientation/scaling**, and **physical duration versus timestep count**. These recur across experiments and syntheses and can change the meaning of a result, not merely its presentation.

Some disagreements need correction before adopting shared definitions: the articles describe different output models, conflicting damping settings, validation curves called test curves, SD versus SEM on a reused figure, a rate called a count, and contradictory graph-training status. A lexicon must expose these rather than choose one silently.

## Method and completeness boundary

1. Inventoried Git-tracked and non-ignored Typst sources; cross-referenced article prose, equations, figure captions/alternative text, source-field names and the shared helpers.
2. Curated recurring scientific, methodological, software and provenance concepts, including related spellings and aliases. The exact matching expressions are retained in `candidates.txt` and displayed in the evidence index.
3. Matched those expressions across all source text, normalizing whitespace, common dash variants and micro symbols while retaining original offsets. Each occurrence records the original match, start/end lines, exact offsets, section and a context label.
4. Separately indexed all extracted dollar-delimited mathematical expressions, repeated mathematical tokens, repeated inline-code strings, repeated `let` declarations and repeated prose words. These support review for omissions beyond the curated families.
5. Verified source hashes, occurrence spans, counts and corpus coverage. Source links are snapshot-specific; ongoing edits can move the linked lines.

“Exhaustive” here means **all source files and every match to the documented candidate expressions**, supplemented by manual semantic review and mechanical lexical indexes. It does not mean an infallible enumeration of every possible synonym. Context classification and equation/token extraction are lightweight heuristics, not a Typst parser, binding resolver or rendered-document analysis. Some matches belong to code, metadata, comments or bibliography; counts include these, while core cross-file eligibility requires substantive use in at least two files. All matches remain inspectable.

Ignored build/test/import copies under `.demolab`, `.r2` and `.scratch` were excluded as duplicate or derived material; a path-only inventory found 4,953 such `.typ` paths. Dependency trees, retained run payloads, remote assets and compiled output were not read as source evidence. Imported JSON values and figure contents were not authenticated or reanalysed. A passage naming the same reused result twice is two textual occurrences, not two independent observations.

## Rebuild and inspect

From the repository root:

```sh
python3 reviews/lexicon-audit-2026-08-28/build_audit.py
```

This uses only the Python standard library plus Git and ripgrep. It writes generated audit files beside itself and never runs experiments or writes source articles. The authored `findings.md` and `notation.md` are not regenerated: review their links and conclusions after source changes. `candidates.txt` is the editable curation list; its row order defines the L001–L237 identifiers.

| Machine-readable file | Contents |
| --- | --- |
| `corpus.json` | Exact source set, titles, line counts and SHA-256 hashes. |
| `occurrences.json` | All curated families, patterns, recommendations and exact occurrence spans. |
| `lexical-index.json` | Mathematical expressions and repeated words, math tokens, inline identifiers and declarations. |
| `summary.json` | Generated coverage and match counts. |

**No terminology has been adopted yet. No `.typ` files, implementation, storage records, article dates or status badges were edited by this audit.** The next decision is which proposals to accept and which source disagreements to resolve; migration is separate work.
