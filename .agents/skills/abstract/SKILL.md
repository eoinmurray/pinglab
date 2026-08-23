---
name: abstract
description: Use only when the user explicitly invokes abstract or $abstract to summarize Pinglab's scientific aims and trajectory.
---

# Abstract

## Signature

| Operator | Input artifact | Output artifact |
| --- | --- | --- |
| `abstract [short|medium|long]` | `ScientificRecord` | `ScientificAbstract` |

Artifact definitions: [../../ARTIFACTS.md](../../ARTIFACTS.md).

Command grammar: `abstract [short|medium|long]`. The project-wide optional `$`
alias and exact-invocation rule apply.

Accept one optional length argument:

- `abstract short` — exactly 2 paragraphs.
- `abstract medium` or bare `abstract` — exactly 4 paragraphs.
- `abstract long` — exactly 6 paragraphs.

For any other argument, state the three valid lengths and that `medium` is the
default. Do not produce a partial abstract.

## Evidence

Follow the Scientific record invariants in `AGENTS.md`. Summarize what Pinglab
has been trying to understand and build, not merely what files exist. Inspect
only enough current repository evidence to support the account, prioritizing:

1. the project purpose and scientific-record rules in `AGENTS.md` and `README.md`;
2. collection aims in `demolab.yaml`;
3. the manuscript, collection introductions, and relevant current writing
   metadata under `writings/`;
4. the compact published artifacts or run evidence behind any reported result;
5. recent Git history when needed to identify the active direction.

Do not use conversation history as evidence for the project's scientific aims,
or imply that every collection or recent commit has equal scientific
importance.

## Output

Write a self-contained abstract in connected prose for a scientifically literate
reader. Each paragraph must advance the account rather than pad the requested
length. Cover the central question, the approach, the strongest supported
findings, and the present research direction in proportions appropriate to the
selected length.

The abstract body must contain exactly the requested number of non-empty prose
paragraphs, separated by blank lines. Do not add a title, headings, bullets,
numbered lists, a source inventory, process narration, or follow-up question to
that body. Required higher-level response footers sit outside the abstract and do
not count toward its paragraph total. Define unfamiliar abbreviations on first
use. Avoid exact numbers unless they materially improve the summary and satisfy
the evidence rule.
