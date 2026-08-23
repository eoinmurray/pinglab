---
name: pinglab
description: Use only when the user explicitly invokes pinglab lexicon or $pinglab lexicon to explain Pinglab's scientific noun vocabulary.
---

# Pinglab

## Signature

| Verb | Input noun | Output noun |
| --- | --- | --- |
| `pinglab lexicon` | `PinglabLexiconContext` | `PinglabLexicon` |

Noun definitions: [../../NOUNS.md](../../NOUNS.md).

Command grammar: `pinglab lexicon`. The project-wide optional `$` alias and
exact-invocation rule apply.

Explain the scientific noun vocabulary from `.agents/NOUNS.md` in concise plain
language. Group nouns by family, state their important relationships and
lifecycles, and explain that ordinary conversation constructs and transforms
them. Identify `pinglab lexicon` as a thin explanatory interface. Separate it
from the global Lexicon, whose commands govern mutation. Do not perform another
command on the user's behalf.

Bare `pinglab` explains that `lexicon` is its only subcommand.
