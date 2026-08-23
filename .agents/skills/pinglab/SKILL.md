---
name: pinglab
description: Use only when the user explicitly invokes pinglab help or $pinglab help to explain Pinglab's project command vocabulary.
---

# Pinglab

## Signature

| Verb | Input noun | Output noun |
| --- | --- | --- |
| `pinglab help` | `PinglabLexiconContext` | `PinglabLexiconReference` |

Noun definitions: [../../NOUNS.md](../../NOUNS.md).

Command grammar: `pinglab help`. The project-wide optional `$` alias and
exact-invocation rule apply.

Explain the project command vocabulary from `AGENTS.md` in concise plain
language. Present each verb with its input and output noun types from
`.agents/NOUNS.md`. Separate it from the global Lexicon and explain that
project commands select scientific workflows while global commands govern
mutation. Do not perform another command on the user's behalf.

Bare `pinglab` explains that `help` is its only subcommand.
