---
name: pinglab
description: Use only when the user explicitly invokes pinglab help or $pinglab help to explain Pinglab's project command vocabulary.
---

# Pinglab

## Signature

| Operator | Input artifact | Output artifact |
| --- | --- | --- |
| `pinglab help` | `PinglabLexiconContext` | `PinglabLexiconReference` |

Artifact definitions: [../../ARTIFACTS.md](../../ARTIFACTS.md).

Activate this skill only for the exact `pinglab help` command, with or without a
leading `$`. Do not activate it from semantic similarity or an ordinary request
about the project.

Explain the project command vocabulary from `AGENTS.md` in concise plain
language. Present each operator with its input and output artifact types from
`.agents/ARTIFACTS.md`. Separate it from the global Lexicon and explain that
project commands select scientific workflows while global commands govern
mutation. Do not perform another command on the user's behalf.

Bare `pinglab` explains that `help` is its only subcommand.
