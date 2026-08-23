---
name: publish
description: Use only when the user explicitly invokes publish or $publish to request Pinglab publication readiness or a local bundle.
---

# Publish

## Signature

| Verb | Input noun | Output noun |
| --- | --- | --- |
| `publish readiness` | `ScientificCollectionState` | `PublishReadiness` |
| `publish bundle` | `PublicationReadyCollection` | `PublishBundle` |

Noun definitions: [../../NOUNS.md](../../NOUNS.md).

Command grammar: `publish readiness|bundle`. The project-wide optional `$` alias and
exact-invocation rule apply.

Use exactly one subcommand:

- `publish readiness` — inspect collection registration, writing metadata,
  referenced artifacts, provenance, generated-output drift, and publication
  readiness.
- `publish bundle` — select the complete supported Demolab build workflow. Run it
  only when the same request supplies applicable global mutation authority;
  otherwise explain that `go` is required and do not build.

Bare `publish` explains the subcommands. Do not run experiments or repair
missing evidence during a publication operation. Report the missing upstream
work instead.
