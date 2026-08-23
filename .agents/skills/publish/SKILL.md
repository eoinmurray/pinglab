---
name: publish
description: Use only when the user explicitly invokes publish or $publish to check or select a build of Pinglab's complete local publication.
---

# Publish

## Signature

| Operator | Input artifact | Output artifact |
| --- | --- | --- |
| `publish check` | `ScientificCollectionState` | `PublicationReadinessReport` |
| `publish build` | `PublicationReadyCollection` | `PublicationBundle` |

Artifact definitions: [../../ARTIFACTS.md](../../ARTIFACTS.md).

Command grammar: `publish check|build`. The project-wide optional `$` alias and
exact-invocation rule apply.

Use exactly one subcommand:

- `publish check` — inspect collection registration, writing metadata,
  referenced artifacts, provenance, generated-output drift, and publication
  readiness.
- `publish build` — select the complete supported Demolab build workflow. Run it
  only when the same request supplies applicable global mutation authority;
  otherwise explain that `go` is required and do not build.

Bare `publish` explains the subcommands. Do not run experiments or repair
missing evidence during a publication operation. Report the missing upstream
work instead.
