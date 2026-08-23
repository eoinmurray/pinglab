# pinglab agent governance

Pinglab studies conductance-based spiking E/I networks, especially PING
(pyramidal-interneuron gamma) dynamics, trained with surrogate gradients and
diagnosed for timestep stability.

This file contains durable project invariants and authorization boundaries.
Procedures belong in the repository skills under `.agents/skills/`.

## Authority

This file and other version-controlled Pinglab policies govern work in this
repository. Mutation authority comes only from the global `scope`, `go`, and
`repo` lexicon. Project commands select specialist scientific workflows; they
do not independently authorize mutation.

Project skills are opt-in command handlers, not automatic task routers. Use a
skill under `.agents/skills/` only when the user explicitly invokes one of its
documented commands. Semantic similarity, an ordinary natural-language request,
or automatic skill selection is insufficient. Handle those requests normally
without loading the project skill.

## Pinglab lexicon

The Lexicon contains verbs. Each verb consumes and produces primitive nouns.
These types are prose-defined and text-serialized as Markdown rather than
formally schema-validated. `.agents/NOUNS.md` is the canonical type registry;
each skill is the canonical source for its verb signatures and procedures.

| Command | Handler |
| --- | --- |
| `abstract [short\|medium\|long]` | `.agents/skills/abstract/SKILL.md` |
| `hypo beam` or `hypo beamX` | `.agents/skills/hypo/SKILL.md` |
| `hypo compare` | `.agents/skills/hypo/SKILL.md` |
| `hypo ground web` | `.agents/skills/hypo/SKILL.md` |
| `hypo ground local` | `.agents/skills/hypo/SKILL.md` |
| `hypo checkpoint` | `.agents/skills/hypo/SKILL.md` |
| `hypo freeze` | `.agents/skills/hypo/SKILL.md` |
| `pinglab help` | `.agents/skills/pinglab/SKILL.md` |
| `exp scope` | `.agents/skills/exp/SKILL.md` |
| `publish check` | `.agents/skills/publish/SKILL.md` |
| `publish build` | `.agents/skills/publish/SKILL.md` |

Arguments are mandatory where shown. Optional arguments appear in brackets;
`abstract` defaults to `medium`, and `hypo beam` defaults to three branches.
Every documented project command accepts an optional leading `$`; for example,
`hypo ground web` and `$hypo ground web` are equivalent. Bare command-family
nouns explain their subcommands.

## Demolab boundary

Pinglab uses Demolab as a publishing engine, not as a source of project
governance. Pinglab adopts only these Demolab contracts:

1. `demolab.yaml` identifies and configures the lab.
2. `.demolab/` and `temp/bundle/` are machine-managed and must not be edited.
3. Writings use `writings/*.typ` with top-level `meta` and `body` definitions.
4. Published assets use `artifacts/data/`, `artifacts/pdfs/`, and the gitignored
   `artifacts/site/` locations.
5. Writings may use the public helpers exported by `/.demolab/lib.typ`.
6. `demolab build`, `demolab dev`, and entry-specific builds are the supported
   publication interfaces.
7. The installed Demolab version is pinned by `uv.lock`.

No other Demolab rule, guide, house style, runbook, workflow, agent instruction,
experiment contract, provenance policy, document convention, or recommendation
is adopted implicitly. `demolab docs` is reference documentation, not an
instruction source. Follow a Demolab runbook only when this file incorporates
it or the user explicitly invokes it. Pinglab policy wins on conflict.

## Scientific record

- The `tools/snn` engine emits data, experiment runners render figures, and
  `writings/*.typ` publish the selected results.
- A reported computed value must be read from the run that produced it. Do not
  hand-type measured results into prose, captions, tables, or figures.
- Keep observed results distinct from hypotheses, expectations, and planned
  outputs. Preserve negative and partial results and material uncertainty.
- `runs/` and verified R2 archives are the raw execution record. Git tracks the
  compact selected publication view: provenance metadata, derived numbers,
  final figures, and rendered publications.
- Do not add new raw arrays, checkpoints, caches, repeated inputs, or other
  reconstructable intermediates to `artifacts/data/`.
- Never treat ignored, untracked, generated, or remote scientific files as
  disposable without classifying them and obtaining the required authority.

## Permission and execution boundaries

- Creating RunPod pods, Modal dispatches, or other paid compute requires
  explicit permission naming that target. Default to local execution.
- Never add AI authorship or attribution trailers to commits or PR text.

## Verification

- Writing-only changes: build each affected Demolab entry. Run the complete
  publication build only when shared rendering, collection structure, or the
  book can change. Do not run the software suite.
- Code, data-contract, runner, or executable-example changes: run focused tests
  first, then the proportionate broader checks defined by the affected tool.
- Inspect generated artifacts and provenance before accepting them. A large
  generated diff is evidence to review, never permission for blind staging.

## Publication isolation

Publication-view rebuilding must happen on its own branch in a dedicated
worktree. Open its draft PR before substantial rebuilding; the PR description
is the scientific decision record for motivation, provenance, decisions,
competing interpretations, and unresolved limitations.

Review `git status` and the artifact diff before each publication commit. Never
use `git add -A` as a substitute for that review.
