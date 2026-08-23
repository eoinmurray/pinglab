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

A project command authorizes no Git mutation, GitHub write, deployment, or
action outside this repository. Never infer a broader authorization from a
narrower command.

## Pinglab lexicon

The Lexicon contains operators, conventionally expressed as verbs. Each
operator consumes and produces primitive artifacts. These types are
prose-defined and text-serialized as Markdown rather than formally
schema-validated. Their contracts live in `.agents/ARTIFACTS.md`, and every
project skill declares its input/output signature.

| Command | Input artifact | Output artifact |
| --- | --- | --- |
| `abstract [short\|medium\|long]` | `ScientificRecord` | `ScientificAbstract` |
| `hypo beam` or `hypo beamX` | `Seed`, `Formulation`, or `ResumableCheckpoint` | `BranchSet` |
| `hypo compare` | `Formulation` | `CanonComparisonCapsules` |
| `hypo ground web` | `Formulation` | `LiteratureEvidenceCapsules` |
| `hypo ground local` | `Formulation` | `RepositoryEvidenceCapsules` |
| `hypo checkpoint` | `OpenSearchTrajectory` | `ResumableCheckpoint` |
| `hypo freeze` | `GroundedSearchTrajectory` or `ResumableCheckpoint` | `FrozenHypothesisPacket` |
| `pinglab help` | `PinglabLexiconContext` | `PinglabLexiconReference` |
| `experiment draft` | `FrozenHypothesisPacket` | `UnrunExperimentSpecification` |
| `publish check` | `ScientificCollectionState` | `PublicationReadinessReport` |
| `publish build` | `PublicationReadyCollection` | `PublicationBundle` |

Arguments are mandatory where shown. Optional arguments appear in brackets;
`abstract` defaults to `medium`, and `hypo beam` defaults to three branches.
Every documented project command accepts an optional leading `$`; for example,
`hypo ground web` and `$hypo ground web` are equivalent. Bare command-family
nouns explain their subcommands. Use the appropriate global Lexicon command to
authorize any mutation required by a selected workflow.

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
- Editing `tools/snn` is authorized by the applicable global `go` command; no
  project command is additionally required.
- GitHub writes require the applicable global Lexicon authorization.
- Never add AI authorship or attribution trailers to commits or PR text.

## Verification

- Writing-only changes: build each affected Demolab entry. Run the complete
  publication build only when shared rendering, collection structure, or the
  book can change. Do not run the software suite.
- Code, data-contract, runner, or executable-example changes: run focused tests
  first, then the proportionate broader checks defined by the affected tool.
- Inspect generated artifacts and provenance before accepting them. A large
  generated diff is evidence to review, never permission for blind staging.

## Campaign and publication isolation

Campaign execution, activation, artifact promotion, and publication-view
rebuilding must happen on the campaign's own branch in a dedicated worktree.
Use one campaign or publication view per worktree and branch. Open its draft PR
before substantial execution or promotion work begins; the PR description is
the scientific decision record for motivation, provenance, competing
interpretations, decisions, and unresolved limitations.

Do not activate a campaign in a general development worktree. Review `git
status` and the artifact diff before every campaign commit, and never use
`git add -A` as a substitute for that review.
