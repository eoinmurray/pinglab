# pinglab agent governance

Pinglab studies conductance-based spiking E/I networks, especially PING
(pyramidal-interneuron gamma) dynamics, trained with surrogate gradients and
diagnosed for timestep stability.

This file contains durable project invariants and authorization boundaries.
Procedures belong in the repository skills under `.agents/skills/`.

## Authority

This file and other version-controlled Pinglab policies govern work in this
repository. Mutation authority is inherited from the global `$scope`, `$go`,
`$repo`, and `$help` lexicon. A matching global command is sufficient throughout
this repository; project commands are optional specialist workflows, not
additional permission gates. Project commands may also independently authorize
only the exact actions declared in the lexicon below.

Project skills are opt-in command handlers, not automatic task routers. Use a
skill under `.agents/skills/` only when the user explicitly invokes one of its
documented `$` commands. Semantic similarity, an ordinary natural-language
request, or automatic skill selection is insufficient, even for read-only work.
Handle those requests normally without loading the project skill.

A project command authorizes no Git mutation, GitHub write, deployment, or
action outside this repository. Never infer a broader authorization from a
narrower command.

## Pinglab lexicon

| Command | Authorization |
| --- | --- |
| `$abstract [short\|medium\|long]` | Summarize Pinglab's scientific aims and trajectory in 2, 4, or 6 paragraphs; `medium` is the default. Read-only. |
| `$lab help` | Explain the project lexicon; read-only. |
| `$lab status` | Inspect repository, experiment, campaign, and publication state; read-only. |
| `$experiment draft` | Create or revise only `writings/expNNN.typ` and hand-authored design SVGs under `artifacts/data/expNNN/`. |
| `$experiment review ID` | Review one experiment's design, implementation, evidence, and interpretation; read-only. |
| `$experiment run ID` | Run one existing experiment locally; it may write only beneath `runs/` and the runner's matching `artifacts/data/ID/`. |
| `$campaign plan` | Develop a campaign plan conversationally; read-only. |
| `$campaign status ID` | Inspect one campaign and its jobs, provenance, and completeness; read-only. |
| `$campaign run ID` | Execute an approved campaign, writing only its declared run root and job state. Paid or pod-creating targets require explicit permission naming the target in the same request. |
| `$campaign resume ID` | Resume only approved incomplete stages within the same declared run root, under the same compute gate as `run`. |
| `$campaign review ID` | Review campaign completeness, provenance, and scientific validity; read-only. |
| `$campaign promote ID` | Promote reviewed compact outputs into matching `artifacts/data/` entries in the campaign's existing publication worktree; do not create the worktree, commit, or publish. |
| `$writing draft` | Create or revise only `writings/*.typ` and their referenced hand-authored assets under matching `artifacts/data/` entries. |
| `$writing review ID` | Review one writing and its figures; read-only. |
| `$writing build ID` | Build one existing entry, updating only `artifacts/pdfs/ID.pdf` and its ignored `artifacts/site/` outputs. |
| `$publish check` | Inspect collection-wide publication readiness; read-only. |
| `$publish build` | Build the complete local publication view, updating only `artifacts/pdfs/` and ignored `artifacts/site/`; do not commit, push, or deploy. |
| `$lab-doctor` | Audit Pinglab governance, structure, provenance, and skill integrity; read-only. |

Arguments are mandatory where shown. Optional arguments appear in brackets;
`$abstract` defaults to `medium`. Other bare nouns explain their subcommands and
do not mutate state. Project commands never stage, commit, switch branches,
create worktrees, push, open or update GitHub objects, merge, or deploy. Use the
appropriate global Lexicon command for those actions.

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
- Editing `tools/snn` is authorized by the applicable global `$go` command; no
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
