# Agent guide router

Start with [README.md](README.md) for the repository overview. This file is a
router, not a reading list: do not read all linked guides by default. Select
guides from the requested action, not merely the files or concepts mentioned.
Each guide is canonical within its scope, not a prerequisite for unrelated work.

Read the selected live guides before performing the matching work. Inspection
of source code alone does not require a guide unless the inspection itself
matches a trigger below. Add guides when the task actually crosses into another
scope; a hypothetical downstream effect is not a trigger. Follow explicit
cross-references in a loaded guide when they apply to the action at hand.

## Routing table

| Guide | Read when the task involves… | Do not read merely because… |
| --- | --- | --- |
| [Experiment Rules](md/notes/experiment-rules.md) | Designing, creating, changing, or running experiment structures, recipes, compute/analyse/present stages, execution helpers, or scientific input selection; changing experiment discovery, preview, or publication behavior. | You are explaining experiment source code or editing unrelated UI. |
| [Compute Rules](md/notes/compute-rules.mdx) | Planning or launching compute, choosing a provider/backend for execution, accessing remote machines, submitting/monitoring jobs, collecting remote outputs, or terminating resources; changing automation that performs these operations. | Source code mentions a backend or an article contains previously collected results. |
| [Pingstore Rules](md/notes/pingstore-rules.md) | Reading or writing run artifacts; validating, importing, assembling, transferring, backing up, recovering, or pruning runs; changing storage, provenance, run discovery, asset selection, or run publication behavior. | You are inspecting source code that references run IDs or artifact paths without accessing the artifacts. |
| [Writing Rules](md/notes/writing-rules.md) | Creating, editing, or reviewing scientific articles or their supporting figures, captions, references, parameters, or data bindings; changing shared article components or metadata behavior; updating evidence or availability/version tags affected by implementation or run changes. | An implementation change might eventually produce reportable results, but no article work is currently needed. |
| [Canonical Parameters](md/notes/canonical-parameters.mdx) | Looking up the reference COBA/PING network and TR01/TR02 training parameters. | You need experiment-specific overrides; the experiment recipe remains authoritative. |

When Writing Rules applies, follow its [Editing and dates](md/notes/writing-rules.md#editing-and-dates)
section for substantive saved revisions and incorporated results; preserve
authored update dates for housekeeping and execution alone. If a code or run
change actually invalidates existing article evidence or tags, the necessary
article correction brings Writing Rules into scope; do not defer that correction
by treating it as hypothetical.

## Routing examples

- Explain an experiment function: inspect its source; no rules guide is required
  solely because the code belongs to an experiment.
- Change a recipe: Experiment Rules.
- Launch that recipe remotely and store its run outputs: Experiment, Compute,
  and Pingstore Rules.
- Revise article prose using evidence already supplied: Writing Rules. Add
  Pingstore Rules if accessing run artifacts to retrieve or verify evidence.
- Adjust general site styling: site instructions, not all scientific guides.
  Add Writing Rules if changing shared article components or metadata behavior.

The explanatory [COBA and PING dynamics note](md/notes/coba-ping-dynamics.mdx)
(formerly exp113) covers model and training derivations; it is not a rules guide.
The [mean-field model note](md/notes/mean-field-model.mdx) derives the exp117 and
exp148 closures and their generalisation to four recurrent pathways, shared
input matrices and private neuronal drive; it is also explanatory, not a rules guide.

## Additional instructions

Before editing a Notes article itself, read that article and its versioning or
change-control instructions. When notes are added, renamed, or removed, update
this router and affected documentation links in the same task.

For Astro setup, content APIs, and localhost behavior, also read
[site/README.md](site/README.md).

Follow any additional `AGENTS.md` in the directory you work in.
