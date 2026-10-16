# Agent guide router

Start with [README.md](README.md) for the repository overview. All four articles
in the Notes collection are canonical agent instructions. Read the relevant
live files before acting; follow every applicable note when a task crosses
boundaries. Revisit this routing when the task's scope changes, including when
moving from inspection to editing, execution, publication, or cleanup.

1. [Experiment Rules](md/notes/experiment-rules.md): read before designing,
   creating, editing, or running experiments, recipes, compute/analyse/present
   stages, execution helpers, or stage-input selection. Also read before
   changing experiment discovery, preview, or publication behavior.
2. [Compute Rules](md/notes/compute-rules.mdx): read before planning or launching
   compute, choosing a provider or backend, accessing remote machines, submitting
   or monitoring jobs, collecting outputs, or terminating compute resources.
   Apply it when editing scripts or automation that perform these operations.
3. [Pingstore Rules](md/notes/pingstore-rules.md): read before creating, consuming,
   validating, importing, assembling, transferring, backing up, recovering, or
   pruning runs and scientific artifacts, or editing their storage/provenance
   code. Also read before changing run discovery, asset selection, or publication.
4. [Writing Rules](md/notes/writing-rules.md): read before creating, editing, or
   reviewing experiment articles and supporting article files, figures,
   captions, references, parameters, or data bindings. Also read before changing
   shared article components or metadata behavior, and when implementation or
   run changes affect an article's evidence or availability/version tags.
   Apply section 3.3 to substantive saved revisions and incorporated results;
   preserve authored update dates for housekeeping and execution alone.

Before editing a Notes article itself, read that article and its versioning or
change-control instructions. When notes are added, renamed, or removed, update
this router and affected documentation links in the same task.

For Astro setup, content APIs, and localhost behavior, also read
[site/README.md](site/README.md). Before building or deploying published run
assets, also read [Publishing](tools/publishing/README.md).

Follow any additional `AGENTS.md` in the directory you work in.
