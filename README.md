# Pinglab

Conductance-based spiking neural networks with excitatory/inhibitory populations
and PING (pyramidal–interneuron gamma) dynamics.

Site: [pinglab.eoinmurray.info](https://pinglab.eoinmurray.info).

Main documentation lives in [`md/notes/`](md/notes/):

1. [Experiment Rules](md/notes/experiment-rules.md): experiment structure, stages and scientific inputs.
2. [Compute Rules](md/notes/compute-rules.mdx): compute providers, access and execution workflows.
3. [Pingstore Rules](md/notes/pingstore-rules.md): run storage, provenance, validation, backups and pruning.
4. [Writing Rules](md/notes/writing-rules.md): scientific writing, article metadata and editorial conventions.
5. [Canonical Parameters](md/notes/canonical-parameters.mdx): reference network and training settings for COBA and PING.
6. [COBA and PING Dynamics, Discretisation, and Learning](md/notes/coba-ping-dynamics.mdx): model and training derivations (formerly exp113).

See [Astro setup and GitHub Pages deployment](site/README.md)
for site development and deployment. Coding agents must read
[`AGENTS.md`](AGENTS.md) for task-specific instruction routing.

Use the [Modal dispatcher](tools/modal_dispatcher/README.md) to launch experiment
compute on Modal and collect validated Pingstore results.

For generic CSD3 / Wilkes3 experiment deployment, see [HPC deployment](tools/hpc/README.md).
