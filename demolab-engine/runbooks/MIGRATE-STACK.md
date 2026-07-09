# Runbook: Migrating the stack to another language

> Let the user write their **science in their language** while Typst publishing and the
> file-based contract stay exactly as they are.

## When to use
When the user's science lives in MATLAB, Julia, R, or another language and they'd rather not
port it to Python — Python is demolab's *opinionated example*, not a lock-in. This needs
almost no restructuring because the contract is *file-based and language-neutral* (see
`../guides/RULES.md`): a tool is reached by **running its CLI** (a subprocess), never by
importing, and it communicates only through files — `config.json`, `output.json`,
`manifest.json`, `output.log`, `run.sh`, and its figure data (`<cmd>.csv`, `.json`, `.npz`,
… — author's choice of format). Any executable that parses arguments and writes those files
is a valid tool. Typst publishes from `numbers.json` + PNG figures and doesn't care what
produced them. So only the **tool** layer is language-bound — pick the smallest change that
gets the user's language in.

## What it does

0. **Confirm the three primitives.** Before starting, confirm the target language can do all
   three — every mainstream scientific language can:

   | | run non-interactively | write JSON (contract files) | write figure data (CSV shown; any format) | unit tests |
   |--|--|--|--|--|
   | **MATLAB** | `matlab -batch "<expr>"` (R2019a+) | `jsonencode` / `jsondecode` | `writematrix` / `writetable` | `runtests` |
   | **Octave** (free MATLAB) | `octave --eval "<expr>"` | `jsonencode` (pkg `io`) | `csvwrite` / `dlmwrite` | `test` |
   | **Julia** | `julia tool.jl <args>` or `julia -e` | `JSON3.jl` | `CSV.jl` / `DelimitedFiles` | `Test` stdlib |
   | **R** | `Rscript tool.R <args>` | `jsonlite` | `write.csv` | `testthat` |

Offer the hybrid first (least churn):

**A. Hybrid (recommended): tools in the new language, Python stays the glue.**
1. **Prereq.** Confirm the runtime is on PATH (see the table) and add it to the toolchain check in the *Doctor the repo* runbook.
2. **Write the tool in their language.** `tools/<tool>/tool.<ext>` — a `tool(cmd, args…)` entry point that parses args, runs the science, and writes the contract files with that language's JSON + data-file primitives (columns above), including the `run.sh` reproducer (which re-invokes the tool the same way the runner does) and `output.log`. Port `setup_run_dir`/`write_output` **once** into a shared helper in that language (e.g. `tools/<tool>/dl_io.m` / `dl_io.jl`) and reuse it across tools — same manifest validation (`headline_metrics` must exist in `output.json`; a declared `headline_video` must exist on disk). **The tool emits data, not plots.**
3. **Keep the runner's shape.** `experiments/expNNN.py` still shells out to the tool — swap the `python tools/<tool>/tool.py lif` invocation for the target's (e.g. `matlab -batch "cd('tools/<tool>'); tool('lif', ...)"`, `julia tools/<tool>/tool.jl lif …`). It still reads `manifest.json` + the tool's data file and renders the figure with matplotlib into `artifacts/data/expNNN/`. Keep a minimal `uv` env (numpy + matplotlib) purely for staging + plotting.
4. **Tests in their language.** `tools/<tool>/test_<tool>.<ext>` run with the target's test runner (table); wire it into `task test`.

**B. Full switch: tool *and* runner in the new language.** Do A, then rewrite the runner in that language (`experiments/expNNN.{m,jl,R}`) to render figures itself (MATLAB `exportgraphics`, Julia `Plots.jl`/`Makie`, R `ggplot2` → PNG in `artifacts/data/expNNN/`) and write `numbers.json`. Drop the Python deps and point `task run` at the new runtime. Julia is the natural fit here — it can own the tool, the runner, *and* the plotting. Take this only if the user wants Python out entirely.

**Untouched in both paths:** `demolab-engine/` (Typst), `writings/*.typ` (they read `numbers.json` the same way), `artifacts/`, and the `numbers.json` schema. Update the `Taskfile` (`run` / `test`) and the toolchain note at the top of `AGENTS.md` to name the new runtime. Provenance still works — reimplement the `_provenance` stamp (git SHA, `dirty` flag, UTC timestamp) in the new `setup_run_dir` equivalent.

---

## Agent contract
- **Triggers** — `MIGRATE-STACK`, "migrate the stack to MATLAB / Julia / R", "rejig the repo
  for `<language>`", "use MATLAB / Julia instead of Python", "my science is in `<language>`".
- **Gates** — §0: the target language must cover all three primitives (run non-interactively,
  write JSON contract files, write figure data) plus unit tests.
- **Report & apply** — offer the hybrid (A) first as the least-churn path; take the full
  switch (B) only if the user wants Python out entirely.
