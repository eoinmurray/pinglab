# notebook runners

Every entry in `src/docs/src/pages/notebooks/` has a matching **notebook runner**
here — a single-file Python script that regenerates every figure, video, and
number cited by that entry.

## Naming

Runners are named `<slug>.py`, matching the entry slug (`<slug>.mdx`). The
slug is a short topic kebab-case name (e.g. `dt-stability`, `shd`), not a
numeric ID. The descriptive title lives in the MDX frontmatter and in the
runner's module docstring, not in the filename.

## Contract

Each runner:

- Takes no required arguments. Running `uv run python src/pinglab/notebooks/<slug>.py`
  regenerates everything the entry cites.
- Sets `SLUG = "<slug>"` at module scope. Drives artifact and figure paths.
- Writes raw run outputs to `src/artifacts/notebooks/<SLUG>/` (gitignored).
- Writes published figures, videos, and `numbers.json` to
  `src/docs/public/figures/notebooks/<SLUG>/` (committed). The MDX imports
  from that directory.
- Wipes both directories by default on start. Pass `--no-wipe-dir` to keep
  stale artifacts (rarely useful — cache drift is how figures fall out of
  sync with code).
- Records its own run identifier (`rNNN` counter, via `_run_id.py`) and
  stamps it onto every video.

A runner is the **promotion gate** — if it doesn't output a figure, that
figure isn't published. No figures are produced by other means.

## Size tiers

Runners pick from fixed size buckets (extra small / small / medium / large / extra large)
defined in `src/docs/src/pages/styleguide.md` § 8. The tier is recorded in
`numbers.json` and surfaced in the MDX.

## Files

- `scope-frame.py` — scope-frame aesthetic reference (canonical SCOPE_FRAME still).
- `ping-regimes.py` — basic PING videos (stim-overdrive, dt, ei_strength sweeps).
- `dt-stability.py` — cuba vs standard-snn Δt-stability (train at one dt, sweep eval-dt; includes snntorch-library as external parity reference).
- `shd.py` — SHD (Heidelberg spoken digits) classification baseline.
- `_run_id.py` — per-runner monotonic counter helper.
