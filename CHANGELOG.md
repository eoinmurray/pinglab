# Changelog

All notable changes to pinglab. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), version scheme: [SemVer](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-01

First tagged release. Closes the *Δt-stability* arc — six SNN models trained at two timesteps each (12 cells), all converged within 86–89% on MNIST under a uniform recipe.

### Added

- **Notebooks nb000 → nb013**: full per-model calibration set (standard-snn, standard-snn-exp, snntorch-library, cuba, coba, ping), the dt-stability comparison (nb012), and the SHD baseline (nb013, in-progress). nb000 is the perf baseline.
- **Models** (src/pinglab/models.py): `CUBANet` (snntorch + zoh discretisations, optional exp synapse), `SNNTorchLibraryNet` (parity reference), `COBANet` (conductance-based with hard reset, refractory, optional E→I→E loop).
- **Oscilloscope CLI** (src/pinglab/oscilloscope.py): unified `train` / `infer` / `image` / `video` / `sim` subcommands with shared flags. `--modal --modal-gpu {T4,L4,A10G,A100,H100}` for remote dispatch. `--from-dir` for config-and-weights inheritance.
- **Readouts**: `rate`, `li`, `spike-count`, `mem-mean`. mem-mean (snnTorch tutorial 5 pattern) is the default for all six nb012 models, with `--readout-w-out-scale` to compensate COBANet's lower hidden firing rate.
- **Output LIF (mem-mean / spike-count)**: exp-Euler + ZOH form, subtract reset, configurable τ_out via `--readout-tau-out`.
- **Eval-time dt transport**: count-preserving upsample / downsample (Parthasarathy et al. §2.1 / §2.3) and resample, anchored at identity transport.
- **Latency-to-correct-answer probe**: P(correct) vs fraction-of-trial-seen, monotonic-by-construction under mem-mean.
- **Loss options**: `--loss {ce, mse}` for cross-entropy or L2 between logits and one-hot targets.
- **Modal batch dispatch**: 12-job parallel runs with single `volume get` at the longest common prefix (sync time 80 min → 3 min).
- **Docs site**: Astro-based, notebook-per-MDX, `<Figure>` / `<NotebookHeader>` / `<RunParameters>` shared components. Frontmatter `collections:` groups notebooks on the homepage.
- **Tests**: 223 unit tests covering encoders, models, flag propagation, mode drift, fuse parity, LIF expeuler, spectral, smoke. ty-clean.

### Findings

- **Bias balloon**: in the snnTorch / Tutorial-3 update $U_{t+1} = \beta U_t + Ws_t + b$, the steady-state membrane $U_{ss} \approx Wr\tau + b\tau/\Delta t$ has a bias term that scales as $1/\Delta t$. Halving eval-dt doubles bias drive; firing rate rises with it. The cuba/coba/ping family restores $(1-\beta)/\Delta t$ and $(1-\beta)$ prefactors at integration time, giving $U_{ss} = Wr + b$ — *dt*-invariant by construction.
- **Output drive scaling under mem-mean**: COBANet's biophysical hidden firing rate (~5–30 Hz) is ~10–100× lower than CUBANet's (~200–300 Hz), so per-step output drive is correspondingly weaker and CE gradient stays at chance. `--readout-w-out-scale 100` (coba) / 500 (ping) at init recovers learning.
- **Spike-count vs mem-mean**: pure spike-count's gradient flows only through the surrogate at threshold crossings (sparse, high-variance) and trained at chance for COBANet; mem-mean's per-step CE on output membrane gives a dense gradient signal.

### Repo hygiene

- Cleaned 13 unused dependencies from `pyproject.toml` (networkx, joblib, tqdm-joblib, imageio, fastapi, orjson, uvicorn, httpx, pyarrow, watchdog, rich, streamlit, typer).
- `ruff format` pass across 46 files.
- `ty check` clean.

[Unreleased]: https://github.com/eoinmurray/pinglab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/eoinmurray/pinglab/releases/tag/v0.1.0
