#!/usr/bin/env bash
# Copy the ar009 ("gamma-gated sparsity") manuscript figures from the notebook
# outputs into figures/ so the LaTeX build is self-contained. Re-run after
# regenerating any figure in the notebooks.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
SRC="$ROOT/src/docs/public/figures/notebooks"
DST="$(cd "$(dirname "$0")" && pwd)/figures"
mkdir -p "$DST"
figs=(
  nb023/overview_compound.png
  nb054/onset_super_compound.png
  nb025/results_compound.png
  nb038/loop_transfer_compound.png
  nb049/training_curves.png
  nb041/rate_vs_fgamma.png
  nb046/spikes_per_cycle_distribution.png
  nb047/rate_vs_w_ie.png
  nb037/perturbation_curves.png
  nb042/rhythm_compound.png
  nb044/dt_sweep.png
  nb048/varying_headline_stream.png
  nb048/acc_grid_tau_rate.png
)
for rel in "${figs[@]}"; do
  flat="${rel/\//_}"          # nb023/overview... -> nb023_overview...
  cp "$SRC/$rel" "$DST/$flat" && echo "copied $flat"
done
