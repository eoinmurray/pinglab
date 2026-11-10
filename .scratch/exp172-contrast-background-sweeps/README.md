# exp172 sensory-contrast and C-background screens

Run from the repository root:

```sh
uv run python .scratch/exp172-contrast-background-sweeps/run.py
```

1. Contrast multipliers: 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6.5. Both sender profiles change about their fixed 26 Hz/channel expected mean: r_c(x) = 26 + c[r_1(x) − 26]. The baseline flat component is 22 Hz/channel, so its new value is 26 − 4c. Sigma is unchanged. Zero is a uniform-input control.
2. Independent C drive: 0–32 Hz/channel in steps of 4. Contrast stays at 1; cue stays at 3 Hz/channel in both sweeps. No joint grid or combined condition is run.
3. Use exact pinned native weights from exp172-r132-compute, representative seed 185. Seventeen fresh simulations share one newly reproduced baseline. Its six spike arrays must exactly match the original; sender arrays must remain unchanged throughout the C-background sweep. Common random-number input streams pair conditions; realised counts need not remain equal when rates change.
4. All outputs overwrite this project's `data/` folder. Native bundles, start snapshots, all-population spikes, input rates/configuration, metrics, code hashes and timings are retained. One circuit diagram covers the shared architecture. Expected size is about 100 MB across 17 fixed conditions; no dense conductance recordings are needed.
5. `data/results.md`, `contrast.png` and `background.png` present S/F, latency in cycles on a second axis, and receiver firing rates. CSV/JSON preserve cycle resolution, missing outcomes and each trial's data. Use `--analyse-only` to regenerate metrics/plots without simulation.
6. This is a one-seed first screen, not held-out validation. Parameters remain unchanged in the formal experiment.

## Completed screen

Seventeen fresh runs completed in 250.25 s wall time (231.07 s summed native simulation); final analysis and plotting took 1.06 s. Retained size is 59 MB. All 34 metric rows, rate means, unchanged-input hashes and F transition fractions passed validation; native run-time checks verified exact fixed weights, baseline reproduction and unchanged sender spikes in the background sweep.

No tested setting eliminates flicker in both periods. Current contrast 1 is the best tested contrast for joint selection success. The best tested background candidate is 32 Hz/channel: S becomes 0.900/0.872, F 0.103/0.184 and acquisition/switch latency remains 3/1 cycles for A/B. A still has four changes (the tiny F decrease reflects a larger denominator); B has seven rather than eight measured changes, with a different state at the settled-window start. This one-seed result is a candidate direction, not a validated reduction in event frequency or an optimum. The formal experiment remains unchanged.

## Selected candidate and raster

```sh
uv run python .scratch/exp172-contrast-background-sweeps/candidate.py
```

This presents the existing `background-32` condition (contrast 1×, C background 32 Hz/channel, cue 3 Hz/channel), selected by maximising worst-period S, then mean S, with lower mean F as the accuracy-tie breaker. It independently has the highest S in both periods and the lowest A F; the lowest B F alone requires worse accuracy and latency. The exact combination is already in the measured sweep, so this step runs no simulation. `data/candidate.png`/SVG, JSON and Markdown contain the A/B/C raster and metrics. Corrected cycle metrics and weight equality are rechecked; rendering took 0.52 s. Grey C intervals retain the remaining unattended-dominant settled cycles. Formal exp172 settings remain unchanged.
