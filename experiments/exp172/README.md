# exp172 — Spatial Gaussian Selective Relay

Test how a uniform rate cue selects between two spatial Gaussian input patterns in an
A/B → C spiking network.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp172/compute.py
uv run python experiments/exp172/analyse.py --source COMPUTE_ID
uv run python experiments/exp172/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp172/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `report/` — report, appendix and supporting components.
6. `test.py` — automated checks.

## Cycle decoder

C selection uses one spatial fit per trough-to-trough cycle, containing a whole excitatory volley. Population-rate minima between detected peaks define the boundaries; flat minima use the point nearest the peak midpoint. This avoids artificial changes from combining the unequal spatial halves of adjacent volleys. Peak detection, spatial templates, strength threshold and three-cycle latency rule are unchanged.

Analysis validates the nine trials declared by the current recipe. Additional historical trials stay in their immutable source run and are explicitly listed as excluded in the new analysis. Regenerate analysis and presentation from the pinned compute run to update report metrics without simulation.


## Flicker mechanism illustration (Figure 5)

The native compute stage also generates an unshifted replay control and a 6-ms delayed competitor packet for the fixed representative attend-A event. Recipe settings own the packet/display windows. Analysis verifies unchanged weights, sender spikes, the pre-intervention prefix, exact control spikes/conductances and every packet spike count; presentation plots actual incoming rates, C spikes and native inhibitory conductance. A red, B blue and C green identify the populations. Native replay and original circuit diagrams are retained.

To add fresh mechanism evidence while retaining the current cue/control compute evidence, explicitly pin the compute run used by the report:

```sh
uv run python experiments/exp172/compute.py --mechanism-source BASELINE_COMPUTE_ID
uv run python experiments/exp172/analyse.py --source MECHANISM_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID
```

The standalone diagnostic performs three fresh simulations; its original must reproduce the pinned representative trial exactly. A normal full compute/analyse/present execution includes Figure 5 automatically. No code or data is imported from exploratory work. The illustrated event/window was selected post hoc on one seed; background-driven premature inhibition is a separate mechanism and is not represented by this timing-only intervention.


## Higher-background candidate (Figure 6 / Table 2)

The candidate keeps Figure 1’s parameters and raises only C’s independent background rate from 28 to 32 Hz/channel. A normal full compute/analyse/present execution generates this comparison automatically; Figure 6 reuses Figure 1’s raster renderer and Table 2 reuses Table 1’s formatting and metric definitions.

To add a fresh native candidate to an existing baseline report:

```sh
uv run python experiments/exp172/compute.py --candidate-source BASELINE_COMPUTE_ID
uv run python experiments/exp172/analyse.py --source CANDIDATE_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID
```

The standalone candidate stage runs one simulation from the baseline’s exact native weight arrays. Analysis checks all unchanged settings and weights, exact sender spike equality, and the corrected cycle decoder. The report binds each figure/table to its pinned native run. No exploratory data or code is imported. This candidate was chosen on the displayed seed; it is not a validated optimum and flickering remains.


## Five-input-seed C-background sweep (Figure 7)

```sh
uv run python experiments/exp172/compute.py --background-source BASELINE_COMPUTE_ID --parallel
uv run python experiments/exp172/analyse.py --source BACKGROUND_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID --background-source BACKGROUND_ANALYSE_ID
```

1. The recipe declares rates 0–32 Hz/channel in 4-Hz steps and input seeds 185–189: 45 fresh trials. Source weights, cue 3 Hz/channel and sensory contrast 1× remain fixed. Native bundles, start snapshots and spikes are retained, with one diagram for the unchanged architecture. Estimated retained size is 150–250 MB; dense conductance recordings are unnecessary.
2. `--parallel` uses two isolated CPU workers with one Torch thread each through the maintained active-worker helper. Prepared source weights are shared by checksum; workers own disjoint trials and the coordinator alone completes the run. Without it, the identical plan runs sequentially.
3. Analysis checks every weight array, actual settings, within-seed sender spike identity across rates, all 45 planned trials, and exact reproduction of the original 185/28-Hz baseline. It computes S/F/L per seed with the corrected cycle decoder, then gives each seed equal weight in means. Sample SD uses ddof=1. Missing latencies stay missing; conditional latency means and observation counts are exported explicitly.
4. Figure 7 shows S/F means with SD ribbons and mean L with SD error bars on a second axis, separately for A/B attention. Incomplete groups are labelled n/5. C E/I firing rates and SD appear below. Raw per-seed metrics and counts remain in the analysis results.
5. This adds new experiment-owned evidence to the report without importing exploratory code/data. It varies input realisations in one fixed circuit; seed 185 was used in tuning, so the five-seed screen is not wholly held out. Figures 1–6 retain their existing single-trial evidence.
