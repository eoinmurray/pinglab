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
4. `present.py` — generate figures and report assets: separate raster and phase figures, candidate, background sweep and mechanism diagnostics.
5. `report/` — report, appendix and supporting components.
6. `test.py` — automated checks.

## Cycle decoder

C selection uses one spatial fit per trough-to-trough cycle, containing a whole excitatory volley. Population-rate minima between detected peaks define the boundaries; flat minima use the point nearest the peak midpoint. This avoids artificial changes from combining the unequal spatial halves of adjacent volleys. Peak detection, spatial templates, strength threshold and three-cycle latency rule are unchanged.

Analysis validates the nine trials declared by the current recipe. Additional historical trials stay in their immutable source run and are explicitly listed as excluded in the new analysis. Regenerate analysis and presentation from the pinned compute run to update report metrics without simulation.


## Archived flicker mechanism illustration

The native compute stage also generates an unshifted replay control and a 6-ms delayed competitor packet for the fixed representative attend-A event. Recipe settings own the packet/display windows. Analysis verifies unchanged weights, sender spikes, the pre-intervention prefix, exact control spikes/conductances and every packet spike count; presentation plots actual incoming rates, C spikes and native inhibitory conductance. A red, B blue and C green identify the populations. Native replay and original circuit diagrams are retained.

To add fresh mechanism evidence while retaining the current cue/control compute evidence, explicitly pin the compute run used by the report:

```sh
uv run python experiments/exp172/compute.py --mechanism-source BASELINE_COMPUTE_ID
uv run python experiments/exp172/analyse.py --source MECHANISM_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID
```

The standalone diagnostic performs three fresh simulations; its original must reproduce the pinned representative trial exactly. The timing-intervention panel has been removed from the main report; saved evidence and diagnostic code remain available. No code or data is imported from exploratory work. The illustrated event/window was selected post hoc on one seed; background-driven premature inhibition is a separate mechanism and is not represented by this timing-only intervention.


## Higher-background candidate (Figure 5 / Table 1)

The candidate keeps Figure 2’s parameters and raises only C’s independent background rate from 28 to 32 Hz/channel. A normal full compute/analyse/present execution generates this comparison automatically; Figure 5 reuses Figure 2’s raster renderer; Table 1 compares baseline and candidate cycle metrics.

To add a fresh native candidate to an existing baseline report:

```sh
uv run python experiments/exp172/compute.py --candidate-source BASELINE_COMPUTE_ID
uv run python experiments/exp172/analyse.py --source CANDIDATE_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID
```

The standalone candidate stage runs one simulation from the baseline’s exact native weight arrays. Analysis checks all unchanged settings and weights, exact sender spike equality, and the corrected cycle decoder. The report binds each figure/table to its pinned native run. No exploratory data or code is imported. This candidate was chosen on the displayed seed; it is not a validated optimum and flickering remains.


## Five-input-seed C-background sweep (Figure 6)

```sh
uv run python experiments/exp172/compute.py --background-source BASELINE_COMPUTE_ID --parallel
uv run python experiments/exp172/analyse.py --source BACKGROUND_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID --background-source BACKGROUND_ANALYSE_ID
```

1. The recipe declares rates 0–32 Hz/channel in 4-Hz steps and input seeds 185–189: 45 fresh trials. Source weights, cue 3 Hz/channel and sensory contrast 1× remain fixed. Native bundles, start snapshots and spikes are retained, with one diagram for the unchanged architecture. Estimated retained size is 150–250 MB; dense conductance recordings are unnecessary.
2. `--parallel` uses two isolated CPU workers with one Torch thread each through the maintained active-worker helper. Prepared source weights are shared by checksum; workers own disjoint trials and the coordinator alone completes the run. Without it, the identical plan runs sequentially.
3. Analysis checks every weight array, actual settings, within-seed sender spike identity across rates, all 45 planned trials, and exact reproduction of the original 185/28-Hz baseline. It computes S/F/L per seed with the corrected cycle decoder, then gives each seed equal weight in means. Sample SD uses ddof=1. Missing latencies stay missing; conditional latency means and observation counts are exported explicitly.
4. Figure 6 shows S/F means with SD ribbons and mean L with SD error bars on a second axis, separately for A/B attention. Incomplete groups are labelled n/5. C E/I firing rates and SD appear below. Raw per-seed metrics and counts remain in the analysis results.
5. This adds new experiment-owned evidence to the report without importing exploratory code/data. It varies input realisations in one fixed circuit; seed 185 was used in tuning, so the five-seed screen is not wholly held out. The baseline and candidate retain their existing single-trial evidence.


## Archived receptive-window hypothesis

The archived receptive-window renderer uses the native mechanism recording. Reanalyse the mechanism compute run to obtain arrival-centre inhibition samples, then present with that analysis as `--mechanism-source`. It places C inhibition above sender arrivals and C E spikes, marking peak order and the measured inhibition at each arrival peak. There is no fitted conductance threshold or additional simulation. Low inhibition is a proxy for receptivity, without a hard gating threshold. The controlled retiming example supports timing-dependent competition, rather than a universal first-burst rule.


## Archived arrival-phase distributions

```sh
uv run python experiments/exp172/compute.py --arrival-source FIGURE1_COMPUTE_ID --arrival-candidate-source FIGURE6_COMPUTE_ID
uv run python experiments/exp172/analyse.py --source ARRIVAL_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID --background-source BACKGROUND_ANALYSE_ID --arrival-source ARRIVAL_ANALYSE_ID
```

Two native recorded replays reproduce the exact Figures 2/5 representative-trial spikes and retain C inhibitory conductances. Analysis checks identical weights/source spikes, adds recorded sender delays, assigns every detected arrival to a complete C inhibitory trough-to-trough cycle in the settled windows, and exports per-arrival phases and normalised circular densities. Both panels distinguish A/B while attended/unattended and overlay native mean C inhibition. All arrivals are included rather than filtering for successful transmission. The recipe owns peak/histogram settings; the caption defines the phase reference and limitations. The removed arrival-phase panel is a single-trial comparison, distinct from the five-input-seed means; it is no longer rendered during presentation. No hard receptive-window threshold is inferred.

## Cue-drive sweep (Figure 6)

Hold C background at the background screen’s highest mean S across attend A/B, breaking ties by lower mean F then lower observed latency. Run cue rates 0–6 Hz/channel in 1-Hz steps on input seeds 185–189, with fixed recorded sensory/background inputs and weight arrays. This reuses the tuning seeds; it is not held-out validation.

```sh
uv run python experiments/exp172/compute.py --cue-source BACKGROUND_ANALYSE_ID --parallel
uv run python experiments/exp172/analyse.py --source CUE_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID --background-source BACKGROUND_ANALYSE_ID --arrival-source ARRIVAL_ANALYSE_ID --cue-source CUE_ANALYSE_ID
```

Analysis checks unchanged configurations and native weights, fixed sensory/background rate arrays, all 35 trials, and exact reproduction of all five 3-Hz reference trials. S/F/L use the existing decoder and windows. Figure 6 reuses Figure 5’s renderer, seed SD and missing-latency handling, with faint latency error bars. At zero cue the attended labels describe nominal time periods only.

## Contrast as the cue (Figures 7–8)

At C background 32 Hz/channel and uniform cue zero, multiply only the attended Gaussian excess above the fixed 22-Hz/channel baseline. A is stronger before the switch, B afterwards. The unattended Gaussian stays at its original amplitude. Ratios are 1, 1.25, 1.5, 2, 3, 4, 5 and 6 across seeds 185–189 (40 trials). Ratio 1 must reproduce every no-cue control from the cue sweep. Contrast also changes mean sensory drive; this tests stimulus-strength selection rather than selection of equal-strength stimuli.

```sh
uv run python experiments/exp172/compute.py --contrast-source CUE_ANALYSE_ID --parallel
uv run python experiments/exp172/analyse.py --source CONTRAST_COMPUTE_ID
uv run python experiments/exp172/present.py --source BASELINE_ANALYSE_ID --mechanism-source MECHANISM_ANALYSE_ID --candidate-source CANDIDATE_ANALYSE_ID --background-source BACKGROUND_ANALYSE_ID --arrival-source ARRIVAL_ANALYSE_ID --cue-source CUE_ANALYSE_ID --contrast-source CONTRAST_ANALYSE_ID
```

Native weights and background input remain fixed. Input-rate exports store two epoch profiles for each sensory source; analysis validates their exact contrast switch and confirms zero uniform cue. Figure 7 plots saved 1.25× and 3× pre-switch profiles; Figure 8 reuses the seed-averaged sweep renderer and latency convention.

To extend an existing contrast sweep, pass `--contrast-extend-source PREVIOUS_CONTRAST_ANALYSE_ID` to the compute command. Only missing ratios are simulated. Analysis retains the previous validated seed rows through an explicit pinned input, validates the new trials and recomputes the combined averages. Existing raw runs remain immutable; only the small input-profile files needed for Figure 7 are carried into the new compute export.

## Figures 2–3: raster and phase illustration

Pass `--figure2-seed 186` with the full presentation command and its `--background-source` to use the saved seed-186, 28-Hz background trial for Figure 2. Cue remains 3 Hz/channel. Seeds 185–189 at those same parameters were inspected; 186 was selected post hoc for smoother detected phase rather than higher S. All detected E volley intervals in A/B/C are below 25 ms in this trial. Figure 3A samples the unwrapped sender-minus-C phase once per detected C E peak, with continuous connecting traces and dots only where that sender’s pattern dominates C’s whole-cycle spatial fit (ties, weak responses and incomplete edge cycles remain unmarked); sender phases interpolate successive E peaks without delay correction or phase resets. Figure 3B shows the same samples and dominance dots wrapped to [−0.5, 0.5) cycles, breaking traces at the wrap boundary. Spikes, original mechanism panels and sweep averages retain their earlier evidence. The caption and exported `figure2_selection` record this separate illustration.


## Measured inhibition window in Figure 3B

Record only the saved seed-186 background trial with `compute.py --phase-inhibition-source BACKGROUND_COMPUTE_ID`, then analyse that compute run and pass the resulting analysis to presentation as `--phase-inhibition-source ANALYSE_ID`. Compute requires simulation permission. Every replayed spike array must exactly reproduce the pinned trial. The analysis measures mean native C inhibitory conductance, using complete settled C E cycles and a peak-interpolated phase reference. It shades phases below the midpoint between the measured mean profile’s minimum and maximum, independently of spatial dominance. Reversed C phase plus the native sender delay maps to sender-emission phase under a locally matched-period approximation. This operational low-inhibition window is a receptivity proxy, not a measured transmission threshold. The measured profile, threshold, bounds, cycle count and source pins are retained in analysis and presentation numbers.


## Direct phase-probe response window (Figure 3B)

Run `compute.py --probe-source PHASE_INHIBITION_COMPUTE_ID`, then analyse the new compute run and pass that analysis to presentation as `--probe-source ANALYSE_ID`. The fixed recipe adds a 0.05 µS uniform AMPA pulse (tau 0.75 ms, delay 2 ms) at 16 phases of eight predeclared C cycles, four per cue period. Native runtime states are cloned for every branch; all no-probe continuations reproduce the saved trial exactly and sender spikes remain identical. Retain parameter snapshots per native execution, the eight restart states, compact C E/I spike arrays, phase/timing plans and the single probe architecture. The completed probe export retains approximately 477 MB in 313 files, predominantly repeated native parameter snapshots required by Pingstore validation. Each branch snapshot binds its exact simulation-start parameters; the eight native runtime states support continuation without replaying the full prefix.

The response is net extra C E spikes in the first 5 ms after probe delivery relative to the matched control, including negative values. Shade phases where the mean over eight states reaches half its positive maximum; interpolate crossings on the circular grid. The criterion and pulse strength are fixed before simulation. Keep the response curve, state-level counts and cue-specific means in analysis numbers. This replaces inhibition-derived shading when a probe analysis is supplied. The result is specific to probe strength, response horizon and this single-seed set of states.


## Actually used transmission phases (Figure 3B)

Run `compute.py --used-window-source PHASE_INHIBITION_COMPUTE_ID`, analyse that compute run, and pass its analysis to presentation as `--used-window-source ANALYSE_ID`. This takes precedence over the archived synthetic-probe and inhibition shading. All natural sender packets with complete one- and two-cycle outcomes in settled windows are tested separately, without selection on C dominance. Exact native dual-sender replay preserves the original C pathways, weights and delays. Restore a matched runtime state, remove one whole source volley, and compare with a control that reproduces the original spikes exactly.

Fit spatial templates to integrated C counts over one and two fixed baseline cycles. The persistent effect is the minimum of the sender-amplitude loss and sender-minus-competitor contrast loss at both horizons, with the two-cycle differences normalized per cycle. A positive score therefore requires spatial support at both horizons. Shade 1/16-cycle phase bins with at least three observed packets and median effect at least half the positive supported peak. Retain all signed effects, occupancy, input packet counts and sensitivity checks. Sparse bins are under-sampled, not proven inaccessible; the band denotes strong actually used transmission in this trial, not every possible receptive phase.

The native compute export contains 150 packet deletions from 75 receiver states, 301 simulation calls and approximately 1 GB of evidence; repeated parameter snapshots dominate the size. No synthetic spike injection is used in this measurement. Analysis checks 12/16/24 bins and 0.25/0.5/0.75 peak cutoffs, separates sender and cue conditions, leaves out each receiver state, and normalizes effects by input packet count as sensitivity checks. The appendix retains the compiled replay architecture and method.
