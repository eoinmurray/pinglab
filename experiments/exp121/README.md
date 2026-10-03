# Decision timing across input and internal parameters

Neuromodulation study with seven matched sweeps and one fixed trained network.
Every figure places decision-time bar histograms in a 2×2 grid (three panels for
three-condition sweeps), plus network frequency and
burst-interval CV against its control parameter. Results prose is omitted;
figure captions and short Methods steps explain the measurements.

## Independent stages

1. Input compute: `uv run python experiments/exp121/compute.py --dataset-root data`.
   Downloads missing MNIST data, selects 100 test images deterministically and
   runs input rates 5, 2.5, 10 and 20 Hz. Default internal settings are fixed.
2. Internal compute: `uv run python experiments/exp121/compute.py --internal-from <input-compute>`.
   Validates the exact paired protocol and training source, reuses recorded
   baseline and image data, then runs the six internal sweeps (1,600 new
   presentations). `--studies` selects explicit sweeps, allowing independently
   added scientific controls to reuse completed earlier sweeps without rerunning
   them. Input is 5 Hz; baseline is included by explicit reuse.
   For the added controls: `--studies ampa inhibition threshold` runs 700 new
   presentations.
3. Analyse: `uv run python experiments/exp121/analyse.py --source <input-compute> --internal-source <internal-compute>`.
   Repeat `--internal-source` for each internal compute run. Validates matching
   provenance, rejects duplicate studies, and saves measurements in figure order.
4. Present: `uv run python experiments/exp121/present.py --source <analyse-run>`.
   Composes Figures 1–7 with snnviz. No downstream stage executes upstream work.

Code owns its SNNLang definition and imports no scratch or other experiment
implementation. All operational inputs/outputs use validated v4 runs, except
the external official MNIST cache, whose raw file hashes are recorded.

## Protocol

1. Trained seed 42, pinned best-validation weights; no retraining. Sample seed
   120082 chooses 10 official test images per class. Encoding seed is 820000
   plus test index; common uniform draws pair rates. Internal interventions
   receive exactly the baseline spike stream, verified by hash.
2. Presentations last 400 ms, dt=0.1 ms; fresh dynamic state for every trial.
   Baseline E/I capacitances 1/0.5 nF, leaks 0.05/0.10 µS, AMPA/GABA 2/6 ms.
   Shared capacitance/leak scaling preserves each E/I parameter ratio. GABA
   changes both inhibitory projections at fixed peak weights and fixed AMPA.
3. AMPA decay is 2, 1.8, 1.6 or 1 ms on input→E, E→E and E→I; GABA stays
   6 ms and the readout stays unchanged. Peak weights are fixed, so shorter
   decay also reduces integrated excitation. I→E strength is 1, 0.8 or 1.2
   times the imported trained matrix; I→I and all other weights are unchanged.
   Excitatory thresholds are −50, −52 or −48 mV; I remains −50 mV. The graph
   executor reads the authored population threshold; the baseline remains −50 mV.
4. Stable-correct time is the earliest unique true-class count lead that holds
   through 400 ms. Final failures/ties have no time; histograms do not treat
   them as 400-ms decisions. Paired timing summaries use joint successes and
   report rescues/losses separately.
5. I-volley frequency is inverse mean interval; CV uses ddof=0. Both require
   three peaks. Detector: 1-ms rate bins, Gaussian sigma 2 ms, height/prominence
   25 Hz/neuron, separation 10 ms. Thresholds 10 and 50 are also measured.
   All 400 ms, including onset, are used. Missing estimates are excluded from
   median/IQR summaries and valid counts are reported. These are descriptive
   image/encoding distributions, not uncertainty across training replicates.

## Consolidation

Experiments exp118–120 are retired as separate code/articles. Their historical
runs remain unchanged. Their earlier single-image two-dimensional grids are
not claimed as the matched multi-image sweeps here; new simulations supply the
missing conditions and activity. Their IDs must never be reused.

## Completed consolidated execution

1. Input compute: exp121-r002-compute (reused unchanged). Internal compute:
   exp121-r005-compute and exp121-r009-compute. Combined analysis:
   exp121-r010-analyse. Presentation: exp121-r011-present, selected by the article's explicit local default.
2. Every measurement array for Figures 1–4 exactly matches exp121-r006-analyse. All new inputs passed baseline hash
   checks. Historical exp118–120 runs were preserved unchanged.
3. Added AMPA, I→E strength and E-threshold sweeps executed 700 presentations.
   A diagnostic reproduced the first stored baseline output and I activity
   exactly after fixing graph execution to honour population-specific thresholds.
4. No test suite was run; validation used executed stages, Ruff, artifact
   inspection and validated v4 stage inputs. A focused threshold regression
   test is included but has not been run.
