# exp136 — Fixed sustained-cue PING comparison

The default runner executes exactly two 4 s trials at the carried-forward setting:
constant one-way coupling and constant zero coupling. The cue sequence is no
cue, S1, S2, no cue, each lasting 1 s. No weights change during a trial.

| Setting | Value |
| --- | --- |
| Sender-to-R strength per pathway | 0.045 μS, or zero in the control |
| Cue-to-sender strength | 0.06 μS |
| Sender / receiver background | 33 / 35 Hz per input channel |
| PLV window / stride | Centred 200 ms / 10 ms |
| Time step | 0.1 ms |

## Run locally

Each stage is explicit and independent; substitute the printed identities when
creating new runs.

```sh
.venv/bin/python experiments/exp136/compute.py
.venv/bin/python experiments/exp136/analyse.py --source exp136-r065-compute
MPLCONFIGDIR=/tmp/exp136-matplotlib .venv/bin/python experiments/exp136/present.py --source exp136-r066-analyse
```

The verified compute run took 26.5 s for coupling on and 23.5 s for coupling off
on one local CPU thread (50.0 s of simulation, plus input/storage overhead).
Analysis took under a second. The default task has no parameter searches,
uncued extra trials, reciprocal modes or scheduled coupling. Explicit optional
signal-routing searches are described below.

## Exact reproducibility

`recipe.py` defines the fixed circuit and parameters; its CLI can also export
an authoring bundle. Compute defaults to the explicit retained parameter pin
`exp136-r059-compute`, native unit `k0450-q0600-noise0-execution`, to preserve the
selected realised weights rather than initialise a different network. It reads
only native parameters and does not execute or select prior conditions.
`--source` can name a compatible retained compute run explicitly.

Both new spike outputs were checked against the selected historical trials:
zero differences for all eight outputs in both conditions. Native start/end
snapshots verify the intended static matrices; sender and cue activity is
identical across the two new trials. Both use the same binary input arrays.

The earlier signal-test presentation is `exp136-r075-present`. It preserves the two original
synchrony figures from `exp136-r067-present`, keeps signal-decoding measurements in the exported numbers,
and displays one native circuit diagram (SVG/PNG each), plus `numbers.json`. The circuit diagram
is above the appendix parameter/input tables. The historical selection evidence
`exp136-r064-present` is an explicit presentation input, preserving provenance for
the report's earlier new-input checks without repeating those simulations.

## Scientific limit

This setting was selected using the original input sequence. Earlier checks
with three new input sequences passed zero of three trials under all strict
locking criteria, despite preserved selectivity and near-zero selected drift.
The current pair reproduces the selected example; it does not add independent
evidence for robustness or stimulus routing. Historical completed runs remain
unchanged.

## Independent-signal routing assay

The optional attention task leaves the fixed network weights unchanged. It adds
independent, continuously present sender drive modulations bounded by ±1.5 Hz per
channel. Three training and three test signal/noise sequences each run in both
cue orders, with exact sensory/noise pairing across orders. Three execution
batches of four lanes keep local memory bounded; these are trial batches, not
parameter searches. Nine simulations compare normal coupling, zero coupling,
and native emitted-spike replay with per-cell timing independently shuffled
inside 50 ms blocks. The latter preserves every sender E cell's exact count in
every block, while changing timing and coincidences.

```sh
.venv/bin/python experiments/exp136/compute.py --task attention
.venv/bin/python experiments/exp136/analyse.py --source exp136-r068-compute
MPLCONFIGDIR=/tmp/exp136-matplotlib .venv/bin/python experiments/exp136/present.py --source exp136-r072-analyse
```

The verified computation is `exp136-r068-compute`; final analysis is
`exp136-r072-analyse` and presentation is `exp136-r075-present`. Decoder refinements
reused the same immutable simulation evidence; no model parameters were retuned.
The new presentation explicitly references the original synchrony presentation.
It exports seven files in total: two scientific figures and one circuit view
in SVG/PNG, plus numbers. The signal-decoding chart and its renderer have been
removed; optional signal computation and decoding remain available.

Readouts use causal 20 ms population-rate windows with lags 0–40 ms to reconstruct
the sensory signal 20 ms before the latest observation. Ridge penalty is fixed
at 10; training data alone determines centring, scaling, and coefficients. One
pooled decoder per condition/signal is shared across cues. Scores are held-out
R² per independent test sequence, pooling its paired cue orders. Negative values
are retained. Cue benefit compares the same signal when cued versus uncued;
cross-signal score differences alone can reflect unequal decodability.

Input and local sender readouts diagnose failures before interpreting R. The
first result is negative: input readouts score 0.597/0.410, local sender readouts
0.017/−0.144, and cued receiver scores −0.062/−0.117 for S1/S2. Cue gains are
−0.055/+0.095. The disconnected receiver has zero cue benefit, with identical
spikes across both orders. Sender gamma power fraction drops from about 0.47
to 0.27 under scrambling (30–80 Hz power relative to 1–150 Hz). This does not
establish selective routing, and weak local decoding prevents interpreting the
receiver failure as absence of a possible neural code. Three test sequences and
one wiring realisation provide descriptive evidence only.

Nineteen scientific checks cover input/cue pairing, independent splits, causal
features, held-out leakage, known-signal reconstruction, count-preserving
scrambling, and the retained timing/control tests. Native snapshots and emitted
replay equality are checked during compute. All zero-coupling receiver trajectories
were also verified identical across paired cue orders.

## Coarse-to-fine signal-routing search

`--task coarse` starts with the Figure 1–2 matrices, mean drives and cue schedule.
It runs the original signal setting plus 16 reproducible Latin-hypercube points
over four axes: modulation amplitude 1.5–15 Hz/channel, Gaussian smoothing
25–250 ms, coupling 0.015–0.135 μS, and cue strength 0.015–0.18 μS.
Connection masks and local weights remain fixed; only pathway totals are scaled.
Every weight remains constant within each trial. Receiver background remains
35 Hz/channel and both sender means remain 33 Hz/channel.

Each candidate uses one independent decoder-training sequence and one validation
sequence, each in both cue orders (four lanes). All candidates share the same
random draws. The fixed readout uses trailing 50 ms E/I population rates at lags
0, 25 and 50 ms, predicting the signal 50 ms before the observation. This longer
readout is applied to the original setting too. Ranking first requires both
sender readouts to reach R² ≥0.1, next prefers positive receiver scores for both
cued signals, then maximises the smaller of the two same-signal cue benefits.
Remaining ties use minimum cued receiver R², minimum sender R², then candidate
order. The fallback when no sender pair reaches threshold maximises the weaker
sender's score; it is not labelled successful attention.

`--task fine --selection <coarse-analyse>` includes the coarse-selected centre
and 16 new space-filling points within ±35% of its values, clipped to the coarse
bounds. Validation is reused for refinement; it is explicitly not a test set.
`--task confirm --selection <fine-analyse>` then runs just the chosen setting on
three new training and three new test sequences, in both cue orders, with
coupled, zero-coupling and count-preserving timing controls. Neither search
uses these final sequences. Readouts are refitted only on fresh training data;
test data never sets scaling, coefficients or model parameters. The timing
control must also retain readable sender signals before timing support is
reported. It changes sender-local dynamics as well as transmitted timing and
therefore does not isolate interareal phase alone.

Each command creates one immutable run. Analyse and present are separate:

```sh
.venv/bin/python experiments/exp136/compute.py --task coarse
.venv/bin/python experiments/exp136/analyse.py --source <coarse-compute>
.venv/bin/python experiments/exp136/compute.py --task fine --selection <coarse-analyse>
.venv/bin/python experiments/exp136/analyse.py --source <fine-compute>
.venv/bin/python experiments/exp136/compute.py --task confirm --selection <fine-analyse>
.venv/bin/python experiments/exp136/analyse.py --source <confirm-compute>
MPLCONFIGDIR=/tmp/exp136-matplotlib .venv/bin/python experiments/exp136/present.py --source <confirm-analyse>
```


### Completed search and confirmation

Coarse `exp136-r076-compute` → `exp136-r077-analyse`; fine
`exp136-r078-compute` → `exp136-r079-analyse`; fresh confirmation
`exp136-r080-compute` → `exp136-r081-analyse` → `exp136-r082-present`.
The report now pins `exp136-r082-present`, with nine exports: three scientific
figures and one circuit view (SVG/PNG each), plus numbers. The original two
synchrony figures are reused byte-for-byte. The repeated coarse/fine centre
also reproduced all eight spike arrays bit-for-bit.

The selected candidate is `fine-09`: amplitude 12.211245278590185 Hz/channel,
smoothing 82.15332918293204 ms, forward total 0.07980206834425732 μS and cue
total 0.11553306711764234 μS. Fresh mean sender R² is 0.51672/0.48398; cued R
R² is 0.08458/−0.00769; cue benefit is 0.32163/0.31824. Scrambled cued R² is
0.20349/0.14756 but cue benefit is −0.09133/−0.16493. Zero-coupling cue benefit
is exactly zero for every pair/signal. Neither the selective-routing nor the
strict timing-support criterion passed. This is improved encoding and a mean
cue bias, not a reliable attention task or clean CTC evidence.

## Readout and receiver refinement

The readout audit reuses `exp136-r080-compute` without changing or simulating any
neurons. Its previously inspected test pairs are now development validation.
It compares 25, 50, 100 and 200 ms windows, population E/I means (six features)
or individual-neuron rates (300 features), and ridge penalties 1, 10, 100 and
1000. All candidates use the same 50 ms target delay, lags 0/25/50 ms, 10 ms
sampling and identical scored times: 1350–1990 and 2350–2990 ms. The common
350 ms exclusion accommodates the longest history. Only training data sets
feature scaling and coefficients. Each target retains one fixed decoder across
both cues; no cue label is supplied as a feature.

Selection now first requires both sender mean R² ≥0.1 and positive same-signal
cue benefit in both directions, then maximises the weaker cued receiver R².
Remaining ties use minimum cue benefit, minimum local R² and candidate order.
If no candidate satisfies both gates, the best-ranked fallback is still
reported but never labelled a successful attention result.

The completed audit is `exp136-r084-analyse`. It selected population counts,
50 ms windows and penalty 1000; minimum cued validation R² was −0.0033.
Individual-neuron counts and longer windows did not solve receiver decoding.

The targeted receiver search freezes that readout, the selected encoding
(±12.211 Hz/channel, 82.153 ms smoothing) and cue total (0.115533 μS). It crosses
R backgrounds 29/32/35 Hz/channel with coupling totals 0.04/0.079802/0.12 μS.
All nine settings use the same three training and three validation pairs in
both cue orders. The mean sender background remains 33 Hz/channel. Neural
weights remain constant throughout each trial; realised masks/local weights
stay unchanged. The separate confirmation uses fresh offsets 600000–602000
for training and 700000–702000 for testing, never used during either selection.
It includes coupled, zero-coupling and count-preserving timing controls.

```sh
.venv/bin/python experiments/exp136/analyse.py --source exp136-r080-compute --task readout
.venv/bin/python experiments/exp136/compute.py --task receiver --selection exp136-r084-analyse
.venv/bin/python experiments/exp136/analyse.py --source <receiver-compute>
.venv/bin/python experiments/exp136/compute.py --task refine-confirm --selection <receiver-analyse>
.venv/bin/python experiments/exp136/analyse.py --source <fresh-compute>
MPLCONFIGDIR=/tmp/exp136-matplotlib .venv/bin/python experiments/exp136/present.py --source <fresh-analyse>
```
