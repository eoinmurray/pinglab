# Introduction to spiking neural networks

Educational article for machine learning practitioners and software developers.
The first PixiJS demonstration simulates a pulse-input leaky integrate-and-fire
neuron locally in the browser. No retained data or staged execution is required.
The second demonstration combines three independent Poisson presynaptic spike trains
through adjustable weights into the same neuron model. A third demonstration
replays an actually trained SNNLab classifier before and after training.

Model: rest/reset −65 mV, threshold −50 mV, membrane capacitance 200 pF,
0.25 ms rectangular current pulses and simulation steps, no refractory period.
Subthreshold voltage uses the analytic constant-current LIF solution each step;
firing is checked at step boundaries before a hard reset.
Automatic inputs are emitted every 12 ms and arrive after a 6 ms illustrative
transmission delay. Each pulse captures the strength selected when it is emitted.
Playback advances 40 simulated ms per real second; travel and firing highlights
are diagrammatic. The panel shows input pulses, a neuron voltage gauge and output spikes on the
left, with a rolling 120 ms voltage plot on the right. The trace caps candidate
voltage at threshold before immediate reset; it does not depict an action
potential waveform. The full-width control bar contains input-strength and
leak-time sliders and a pause button.
Reduced-motion users start paused.

The synapse demonstration uses nonnegative current weights in nA (defaults 0.8, 3.2, 8).
Sources are independent homogeneous Poisson processes at 40, 30 and 20 Hz.
Each stream samples continuous exponential interarrival times with mean 1000/rate
milliseconds. All events are retained, including multiple spikes per integration
step; arrivals are placed on the next 0.25 ms integration boundary.
Seeds are 0x9e3779b9, 0x243f6a88 and 0xb7e15162; reopening reproduces the streams.
There is no common starting event. All spikes arrive after 6 ms.
Arrival counts are evaluated at 0.25 ms steps, and their
weighted sum is applied as constant input current for that simulation step. Weights are evaluated on arrival;
changing a slider affects spikes still in flight. Zero weights preserve the
prescribed source spikes but remove
their postsynaptic contribution. Source flashes and pulse motion are schematic.
Rate labels sit to the left of source neurons, matching Figure 3; the highest
rate is blue and bold. Weight symbols sit beside connections,
away from source labels and lines; numeric weights are shown in the controls.
The postsynaptic time constant is fixed at 30 ms. Current pulses last one step;
there is no further synaptic filtering. The first panel defaults to 5.6 nA
per pulse. Slider ranges were rescaled to retain similar voltage excursions
and clearly different synapse thicknesses.
Both captions give rest/reset equality. The voltage axis spans −65 to
−45 mV; the −50 mV firing threshold is marked within it. Voltage traces
retain their full scale through threshold; changing axis ticks does not change
the model or clip subthreshold voltage.

The panel matches exp099's background, border, radius, padding and responsive
figure grid. Source: `report/report.mdx`, `report/NeuronAnimation.astro`, and
`js/neuron.js`. Preview at `/exp127` with the site's Astro development server.

## Learning classifier

`recipe.py` defines a native SNNLab language graph with three spike inputs, two
CUBA LIF output neurons, six nonnegative synaptic weights, 2 ms exponential
current synapses and a 1 ms delay. `snnlab.sim.execution.train` performs AdamW
training with fast-sigmoid surrogate gradients; `simulate` records both
checkpoints on the same held-out input. There is no external classification head.
The sum of each output neuron's spikes over 200 ms is its class score. Runtime
weight tensors are oriented input × output; graph parameter declarations use
output × input. Replay thickness uses the runtime tensors and one common scale.

Training/validation/test contain 128/64/128 independent synthetic presentations.
Class A rate pattern is 125/12.5/31.25 Hz; B is 12.5/125/31.25 Hz. Each source rate varies
independently by ±15% per presentation. Poisson counts are clipped to binary
occupancy in 1 ms bins to satisfy the SNNLab spike-input contract; raw counts
and the binary input arrays are both retained. Neuronal state resets for each
presentation. One seed is used; this is an educational task, not a benchmark.

All weights initially share the same initializer. SNNLab's fan-in normalization
makes the effective initial per-edge weight 0.35 nA. Both output neurons spike
before training, with equal counts and no discrimination.
Twenty-four epochs are evaluated on fixed validation inputs, including epoch 0;
the earliest minimum-validation-cross-entropy checkpoint is selected. Replay
examples are the first test item of each class, chosen independently of outputs.
Recorded accuracy covers all 128 test examples; latency and activity describe
the selected replay example. These metrics are retained in the exports; the figure
shows accumulated counts and predictions. Latency is the first recorded time with a winning
count ≥5 and a lead ≥3; an unmet rule is null, not 200 ms or zero. The first
decision can be wrong or later change. Activity counts only output spikes over
the full 200 ms. Count ties in final predictions choose class A. Playback never
interpolates weights or simulates undocumented intermediate networks.

The current retained execution chain is `exp127-r012-compute` → `exp127-r013-analyse` →
`exp127-r014-present`. The article explicitly pins the presentation. Run stages
individually from the repository root:

```sh
.venv/bin/python experiments/exp127/compute.py
.venv/bin/python experiments/exp127/analyse.py --source <compute-run-id>
.venv/bin/python experiments/exp127/present.py --source <analyse-run-id>
```

The first attempt rejected nonbinary Poisson counts at SNNLab's input boundary;
its temporary run contains no completed evidence and was removed after the
successful successor. The successful run retains raw counts and uses the
required binary occupancy explicitly.

The original retained r002 → r003 → r004 chain used weaker initial weights and
a silent baseline. It is preserved; the revised r005 → r006 → r007 chain gives
a visibly spiking untrained baseline without changing the data or replay items.
The revised selected checkpoint is epoch 17; test accuracy remains 50% → 100%.

The r008 presentation adds the original recorded output-voltage arrays to the
classifier replay. Larger output neurons carry gauges on the shared −65 to
−50 mV scale, with threshold ticks and reset values taken from the saved graph.
Poisson input channels have no membrane-voltage state and remain spike sources.
No training or simulation was rerun for this display change.

The r009 → r010 → r011 chain increases classifier input rates by 25%, with a
new native SNNLab training and recording run. The selected checkpoint is epoch
24; held-out accuracy remains 50% → 100%. Larger, left-aligned rate labels sit
to the left of the input neurons; the largest rate is highlighted in blue and
bold. Switching examples updates the highlight.

## Training replay

Figure 3 records a separate teaching run through the same SNNLab language graph
and native simulation executor, using a standard PyTorch AdamW loop. It reuses
all 128 training inputs as one fixed batch for 48 updates (not 48 epochs of the
minibatch run). Each forward call resets neuronal state. Cross-entropy uses
native output spike counts; backward uses the declared fast-sigmoid surrogate,
gradients are clipped at 5, and weights retain the nonnegative constraint.
`training_trace.npz` stores all losses, weights, unclipped gradients, and output
spikes/voltages for the first training item of each class. Analyse validates the
recording and selects eight consecutive updates with the largest endpoint loss
decrease, earliest on ties. Present exports `training.json` with the full loss
curve and that interval. Gradients and loss concern the whole batch; the animated
network shows one representative example. Backward dots and interpolation between
measured weight values are explicitly illustrative. The complete loop stages are
played sequentially; the full loss graph remains visible. The inference panel is
Figure 4 and still uses the independently validation-selected minibatch checkpoint.
