# A/B → C attention circuit

Illustrative Bosman-style convergence, generated from a real `snnlab.lang`
bundle. Run from the repository root:

```sh
.venv/bin/python notes/figures/ctc-attention/draw.py
```

1. A and B each contain 128 excitatory and 32 inhibitory COBA-LIF neurons,
   with separate sensory and attention spike-input ports. C has the same local
   E/I circuit and receives equal excitatory projections from A and B into both
   its E and I populations.
2. E→I and I→E recur in each local circuit. Optional local E→E and I→I
   projections are included, alongside optional C_E→A_E/A_I and C_E→B_E/B_I
   feedback. The complete illustrative graph has 26 projections.
3. Attention ports model an additional excitatory bias to A or B; this is an
   illustrative implementation of upstream control, not a measured Bosman pathway.
4. Black/grey arrowheads denote excitation; red bars denote inhibition. Inter-area
   projections are darker and heavier than local recurrence; this is visual
   hierarchy, not a synaptic-strength scale. Solid paths
   are kept and dashed paths are optional. Matrix labels
   use target first, source second. Node subscripts identify the circuit;
   X denotes sensory input and U denotes attention input in matrix labels.
5. Membrane and synaptic constants follow the repository's COBA/PING reference.
   Population sizes, input widths and projection strengths are illustrative.
   No simulation, gamma calibration or attention effect has been established.
6. A_E/B_E→C_I together with C_I→C_E implements feedforward inhibition.
   Direct afferents to C_I are a modelling choice motivated by Bosman's proposed
   mechanism, not identified synapses measured in that experiment.
7. Direct A–B links, inter-area projections from I cells, theta-reset input and
   separate laminar populations are omitted. Feedback uses the existing C_E
   population as a simplification of Bosman's deep-layer feedback interpretation.
   Disable optional feedback when constructing the directional A→C baseline.
8. Background C supplies a separate 64-channel spike-input port to both C_E and
   C_I, with illustrative incoming strength 0.1 µS per pathway. This port is
   intended for task-independent background activity: it carries no digit or
   cue, and its statistics stay fixed across cue conditions. Its independence,
   strength and targets are modelling choices; no spike generator or rate is
   fixed by this diagram.

The script saves `network.bundle/`, `network.svg` and `network.png`.
The SVG and PNG use a 1920×1080 (16:9) canvas, with horizontal local E/I pairs.
The graph, topology and neuron dynamics are inspectable in the native bundle.
Input and population cards share the same size, fill, borders and mathematical
label style. Inputs show X_A/X_B (sensory), X_C (background) and U (attention);
populations show E/I symbols. Type and neuron-count text are hidden. Optional feedback is drawn in a separate inset,
which repeats the same populations without adding neurons or changing the bundle.
All 26 native projections appear once across the main circuit and inset. Local
paths use 45% colour composited on white and a thinner stroke; inter-area paths
use full contrast and a heavier stroke, including optional feedback. The note
embeds the SVG with the shared figure lightbox. All labels use DejaVu Sans,
including mathematical symbols; SVG glyphs are embedded as vector outlines so
browser font substitution cannot change the typography.

Reference: Bosman et al. (2012), *Attentional Stimulus Selection through
Selective Synchronization between Monkey Visual Areas*.
[doi:10.1016/j.neuron.2012.06.037](https://doi.org/10.1016/j.neuron.2012.06.037).
