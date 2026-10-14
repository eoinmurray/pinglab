# Introduction to Conductance based Synapses

Educational article with a live PixiJS exponential-conductance synapse connected
to a conductance-based leaky integrate-and-fire neuron. No external data or staged execution is required. Sources:
`report/report.mdx`, `report/SynapseAnimation.astro`, `report/references.mdx`,
and `js/synapse.js`. Preview at `/exp128/`.

The panel uses exp127's figure grid, white background, 1 px #d6dfe7 border,
12 px padding and radius. The scene uses the same 1240 × 360 logical canvas
on desktop and 620 × 360 on mobile, with labels scaled in the same way.
A, B and C each occupy one third of the scene width. Controls retain a single
full-width row with exp127's separator and padding. A shows spikes travelling directly to one schematic channel embedded in the
receiving neuron's membrane. The symbol encodes aggregate synaptic conductance;
the animation omits the presynaptic terminal and transmitter dynamics. B shows conductance on a fixed 0–14 nS axis. C shows
neuron voltage on a fixed −80 to −20 mV axis, with rest at −65 mV. Both use a
rolling 120 ms window. Weight, decay time and reversal are the only sliders;
pause is the only button. Every slider change immediately simulates a fresh
120 ms preview, preserving playback choice. Weight no longer rescales the
conductance axis, so amplitude changes remain visible while paused or playing.

Model: C = 100 pF, leak = 10 nS, rest = −65 mV, initial g = 0 and V = rest.
Spike weight defaults to 8 nS and decay to 12 ms. Reversal defaults to 0 mV.
Spikes are emitted at 6 + 30j ms and arrive 8 ms later. With 0.25 ms steps,
conductance decays exactly; voltage uses the exact frozen-conductance passive
solution evaluated with midpoint conductance. Each arrival adds its increment
at the end of its step. Playback advances 30 simulated ms per real second.
Threshold is −50 mV and reset is −65 mV, with no refractory interval.
Trace peaks are capped at threshold before reset; they are not action-potential
waveforms. Rose pulses and a counter show output spikes. There is no channel
saturation or voltage-dependent receptor gating. Channel opening width encodes aggregate conductance on a fixed
scale rather than physical channel size. Reduced-motion users start paused;
offscreen and hidden panels stop their tickers.

The third section adds a dual-input demonstration using the same component.
Two prescribed presynaptic neurons, E and I, feed separate membrane channels,
with blue E and rose I pulses and conductance traces. Both reversal potentials
are adjustable: E spans −80 to 0 mV and I spans −90 to 0 mV. Both weights span 0–8 nS
and default to 8 nS for E and 4 nS for I. E uses a 12 ms decay, 0 mV reversal and arrivals at
14 + 30j ms; I uses a 20 ms decay, −75 mV reversal and arrivals at 24 + 40j ms.
The subthreshold voltage update includes both midpoint conductances. The sources
have prescribed regular spike times, without integrated voltage dynamics.
Sliders recompute the preview, including while paused; zero disables the
corresponding conductance. Equation (4) defines threshold/reset, equation (5) defines both synaptic states
and equation (6) their summed effect on the same receiving voltage.

At the defaults, the initial 120 ms previews produce five output spikes in
Figure 1 and four in Figure 2. Reversal controls affect current and spike
generation but do not affect conductance kinetics. All controls recompute
the preview immediately while preserving playback choice.

Figure 2 uses a fixed −70 to −45 mV voltage axis with ticks every 5 mV; Figure 1 retains its
−80 to −20 mV axis. Threshold, rest, ticks and traces share each axis mapping.
