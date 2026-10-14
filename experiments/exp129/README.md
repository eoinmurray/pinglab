# Introduction to Capacitors

Educational article with a live PixiJS ideal resistor–capacitor circuit. No
external data or staged execution is required. Sources: `report/report.mdx`,
`report/CapacitorAnimation.astro`, `report/references.mdx` and `js/capacitor.js`.
Preview at `/exp129/`.

Matches exp127/exp128: white panel, 1 px #d6dfe7 border, 12 px padding and
radius, blue/rose palette, three labelled panels and one row of sliders and controls (scrollable on narrow screens).
The logical canvas is 1240 × 360 on desktop and 620 × 360 on mobile.
A occupies the left two-thirds and shows two taller plates and larger, plain blue electron dots, the external circuit and schematic field
arrows. B (voltage, 0–6 V) is stacked above C (conventional current, −0.6 to
0.6 mA) in the remaining third. Both use a rolling 10 s window. There is no time-constant readout or plus/minus marker on the electron dots.

Model uses SI units and the analytic exponential voltage update. Charge targets
the battery voltage, Hold retains voltage with zero current, and Discharge
targets zero through the same resistor. Mode changes retain voltage and add
both sides of the current discontinuity to the plot. Sliders and Restart reset
to uncharged Charge mode and preserve playback choice. Defaults: C = 10 μF,
R = 50 kΩ, battery = 5 V, τ = 0.5 s. Ranges: 2–20 μF, 10–100 kΩ, 1–6 V.
Playback advances one simulated second per real second, with analytic substeps
no longer than 10 ms. Long frame gaps are capped at 100 ms.

At 120 μC there are 12 missing/excess dots per plate, a fixed 10 μC per dot
scale across controls. Each plate starts half filled: 16 electron representatives
occupy the left eight columns of a visible 2 × 16 cell grid. Charging removes
occupied cells from the right edge of that filled half above and fills adjacent
empty cells below. Positions are cell centres, shared by settled and travelling
dots. Positive ionic backgrounds are implicit. A representative transfer launches
when rounded model charge crosses a dot boundary. The departing dot starts at
its occupied cell; arrivals travel through the external wires to their target
cell and become a settled dot after 0.4 simulated seconds. Charging animates
both battery-connected branches; discharge carries a dot from its lower cell,
through the external resistor loop, into its vacant upper cell. Multiple
transfers may be in flight. These are illustrative transit times, so settled
dot counts can lag the continuous model charge. Pause freezes transfers. Mode
changes settle them to the rounded charge immediately; slider changes reset
the grid. Grid slots stay fixed relative to each plate; opacity and plate fill
remain fixed. No dot crosses the insulating gap. The diagram does not depict
battery chemistry, microscopic trajectories or physical transit times.
Field-arrow opacity scales with voltage / 6 V. The diagram does
not solve spatial electrostatics, dielectric polarisation, leakage or breakdown.

Visual controls: plate gap is 160 / C_μF logical pixels (80 at 2 μF, 16 at
10 μF, 8 at 20 μF), with fixed plate width and thickness. This encodes the
ideal parallel-plate inverse spacing relationship at fixed area and permittivity.
The assembly is centred vertically at y = 200; wires, grid rows, field arrows
and charge labels track the moving plates. Resistor width spans 24–96 logical
pixels as R increases from 10 to 100 kΩ. Battery symbol scale spans 0.65–1.35
as V_b increases from 1 to 6 V; its wire endpoints track the scaled terminals.
Resistor and battery size are schematic parameter encodings, not claims about
physical component dimensions.

Reduced-motion users start paused. Offscreen and hidden panels stop their
tickers. Intersection observers, resize observers and event listeners are
disposed when the component disconnects. The article applies Writing Guide
36.0.0, declares no retained data inputs and carries `txt`.

## Biophysical neuron animation

`report/NeuronMembraneAnimation.astro` and `js/membrane.js` add a membrane patch,
without circuit symbols. A prescribed presynaptic spike travels to an amber synaptic channel. Blue Na⁺
and rose K⁺ ions move through separate voltage-gated pores; a stylised lipid bilayer separates fixed outside/inside reservoirs. Other
ions, leak pathways and pumps are omitted from the drawing. Reservoir dot ratios
are illustrative; concentrations and reversal potentials remain fixed in the
model. No displayed bulk pool drains or determines membrane charge.

The classical squid-axon Hodgkin–Huxley model uses 6.3 °C kinetics, C_m =
1 μF/cm², maximum sodium/potassium/leak conductance densities 120/36/0.3 mS/cm²,
and reversal potentials 50/−77/−54.3 mV. Rate equations follow the absolute
inside-minus-outside voltage convention in NEURON's maintained
[hh model](https://github.com/neuronsimulator/nrn/blob/master/src/nrnoc/hh.mod).
Na and K availability scale their maximum conductances independently from 0–100%.
Capacitance ranges 0.5–2 μF/cm². Synaptic weights range 0–0.6 mS/cm², default
0.2. Presynaptic spikes are emitted at 1 + 40j ms and arrive at 5 + 40j ms,
with 4 ms schematic travel. Each arrival adds the weight to an exponentially
decaying synaptic conductance (τ = 2 ms, E_syn = 0 mV). RK4 evaluates the
voltage-dependent synaptic current and exact exponential conductance decay
at its start, midpoint and endpoint stages. There is no prescribed current
pulse. The added synapse is an illustrative coupling to the classical membrane,
not part of the original squid-axon parameter fit. No threshold/reset rule
is applied: spikes emerge from voltage-dependent activation/inactivation.

RK4 integrates voltage and m/h/n at 0.01 ms, with removable rate singularities
handled analytically. A 200 ms unstimulated prelude precedes each fresh preview;
a paused view calculates a complete 60 ms preview, while a playing view
starts at t = 0 so the first arriving spike and cascade can be watched. Restart
always rewinds to t = 0. Sliders preserve playback choice. This prelude is not a claim of equilibrium for
altered channel settings. Plot samples are retained every 0.05 ms. B uses a
fixed −85 to 50 mV range. C shows inward-positive synaptic (amber), Na (blue) and K (rose) current densities
on a fixed −1000 to 1000 μA/cm² range; model equations use outward-positive
ionic currents. Both use a rolling 60 ms window. Playback advances 5 simulated
ms per real second. Reduced-motion users start paused; hidden/offscreen panels
stop and all observers/listeners are cleaned up on disconnect.

Membrane thickness is 28 / C_m logical pixels at fixed area, a schematic
inverse-capacitance cue. Channel-pore width represents combined conductance,
not physical channel dimensions. A representative ion is emitted per 25 nC/cm²
of absolute integrated Na or K flux (μA/cm² × ms = nC/cm²). Amber markers
represent net synaptic charge flow, with one emitted per 2 nC/cm²; they do not
identify a single ionic species. Faint amber lines mark arrivals in both plots. Travel lasts 1
simulated ms and crosses only the matching pore; direction follows current sign.
These dots are flux markers rather than tracked molecular trajectories; arrival
positions and reservoir counts do not feed back into the model.

Numerical checks are repeated after the synaptic input replaces the current
pulse. They cover default spiking, weak and zero weights, Na/K blockade,
pre-arrival causality, synaptic decay, timestep convergence and endpoint control
combinations. These are illustrative model checks, not new experimental
observations or retained data runs.

The synaptically driven default preview produces two spikes in 60 ms, maximum
39.41630 mV and minimum −76.03089 mV. Halving dt changes the sampled peak by
less than 0.002 mV and retains both spikes. Weight 0.05 mS/cm² gives only a
small response (maximum −62.11834 mV); zero weight leaves the model at its
unstimulated resting voltage. Na blockade prevents the regenerative spike.
All 16 endpoint control combinations keep voltage finite and gates within [0,1];
the largest observed Na/K current magnitude is below the fixed 1000 μA/cm²
axis limit. Before the first arrival, synaptic conductance is exactly zero and
the default membrane stays at rest. Synaptic current reverses at its fixed
0 mV reversal potential.
