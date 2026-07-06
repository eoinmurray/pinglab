#let meta = (
  title: "PING fundamentals",
  date: "2026-05-13",
  description: "PING stripped to its biophysical fundamentals and characterised in isolation from any task: the E→I→E loop produces gamma at ≈ 30 Hz and compresses the E dynamic range an order of magnitude below COBA.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let body = [
  == Abstract

  Strips PING down to its biophysical fundamentals and characterises it _in isolation from any task_ — no training, no readout, no loss, just a free-running network driven by Poisson input. The E → I → E loop produces gamma at ≈ 30 Hz with the standard $tau_"AMPA" + tau_"GABA"$ cycle period, and the f–I curves show the loop is dynamic-range compression: PING holds E an order of magnitude below COBA across two orders of magnitude of drive, while COBA saturates near its refractory ceiling.

  == Method

  The architecture characterised here is sketched atop Figure 1. An input layer drives the excitatory population through $W_"in"$; the E→I→E loop is formed by $W_"ei"$ (E to I) and $W_"ie"$ (I back to E), with no I→I synapse. Disabling the I→E loop (_ei-strength 0_) recovers the COBA reference; engaging it (_ei-strength 1.5_) produces the gamma rhythm.

  The full conductance-based model that produces PING is documented in #link("/ar003/")[COBANet]; this notebook is the empirical companion. Two conditions run on the same network — _ei-strength 0_ (loop disabled, the COBA reference) and _ei-strength 1.5_ (loop engaged, the PING regime) — each driven by uniform Poisson input fed through $W_"in"$ (untrained network). Because PING's loop clamps the E rate while COBA's is unclamped, a single input rate can't show both legibly: the two rasters therefore use *per-condition input rates* — COBA at 5 Hz (a legible ≈ 24 Hz async raster), PING at 45 Hz (into gamma, $f_gamma approx 40$ Hz). Population spectra are Welch PSDs on the population-mean E trace (one window per trial of length $T$, parabolic-interpolated peak; same recipe as #link("/exp041/")[exp041] and #link("/exp049/")[exp049]).

  The run scale reports the geometry that actually produced the figures. The two loop conditions (COBA, _ei-strength 0_; PING, _ei-strength 1.5_) are the grid; the raster/PSD input rates are per-condition (COBA 5 Hz, PING 45 Hz) and the free-running f–I panels sweep uniform Poisson drive (2–100 Hz, $T = 400$ ms).

  *Rate response.* The f–I panels of Figure 1 sweep spatially *uniform* Poisson input (all channels at the same rate, no digit structure) on the _untrained, free-running_ network, comparing COBA against PING on a shared rate axis — the bare dynamic-range compression of the architecture.

  == Results

  #figure(
    image("/artifacts/data/exp023/overview_compound.png", width: 100%),
    caption: [
      The architecture and its behaviour, off and on, on the same network. Each column carries its schematic on top — *A* COBA (input→E→output, no I population) and *B* PING (the same plus the recurrent E↔I loop, $W_"ei"$ down and $W_"ie"$ up) — directly above the plots it produces. Below each schematic is one combined raster (E black in the lower portion, I red in the upper portion — same axes, no gap) above a bottom row pairing the Welch PSD of the population-mean E trace with the free-running f–I curve (mean per-cell rate vs input rate, E black / I red; each column has its own y-axis — COBA runs to its ≈ 400+ Hz ceiling, PING is on a smaller scale so its loop-clamped E and climbing I are legible). Rasters and spectra are on uniform Poisson input (400 ms) at per-condition rates (COBA 5 Hz, PING 45 Hz — see Methods); the f–I curves sweep that drive on the untrained network. All runs are at $Delta t = 0.1$ ms. *A — COBA* (_ei-strength 0_, loop off): asynchronous E firing, no I activity, no recurrent gamma peak, and an E f–I curve that climbs steeply with input rate and saturates toward the refractory ceiling. *B — PING* (_ei-strength 1.5_, loop on): the I population fires a synchronous burst once per cycle and the E raster breaks into vertical bands (the I volley trails each E volley by the AMPA delay); the PSD carries a clean gamma peak (dashed red line; $f_gamma$ measured from this raster's E-population PSD, marked on the plot); and its f–I shows the E rate held to single digits across the whole sweep (note the much smaller y-axis than COBA's) while inhibition climbs — the loop's dynamic-range compression. The single E→I→E loop is the only change: it converts asynchronous firing into a gamma rhythm _and_ clamps the E rate.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__coba__v_e.svg", width: 100%),
    caption: [
      Membrane voltage $V_E$ (black) of the most active E cell, recurrent loop *off* (COBA, _ei-strength 0_). Driven only by feedforward input, the cell charges toward threshold ($V_"th" = -50$ mV, faint dashed) and hard-resets to $V_"reset" = -65$ mV each time it fires — irregular, input-paced spiking with no rhythmic structure. The rhythm-free control for Figure 3a.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__coba__g_e.svg", width: 100%),
    caption: [
      Synaptic and leak conductances on the same E cell, COBA mode. Only the excitatory conductance $g_E$ (black) is active — it steps up with each input spike and decays with $tau_"AMPA"$ — while the leak $g_L$ (faint dotted) is fixed. There is no inhibitory $g_I$ because the I→E loop is off. All conductances are *non-negative*: they count open channels, not current direction or sign.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__coba__i_e.svg", width: 100%),
    caption: [
      Signed synaptic currents into the E cell, COBA mode, with $I_X^"in" = -g_X (V - E_X)$ (positive = depolarising). With no inhibition, the only synaptic current is the depolarising excitatory current $I_E^"in"$ (black); the leak current (faint) hovers near zero. Compare Figure 3c, where the loop introduces an inhibitory current of the _opposite_ sign.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__ping__v_e.svg", width: 100%),
    caption: [
      Membrane voltage $V_E$ (black) of the most active E cell, recurrent loop *on* (PING, _ei-strength 1.5_), same input as 2a. Rhythmic inhibitory bursts now hold the cell below threshold most of the time: each I-burst drags $V_E$ toward $E_i = -80$ mV, and the membrane recovers — and can fire — only as the inhibition decays between bursts. The loop has turned the irregular firing of 2a into cycle-gated firing.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__ping__g_e.svg", width: 100%),
    caption: [
      Conductances on the same E cell, PING mode. The inhibitory conductance $g_I$ (red) now dominates — it spikes once per gamma cycle as the synchronous I-burst arrives through $W_"ie"$, then decays with $tau_"GABA"$ — while $g_E$ (black) carries the smaller feedforward input and $g_L$ (faint) is fixed. All three traces are *non-negative*: $g_I$ is large and positive, and that is what shunts the cell — it carries no sign of its own.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__ping__i_e.svg", width: 100%),
    caption: [
      The load-bearing panel. Signed currents into the E cell, PING mode. The excitatory current $I_E^"in" = -g_E (V - E_E)$ (black) is depolarising; the inhibitory current $I_I^"in" = -g_I (V - E_I)$ (red) is *negative (hyperpolarising) even though $g_I$ is positive* (Figure 3b) — the minus sign comes entirely from the driving force $(V - E_I) > 0$ whenever $V$ sits above $E_i = -80$ mV. The COBA principle made literal: _the sign lives in the driving force, never in the conductance._ The sharp negative pulses are the per-cycle inhibitory kicks that gate the rhythm.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__ping__v_i.svg", width: 100%),
    caption: [
      Membrane voltage $V_I$ (red) of the most active I cell, PING mode. The I cell integrates the excitatory volley from the E population and fires once per gamma cycle, so $V_I$ tracks the E-burst envelope — ramping up as E cells fire, crossing threshold ($V_"th" = -50$ mV, faint dashed), resetting, and waiting for the next volley. This single I-spike per cycle delivers the inhibitory burst seen on the E cell in 3b–3c.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__ping__g_i.svg", width: 100%),
    caption: [
      Conductances on the I cell, PING mode. The I cell receives *only excitation*: $g_E$ (black) is the arriving E-spike envelope (delivered via $W_"ei"$) and $g_L$ (faint) is the fixed leak. There is no inhibitory conductance — the architecture has no I→I synapse — which is exactly why the I population can synchronise into the sharp once-per-cycle bursts that clock the loop.
    ],
  )

  #figure(
    image("/artifacts/data/exp023/traces__ping__i_i.svg", width: 100%),
    caption: [
      Signed currents into the I cell, PING mode. With no inhibitory input, only the depolarising excitatory current $I_E^"in"$ (black) and a small leak current contribute — the I cell is driven purely up to threshold. Read 3a–3f together and the loop is in cross-section: E spikes ramp $g_E$ on I (3e–3f) → I fires once per cycle (3d) → that spike delivers the inhibitory burst $g_I$ on E (3b) → its negative current shuts E down (3c, 3a) → $g_I$ decays → E refires. That delayed E→I→E loop is PING.
    ],
  )
]
