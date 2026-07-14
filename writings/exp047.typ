#import "/.demolab/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Pool-size invariance requires inverse synaptic scaling",
  date: "2026-07-14",
  description: "Paired controls separate fixed summed I→E coupling from fixed realised synaptic strength as the inhibitory pool grows.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let r = json("/artifacts/data/exp047/numbers.json")
#let ft = r.summary.fixed_total
#let fs = r.summary.fixed_synapse
#let e(control, level, n) = control.at(level).at(n).r_e_hz_mean
#let i(control, level, n) = control.at(level).at(n).r_i_hz_mean
#let re-ft-lo = e(ft, "2", "16")
#let re-ft-hi = e(ft, "2", "256")
#let re-fs-lo = e(fs, "0.0078125", "16")
#let re-fs-hi = e(fs, "0.0078125", "256")
#let ri-fs-lo = i(fs, "0.0078125", "16")
#let ri-fs-hi = i(fs, "0.0078125", "256")
#let cfg = r.config
#let n-lo = cfg.n_i_sweep.at(0)
#let n-mid = cfg.n_i_sweep.at(1)
#let n-hi = cfg.n_i_sweep.at(2)
#let g-lo = cfg.reference_g_ie.at(0)
#let g-mid = cfg.reference_g_ie.at(1)
#let g-hi = cfg.reference_g_ie.at(2)
#let j-lo-ns = calc.round(cfg.reference_j_ie.at(0) * 1000, digits: 2)
#let j-mid-ns = calc.round(cfg.reference_j_ie.at(1) * 1000, digits: 2)
#let j-hi-ns = calc.round(cfg.reference_j_ie.at(2) * 1000, digits: 2)
#let n-seeds = cfg.seeds.len()
#let re-ft-lo-fmt = calc.round(re-ft-lo, digits: 2)
#let re-ft-hi-fmt = calc.round(re-ft-hi, digits: 2)
#let re-fs-lo-fmt = calc.round(re-fs-lo, digits: 2)
#let re-fs-hi-fmt = calc.round(re-fs-hi, digits: 2)
#let ri-fs-lo-fmt = calc.round(ri-fs-lo, digits: 2)
#let ri-fs-hi-fmt = calc.round(ri-fs-hi, digits: 2)

#let body = [
  == Abstract

  The apparent invariance of pyramidal–interneuron network gamma (PING) firing rate to inhibitory-pool size is a consequence of the simulator's fan-in normalization, not independence from interneuron count. Excitatory cells are denoted E and inhibitory cells I. The nominal I→E parameter $G_(I E)$ is divided by the presynaptic pool size, so the realised mean synaptic conductance is $j_(I E) = G_(I E) / N_I$. We vary $N_I$ under paired controls. Rates remain flat when $G_(I E)$ is fixed and $j_(I E) prop 1 / N_I$; when the realised synapse $j_(I E)$ is fixed, rates change strongly with $N_I$. Pool-size invariance therefore requires compensatory inverse scaling of individual synapses.

  == Methods

  *1. Weight convention.* For a dense I→E matrix with shape $N_I times N_E$, the simulator draws a non-negative weight with nominal mean $G_(I E)$ and then divides every entry by its presynaptic fan-in $N_I$. Thus

  $ cal(E)[W_(k j)^(I E)] approx j_(I E) = G_(I E) / N_I, quad cal(E)[sum_(k=1)^(N_I) W_(k j)^(I E)] approx G_(I E). quad (1) $

    $G_(I E)$ is consequently an expected _summed coupling_, not the conductance of one synapse. An inhibitory volley with active set $cal(A)$ gives E cell $j$ the conductance increment

    $ Delta g_j^I = sum_(k in cal(A)) W_(k j)^(I E), quad (2) $

    where:

    - $cal(E)$ denotes expectation over random weight initialization;
    - $W_(k j)^(I E)$ is the realised conductance from inhibitory cell $k$ to excitatory cell $j$;
    - $j_(I E)$ is the mean realised conductance of one I→E synapse;
    - $G_(I E)$ is the expected sum of all I→E weights arriving at one E cell;
    - $N_I$ and $N_E$ are the inhibitory and excitatory pool sizes;
    - $cal(A)$ is the set of inhibitory cells active in a volley; and
    - $Delta g_j^I$ is the resulting inhibitory-conductance increment at excitatory cell $j$.

    Equation 2 therefore depends on both the realised synapses and the number of participating inhibitory cells.

  *2. Paired controls.* We sweep $N_I in {#n-lo, #n-mid, #n-hi}$ under two conventions. In the fixed-summed-coupling arm, $G_(I E) in {#g-lo, #g-mid, #g-hi}$ μS is held fixed, forcing $j_(I E) prop 1 / N_I$. In the fixed-synapse arm, $j_(I E) in {#j-lo-ns, #j-mid-ns, #j-hi-ns}$ nS is held fixed and the nominal parameter is set to $G_(I E) = N_I j_(I E)$. The two arms coincide at the reference pool $N_I = #cfg.n_i_reference$.

  *3. Simulation.* Untrained dense PING network, $N_E = #cfg.n_e$, $Delta t = #cfg.dt_ms$ ms, $T = #cfg.t_ms$ ms, #cfg.n_batch independent Poisson-input trials per network and #n-seeds network/input seeds per condition. E→I summed coupling is fixed at #cfg.g_ei_total μS; input comprises #cfg.n_in independent #cfg.input_rate_hz Hz Poisson channels. Markers show seed means and error bars ±1 standard deviation (SD).

  == Results

  Fixed summed coupling produces overlapping rate curves. At $G_(I E) = #g-mid$ μS, the E rate is #re-ft-lo-fmt Hz for $N_I = #n-lo$ and #re-ft-hi-fmt Hz for $N_I = #n-hi$. This is the expected consequence of holding the summed I→E coupling fixed while shrinking each realised synapse as $1 / N_I$.

  Fixed realised synaptic strength gives the opposite result. At $j_(I E) = #j-mid-ns$ nS, increasing $N_I$ from #n-lo to #n-hi changes the E rate from #re-fs-lo-fmt to #re-fs-hi-fmt Hz and the I rate from #ri-fs-lo-fmt to #ri-fs-hi-fmt Hz. Every tested synaptic strength shows the same pool-size dependence. The direction is mechanistically consistent: more inhibitory cells at unchanged individual strength produce greater population-level inhibition, while the recurrent feedback reduces both E activity and the I activity it recruits.

  #figure(
    image(
      "/artifacts/data/exp047/pool_size_controls.svg",
      width: 100%,
      alt: "Four panels comparing E and I firing rates across inhibitory pool sizes. Rates are flat when summed coupling is fixed, but fall strongly with pool size when realised synaptic strength is fixed.",
    ),
    caption: [*Excitatory and inhibitory firing rates across inhibitory-pool size under two I→E scaling controls.* Top: excitatory-cell rate in Hz per cell. Bottom: inhibitory-cell rate in Hz per cell. *(a)* Holding the expected summed I→E coupling $G_(I E)$ fixed makes the realised synapse $j_(I E) = G_(I E) / N_I$ shrink as the pool grows. *(b)* Holding $j_(I E)$ fixed makes summed coupling grow with $N_I$. Each curve is one coupling level; markers are means and error bars ±1 standard deviation (SD) over #n-seeds seeds, with #cfg.n_batch trials per seed. The controls coincide at the reference pool $N_I = #cfg.n_i_reference$. Rates are invariant to pool size only under inverse scaling of individual synapses.],
  )

  == Conclusion

  The network is not intrinsically insensitive to inhibitory-pool size. It is insensitive only along a specific normalization path: individual I→E synapses must scale approximately as $1 / N_I$ so that expected summed coupling stays fixed. The operative control variable in this dense model is population-level inhibitory drive, jointly determined by pool size, realised synaptic strength, and volley participation. This experiment does not establish that the same inverse scaling holds biologically; it identifies the compensation required for invariance in the model.
]
