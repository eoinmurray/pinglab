#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp047")

#let meta = (
  status: "[▦ DATA | v28.0.0]",
  title: "Pool-Size Effects Depend on Synaptic Scaling",
  created_at: "2026-07-14T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "Paired controls separate fixed summed I→E coupling from fixed expected synaptic strength as the inhibitory pool grows.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp047",)
#let preview-figures = (
  (path: "exp047/pool_size_controls.svg", label: "pool size controls"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp047/numbers.json"))
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
#let j-mid-ns = calc.round(cfg.reference_j_ie.at(1) * 1000, digits: 2)
#let j-ns = cfg.reference_j_ie.map(x => x * 1000)
#let n-seeds = cfg.seeds.len()
#let re-ft-lo-fmt = calc.round(re-ft-lo, digits: 2)
#let re-ft-hi-fmt = calc.round(re-ft-hi, digits: 2)
#let re-fs-lo-fmt = calc.round(re-fs-lo, digits: 2)
#let re-fs-hi-fmt = calc.round(re-fs-hi, digits: 2)
#let ri-fs-lo-fmt = calc.round(ri-fs-lo, digits: 2)
#let ri-fs-hi-fmt = calc.round(ri-fs-hi, digits: 2)

#let body = [
  == Abstract

  Asked whether inhibitory population size alters PING activity by adding
  inhibition or merely redistributing a fixed inhibitory budget. Compared
  pool-size sweeps that held either summed inhibition or individual synaptic
  strength fixed.

  Firing rates stayed stable under fixed total inhibition but fell as the pool
  grew when individual synaptic strength was preserved. Supports inverse scaling
  as a compensation rule for this regime; it does not establish uniqueness or
  demonstrate gamma rhythmicity.

  == Results

  #with-result-sections[

  === Pool-size dependence changes with synaptic scaling

  At nominal $G_(I arrow E) = #g-mid$ μS, E rates changed from #re-ft-lo-fmt
  to #re-ft-hi-fmt Hz across #n-lo–#n-hi I neurons. At
  $macron(w)_(I arrow E) approx #j-mid-ns$ nS, E rates fell from #re-fs-lo-fmt
  to #re-fs-hi-fmt Hz and I rates from #ri-fs-lo-fmt to #ri-fs-hi-fmt Hz. All
  tested fixed-mean levels showed this decrease, consistent with stronger summed
  inhibition and reduced E–I feedback.

  #figure(
    data-image(data-file("exp047/pool_size_controls.svg"),
      width: 100%,
      alt: "Four panels of E and I firing rates for inhibitory pools of 16, 64 and 256 neurons. Fixed summed coupling gives nearly flat rates; fixed expected synaptic strength gives falling rates.",
    ),
    caption: [*Reanalysed population firing rates.* Top: excitatory (E); bottom:
      inhibitory (I). Left: fixed expected summed coupling $G_(I arrow E)$; right:
      fixed expected synaptic strength $macron(w)_(I arrow E)$. Markers are means ±1 sample
      standard deviation across #n-seeds seeds, with #cfg.n_batch trials per seed.
      Shared conditions reused the same simulations.],
  )

  ]

  == Methods

  We compared two scaling controls using recorded outputs from simulations of untrained,
  dense recurrent excitatory–inhibitory networks, without additional simulation.
  #set math.equation(numbering: "(1)")

  + *Initialize fan-in-normalized weights.* For an I→E matrix with
    #cfg.n_e excitatory columns and $N_I$ inhibitory rows, each weight was
    $W_(k j)^(I E) = G_"draw" / N_I$, with $G_"draw" = max(0, X)$ and
    $X tilde cal(N)(mu_"init", sigma_"init"^2)$ a Gaussian draw of mean $mu_"init"$ and standard
    deviation $sigma_"init" = 0.1 mu_"init"$, in μS. Defining expected summed coupling
    $G_(I arrow E) = cal(E)[G_"draw"]$ gives

    $ cal(E)[W_(k j)^(I E)] = macron(w)_(I arrow E) = G_(I arrow E) / N_I, quad
      cal(E)[sum_(k=1)^(N_I) W_(k j)^(I E)] = G_(I arrow E). $

    Here $cal(E)$ averages over weight initialization, $k$ indexes inhibitory
    neurons and $j$ excitatory neurons; $macron(w)_(I arrow E)$ is the expected conductance of
    one synapse, not an identical realised weight. An inhibitory volley gives

    $ Delta g_("inh,post=E",j) = sum_(k in cal(I)_"active") W_(k j)^(I E), $

    where $cal(I)_"active"$ is the active inhibitory set and $Delta g_("inh,post=E",j)$ the
    inhibitory conductance increment at E neuron $j$, in μS; both weights and participation
    therefore enter the increment.

  + *Apply paired pool-size controls.* We swept
    $N_I in {#n-lo, #n-mid, #n-hi}$. Fixed-summed controls used parent means
    $mu_"init" in {#g-lo, #g-mid, #g-hi}$ μS, so $macron(w)_(I arrow E) prop 1 / N_I$;
    fixed-mean-synapse controls scaled $mu_"init"$ with $N_I$, giving nominal
    $macron(w)_(I arrow E) in {#j-ns.at(0), #j-ns.at(1), #j-ns.at(2)}$ nS.
    Nominal values approximate the post-clamp expectations; the arms coincide
    at #cfg.n_i_reference I neurons, with one additional shared condition at
    #n-mid I neurons and nominal 1 μS summed coupling.

  + *Drive and measure the networks.* Nominal E→I summed coupling stayed at
    #cfg.g_ei_total μS; input weights used parent mean #r.recipe.w_in_mean μS,
    #calc.round(r.recipe.w_in_initial_zero_fraction * 100)% initial zeros,
    compensation for zeroing and fan-in normalization. Each network received
    #cfg.n_in independent #cfg.input_rate_hz Hz input channels, implemented
    as Bernoulli spikes per #cfg.dt_ms ms timestep, for #cfg.t_ms ms and
    #cfg.n_batch trials, with seeds #cfg.seeds.map(str).join(", ").
    We included all conditions and averaged spike counts over the full duration,
    trials and neurons within each population; overlapping controls reused
    measurements, giving #n-seeds seeds at each of 18 conditions from
    #(14 * n-seeds) distinct simulations. The reanalysis used these rates,
    not raw spike trains; population rates alone do not establish gamma
    oscillations #cite(1).

  #run-view("exp047", inputs)

  #reference-list((
    (text: [G. Buzsáki and X.-J. Wang. “Mechanisms of Gamma Oscillations.”
      _Annual Review of Neuroscience_ 35, 203–225 (2012).],
      doi: "10.1146/annurev-neuro-062111-150444"),
  ))

]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How should synaptic strength scale when inhibitory pool size changes? Compare fixed total coupling against fixed per-synapse coupling.],
    preview-figures, json-inputs: ("exp047",),
  )
}

#let meta = meta + (assets: input-assets("exp047", inputs))
#let body = with-datasets("exp047", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
