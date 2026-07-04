#let meta = (
  title: "What sets the PING rate: the per-synapse weight, not the pool size",
  date: "2026-07-04",
  description: "Sweeping the I→E weight against the inhibitory pool size on an untrained PING network — the firing rate collapses onto one curve in W^IE, independent of N_I.",
  collection: "ping-networks",
  status: "building",
)

#let run = json("/artifacts/data/exp047/numbers.json")

#let body = [
  In a PING (pyramidal–interneuron network gamma) circuit, excitatory cells drive a
  shared inhibitory pool, and that inhibition feeds back to shunt the excitatory cells.
  A natural question when scaling such a network is whether the *number* of inhibitory
  neurons $N_I$ sets the operating rate, or whether it is the *strength* of each
  inhibitory synapse $W^(I E)$.

  Here the two are swept independently on an *untrained* PING network at the canonical
  biophysical initialisation — no learning, just the fixed-point rate the loop settles
  into under uniform Poisson drive. Each excitatory cell receives inhibition
  $ I_i^"inh"(t) = sum_(j in "I, spiked") W^(I E)_(i j) g_i (t), $
  where $W^(I E)$ is the per-synapse weight (mean swept) and $g_i$ is the fixed
  inhibitory release conductance. The claim is that the rate is a function of the
  per-event shunt $W^(I E)$ relative to the fixed release level — not of $N_I$.

  The sweep confirms it. Plotting the per-cell E and I rate against $W^(I E)$, one line
  per pool size, the $N_I in {16, 64, 256}$ curves land on top of one another: pool
  size does not move the rate, the per-synapse weight does. Increasing $W^(I E)$ shunts
  each excitatory cell harder per inhibitory spike, so the whole loop settles at a lower
  rate — monotonically, across three decades of $N_I$.

  #figure(
    image("/artifacts/data/exp047/rate_vs_w_ie.png", width: 100%),
    caption: [
      E (left) and I (right) per-cell firing rate versus the per-synapse weight
      $W^(I E)$, one line per inhibitory pool size $N_I$. The lines overlap — the rate
      is set by $W^(I E)$, not $N_I$. The dotted line marks the biophysical-init default.
    ],
  )

  The three commonly quoted "scalings" — constant per-synapse, synaptic $1\/N$, and
  critical $1\/sqrt(N)$ — are then just rules for *choosing* $W^(I E)$ given $N_I$, i.e.
  paths across this single master curve. They need an arbitrary anchor and add nothing
  to the mechanism, so they are not plotted.

  #let cfg = run.config
  Run scale: #cfg.n_hidden excitatory cells, #cfg.n_batch trials,
  #cfg.t_ms ms at #cfg.dt ms steps, uniform Poisson input at #cfg.input_rate Hz, seed
  #cfg.seed — a #(cfg.w_ie_values.len())×#(cfg.n_i_sweep.len()) grid of
  $W^(I E)$ × $N_I$.
]
