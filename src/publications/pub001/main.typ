#set document(
  title: "Inhibition-gated sparsity in trained spiking networks",
  author: ("Eoin Murray", "Timothy O'Leary"),
)

#set page(paper: "a4", margin: 2cm, columns: 2)
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#place(top, scope: "parent", float: true)[
  #align(center)[
    #text(size: 16pt, weight: "bold")[
      Inhibition-gated sparsity in trained spiking networks
    ]
    #v(0.5em)
    #text(size: 11pt)[Eoin Murray and Timothy O'Leary]
  ]
  #v(1em)
]

= Abstract <abstract>

TODO

= Introduction <introduction>

TODO

= Methods <method>

== The neuron model <neuron-model>

This paper uses a conductance-based leaky integrate-and-fire (LIF) network with separate excitatory and inhibitory populations, fixed recurrent E↔I matrices, and exponential synapses. The COBA neurons obey

$ C_m frac(d V, d t) = - g_L (V - E_L) - g_e (V - E_e) - g_i (V - E_i) $ <eqn-1>

where $V$ is membrane potential, $t$ is time, $C_m$ is capacitance, $g_L$ is leak conductance, $g_e$ is the excitatory synaptic conductance and $g_i$ is the inhibitory synaptic conductance with reversal potentials $E_L = -65 "mV"$ (leak), $E_e=0 "mV"$ (excitatory), $E_i=−80 "mV"$ (inhibitory). Each conductance term pulls $V$ toward its reversal potential at a rate proportional to the conductance. Excitatory synapses drive $V$ up towards $E_e = 0$ and inhibitory synapses both pull it down towards $E_i=-80$ and increase total conductance, which lowers the effective membrane time constant - this is _shunting inhibition_ and is required to create the PING oscillations.

@eqn-1 can be written as a linear first-order ODE in the form $C_m dot(V) = -A V +B$ with $A =g_L + g_e + g_i$ and $B =g_L E_L + g_e E_e + g_i E_i$. Setting $dot(V) = 0$ we find the voltage in which the inflow $B$ exactly balances the outflow $A V$, and name that voltage $V_oo$. We define accordingly:

$ g_"tot" =g_L + g_e + g_i $ <eqn-2>

$ tau_"eff" = frac(C_m, g_"tot") $ <eqn-3>

$ V_oo = frac(g_L E_L + g_e E_e + g_i E_i, g_"tot") $ <eqn-4>

Substituting Equations~(#ref(<eqn-2>, supplement: none))--(#ref(<eqn-4>, supplement: none)) into @eqn-1 and dividing through by $g_"tot"$ yields the standard “decay-to-steady-state” form

$ tau_"eff" frac(d V, d t) = - (V - V_oo) $ <eqn-5>

Integrating over an interval where $tau_"eff"$ and $V_oo$ are held constant gives closed form solution

$ V(t) = V_oo + (V_0 - V) exp(frac(-t, tau_"eff")), $

the membrane interpolates from its starting value $V_0$ toward $V_oo$ on timescale $tau_"eff"$. However in continious time $tau_"eff"$ and $V_oo$ are _not_ constant, both depend on $g_e(t)$ and $g_i(t)$ which evolve as spikes arrive. A zero-order hold makes this numerically tractable, freeze $g_e$ and $g_i$ over one timestep $Delta t$, integrate @eqn-5 exactly under that freeze, then update $g_e$ and $g_i$ for the next timestep, then @eqn-5 integrates analytically to

$ V_"t+1" = V_oo + (V_t - V_oo) exp(frac(-t, tau_"eff")). $

After the membrane update, apply the spike reset:

$ S_"t+1" = bold(1)[V_"t+1" >= V_"th"], V_"t+1" <- V_"reset" "if" S_"t+1" = 1 "or refractory" $

with $V_"th" =-50 "mV"$ and $V_"reset" = -65 "mV"$. Refractory periods are $tau^e_"ref" = 3 "ms"$ and $tau^i_"ref" = 1.5 "ms"$, held in a per-neuron countdown that supresses spiking until exhaustion.

Each conductance is the convolution of its presynaptic spike train with a decaying exponential. For a single neuron receiving feedforward input only:

$ g_"e, t+1" = alpha g_"e, t" + W_"in" s^"inp"_t $

where $alpha = exp(frac(-Delta t, tau_"AMPA"))$ is the per-step AMPA decay factor. Between spikes the conductance decays exponentially toward zero; each arriving spike adds a kick proportional to the synaptic weight. The inhibitory conductance $g_i$ follows the same form with GABA decay $gamma = exp(frac(-Delta t, tau_"GABA"))$.

In a network with E and I populations, two recurrent pathways close the PING loop alongside the feedforward input. There are no E→E connections.

$ g^E_"e,t+1" = alpha g^e_"e,t" + W_"in" s^"inp"_t $ <eqn-6>

$ g^E_"i,t+1" = gamma(g^e_"i,t" + W_"ie" s^"i"_t) $ <eqn-7>

$ g^E_"e,t+1" = gamma(g^i_"e,t" + W_"ei" s^"e"_t) $ <eqn-8>

@eqn-6 is the excitatory conductance on E neurons — purely feedforward, no recurrent E→E term. @eqn-7 is the inhibitory conductance on E neurons — driven by I spikes through $W_"ie"$. @eqn-8 is the excitatory conductance on I neurons — driven by E spikes through $W_"ei"$. I neurons receive no inhibition. The trainable feedforward input $W_"in" s^"inp"_t$ enters E’s excitatory conductance after the decay step, arriving fresh each timestep regardless of recurrent state.

The model emits two modes, COBA and PING, when PING mode is active the $W_"ei"$ and $W_"ie"$ are fixed at init and are _not_ trainable. $W_"ei"$ and $W_"ie"$ are controlled by a single scalar flag $s$:

$ W_"ei" ~ N (s, 0.1 s) $

$ W_"ie" ~ N(2.0 s, 0.2 s) $

At $s = 0$ both matrices are zero, the loop is open, and the network is feedforward COBA. At $s=1$ the default initialization gives $W_"ei" approx 1 mu"S"$ and $W_"ie" approx 2 mu"S"$, strong enough to sustain the PING rhythm from the first forward pass of training.

== The network model <network-model>

== Training

= Results <results>

== Lower rates

== Control

== Perturbation

= Discussion <discussion>

== Why PING has a rate floor

= Conclusion <conclusion>

TODO
