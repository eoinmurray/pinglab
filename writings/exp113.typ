#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": cite

#let meta = (
  tags: ("txt", "v36.0.0"),
  title: "COBA and PING Dynamics, Discretisation, and Learning",
  created_at: "2026-09-14T00:00:00Z",
  updated_at: "2026-09-15",
  description: "A textbook derivation of the conductance-based COBA and PING equations, their discrete simulation, and their training by surrogate-gradient BPTT, voltage-gradient damping, gradient clipping, AdamW, and Dale projection.",
  collection: "demo",
)

#let inputs = ()

#let report-body = [
  #journal-abstract(body: [
    COBA and PING classifiers share conductance-based leaky integrate-and-fire
    dynamics but differ in whether excitatory and inhibitory populations form a
    reciprocal feedback loop. We derive their event-driven synapses,
    exponential-Euler membrane update, spike and readout rules, and complete
    training calculation. We then explain why long recurrent gradient paths can
    amplify in this model and how voltage-gradient damping stabilizes selected
    backward sensitivities without changing the forward simulation.
  ])

  == Model family and notation

  The COBA and PING classifiers use the same conductance-based leaky
  integrate-and-fire neuron. Their distinction is architectural. The COBA
  condition disables the recurrent excitatory-to-inhibitory-to-excitatory
  pathway, whereas PING enables the loop
  $E arrow.r I arrow.r E$.#cite(1)

  Physical time is $t$, the simulation-step index is $k$, and
  $t_k=k Delta t_"sim"$, where $Delta t_"sim"$ is the integration timestep.
  A presentation contains $N_t$ timesteps and has duration
  $T_"present"=N_t Delta t_"sim"$. Superscripts on conductances identify
  AMPA or GABA polarity, while subscripts identify the receiving population.

  == Continuous membrane dynamics

  For neuron $j$ in population $X in {E,I}$, the membrane equation is

  $
    C_(m,X) frac(d V_(m,X,j), d t)
      =-g_(L,X)(V_(m,X,j)-E_L)
       -g^E_(X,j)(V_(m,X,j)-E_E)
       -g^I_(X,j)(V_(m,X,j)-E_I).
  $ <eq:continuous-membrane>

  In @eq:continuous-membrane, $V_m$ is membrane voltage, $C_m$ is
  capacitance, $g_L$ is leak conductance, and $g^E$ and $g^I$ are total AMPA
  and GABA conductances. The reversal potentials are

  $
    E_L=-65 "mV", quad E_E=0 "mV", quad E_I=-80 "mV".
  $ <eq:reversal-potentials>

  A conductance $g$ with reversal potential $E$ supplies current
  $I=g(E-V_m)$. @eq:reversal-potentials therefore makes AMPA normally
  depolarizing and GABA normally hyperpolarizing. The fixed membrane parameters
  are

  $
    C_(m,E)=1 "nF", quad g_(L,E)=0.05 "µS", quad
    C_(m,I)=0.5 "nF", quad g_(L,I)=0.1 "µS".
  $ <eq:membrane-parameters>

  The corresponding passive membrane time constants follow from
  $tau_m=C_m/g_L$:

  $
    tau_(m,E)=20 "ms", quad tau_(m,I)=5 "ms".
  $ <eq:passive-time-constants>

  == Continuous synaptic dynamics

  The continuous spike train emitted by presynaptic neuron $i$ is

  $
    s_i(t)=sum_j delta(t-t_(i,j)),
  $ <eq:continuous-spike-train>

  where $t_(i,j)$ is event time $j$ and $delta$ is the Dirac impulse. A
  postsynaptic conductance $g_j$ with decay constant $tau$ satisfies

  $
    frac(d g_j, d t)
      =-frac(g_j, tau)+sum_i w_("event",i,j) s_i(t).
  $ <eq:continuous-synapse>

  Here $w_("event",i,j)$ is the conductance increment produced by one
  presynaptic event. AMPA and GABA use separate copies of
  @eq:continuous-synapse with $tau_"AMPA"=2 "ms"$ and the canonical
  $tau_"GABA"=6 "ms"$.

  == Discretising the synaptic equations

  Between events, @eq:continuous-synapse reduces to

  $
    frac(d g, d t)=-frac(g, tau), quad frac(d g, g)=-frac(d t, tau).
  $ <eq:synapse-separation>

  Integrating @eq:synapse-separation from $t_k$ to
  $t_(k+1)=t_k+Delta t_"sim"$ gives

  $
    integral_(g[k])^(g^-[k+1]) frac(d g, g)
      =-integral_(t_k)^(t_k+Delta t_"sim") frac(d t, tau), quad
    ln frac(g^-[k+1], g[k])=-frac(Delta t_"sim", tau).
  $ <eq:synapse-integral>

  Exponentiating @eq:synapse-integral yields the event-free decay

  $
    g^-[k+1]=g[k] exp(-frac(Delta t_"sim", tau)).
  $ <eq:synapse-decay>

  Integrating @eq:continuous-synapse across an infinitesimal interval
  containing one event gives

  $
    g(t_i^+)-g(t_i^-)=w_"event".
  $ <eq:synapse-jump>

  Combining @eq:synapse-decay and @eq:synapse-jump gives the
  timestep recurrence

  $
    bold(g)[k+1]
      =exp(-frac(Delta t_"sim", tau)) bold(g)[k]
       +bold(s)[k] W+bold(u)[k].
  $ <eq:discrete-synapse>

  In @eq:discrete-synapse, $bold(s)[k]$ is a dimensionless binary
  presynaptic spike vector, $W$ has presynaptic neurons on rows and
  postsynaptic neurons on columns, and $bold(u)[k]$ is an external conductance
  increment. The simulator performs exact exponential decay followed by the
  full, undecayed event increment.#cite(1)

  == E–I conductance recurrences

  Define the dimensionless decay factors

  $
    beta_"AMPA"=exp(-frac(Delta t_"sim", tau_"AMPA")), quad
    beta_"GABA"=exp(-frac(Delta t_"sim", tau_"GABA")).
  $ <eq:synaptic-decay-factors>

  For one E–I layer, @eq:discrete-synapse expands to

  $
    bold(g)^E_E[k+1]
      =beta_"AMPA" bold(g)^E_E[k]
       +bold(s)_E[k] W_(E arrow.r E)
       +bold(x)[k] W_"in"+bold(u)^E_E[k],
  $ <eq:e-excitatory-conductance>

  $
    bold(g)^I_E[k+1]
      =beta_"GABA" bold(g)^I_E[k]
       +bold(s)_I[k] W_(I arrow.r E)+bold(u)^I_E[k],
  $ <eq:e-inhibitory-conductance>

  $
    bold(g)^E_I[k+1]
      =beta_"AMPA" bold(g)^E_I[k]
       +bold(s)_E[k] W_(E arrow.r I)
       +bold(x)_I[k] W_("in",I)+bold(u)^E_I[k],
  $ <eq:i-excitatory-conductance>

  $
    bold(g)^I_I[k+1]
      =beta_"GABA" bold(g)^I_I[k]
       +bold(s)_I[k] W_(I arrow.r I)+bold(u)^I_I[k].
  $ <eq:i-inhibitory-conductance>

  The canonical architecture sets $W_(E arrow.r E)=W_(I arrow.r I)=0$.
  COBA additionally disables $W_(E arrow.r I)$ and $W_(I arrow.r E)$.
  PING enables those two projections: excitatory spikes recruit inhibitory
  neurons through @eq:i-excitatory-conductance, and inhibitory spikes
  subsequently produce GABA conductance through
  @eq:e-inhibitory-conductance. A recurrent spike generated in one iteration is
  consumed in the next, giving each recurrent projection a one-timestep delay.

  == Exponential-Euler membrane discretisation

  The simulator first updates conductances with
  @eq:e-excitatory-conductance–@eq:i-inhibitory-conductance, then holds those
  new values constant during the membrane step. Expanding
  @eq:continuous-membrane for one neuron gives

  $
    C_m frac(d V_m, d t)
      =-(g_L+g_E+g_I)V_m+g_L E_L+g_E E_E+g_I E_I.
  $ <eq:expanded-membrane>

  Define the total conductance and its conductance-weighted equilibrium voltage
  by

  $
    g_"tot"=g_L+g_E+g_I, quad
    V_infinity=frac(g_L E_L+g_E E_E+g_I E_I, g_"tot").
  $ <eq:effective-equilibrium>

  Substituting @eq:effective-equilibrium into
  @eq:expanded-membrane gives

  $
    C_m frac(d V_m, d t)=-g_"tot"(V_m-V_infinity), quad
    tau_"eff"=frac(C_m, g_"tot").
  $ <eq:effective-membrane-equation>

  Let $q=V_m-V_infinity$. Because the conductances and $V_infinity$ are
  constant within the timestep,

  $
    frac(d q, d t)=-frac(q, tau_"eff"), quad
    frac(d q, q)=-frac(d t, tau_"eff").
  $ <eq:voltage-separation>

  Integrating @eq:voltage-separation over one timestep gives

  $
    ln frac(q[k+1], q[k])=-frac(Delta t_"sim", tau_"eff"), quad
    q[k+1]=q[k] exp(-frac(Delta t_"sim", tau_"eff")).
  $ <eq:voltage-integral>

  Returning to membrane voltage produces the pre-threshold candidate

  $
    V_"candidate"[k+1]
      =V_infinity+(V_m[k]-V_infinity)
       exp(-frac(Delta t_"sim" g_"tot", C_m)).
  $ <eq:exponential-euler-voltage>

  Equivalently, define

  $
    a = exp(-frac(Delta t_"sim" g_"tot", C_m)), \
    Delta V[k] = (V_infinity-V_m[k])(1-a), \
    V_"candidate"[k+1] = V_m[k]+Delta V[k].
  $ <eq:membrane-increment>

  @eq:exponential-euler-voltage is the update used by the simulator. It is
  exact for the linear membrane equation under the stated zero-order hold on
  conductances; it is not an exact solution of the continuously co-evolving
  neuron and synapses.

  == Spike, reset, and refractory dynamics

  Let $r[k]$ be the non-negative integer refractory counter. It is decremented
  before the threshold decision:

  $
    r^-[k+1]=max(r[k]-1,0).
  $ <eq:refractory-decrement>

  A neuron emits the binary spike

  $
    s[k+1]=H(V_"candidate"[k+1]-V_"th")
      bold(1)[r^-[k+1]=0],
  $ <eq:hard-spike>

  where $H$ is the Heaviside step function, $V_"th"=-50 "mV"$, and
  $bold(1)$ is an indicator. A neuron that spikes, or remains refractory after
  @eq:refractory-decrement, is stored at
  $V_"reset"=-65 "mV"$. A new spike sets the counter to the population's
  refractory step count; otherwise it retains $r^-[k+1]$.

  The canonical timestep is $Delta t_"sim"=0.1 "ms"$. The excitatory and
  inhibitory refractory durations, $1.2 "ms"$ and $0.6 "ms"$, therefore
  correspond exactly to 12 and 6 timesteps.

  == Output readout and classification loss

  The canonical classifier uses an output LIF with mean pre-reset voltage.
  Let $bold(h)[k]$ be the final hidden E-population spike vector and
  $W_"out"$ the hidden-to-output matrix. With output decay
  $beta_o=exp(-Delta t_"sim"/tau_o)$, the pre-reset dimensionless output state
  is

  $
    bold(a)_o[k+1]=beta_o bold(u)_o[k]
      +frac(1-beta_o, Delta t_"sim") bold(h)[k+1] W_"out".
  $ <eq:output-candidate>

  The output spike and subtractive reset are

  $
    bold(s)_o[k+1]=H(bold(a)_o[k+1]-theta_o), quad
    bold(u)_o[k+1]=bold(a)_o[k+1]-theta_o bold(s)_o[k+1],
  $ <eq:output-reset>

  where $theta_o=1$. Output state and its evidence accumulator begin at zero
  for each independent image presentation. Hidden E/I voltages, conductances,
  spikes and refractory counters likewise initialize anew for every independent
  presentation. The decoder therefore receives presentation boundaries through
  these per-forward resets. The class score is the mean pre-reset state over the
  complete readout window:

  $
    z_c=frac(1, N_t) sum_(k=1)^(N_t) a_(o,c)[k].
  $ <eq:class-score>

  For minibatch member $b$, softmax-normalized evidence is

  $
    p_(b,c)=frac(exp(z_(b,c)), sum_(q=1)^(N_"out") exp(z_(b,q))).
  $ <eq:softmax-evidence>

  It is used for optimization but is not assumed to be a calibrated
  probability. For true class $y_b$, the mean cross-entropy training loss is

  $
    L_"CE"=-frac(1,B) sum_(b=1)^B ln p_(b,y_b).
  $ <eq:cross-entropy>

  Some conditions add a one-sided firing-rate ceiling. Let
  $K_(b,l,j)=sum_(k=1)^(N_t) s_(b,l,j)[k]$ be the spike count of hidden E neuron
  $j$ in layer $l$ for presentation $b$. Its sample-wise population-mean rate
  is

  $
    bar(r)_(b,l)=frac(1, N_(E,l) T_"present")
      sum_(j=1)^(N_(E,l)) K_(b,l,j).
  $ <eq:sample-population-rate>

  With tested ceiling $r_(E,"ceil")$ and coefficient $lambda_"rate"$, the
  penalty is

  $
    L_"rate"=frac(lambda_"rate", B L)
      sum_(b=1)^B sum_(l=1)^L
      max(0,bar(r)_(b,l)-r_(E,"ceil"))^2, quad
    L_"total"=L_"CE"+L_"rate".
  $ <eq:total-loss>

  The canonical reference conditions set $lambda_"rate"=0$; activity-ceiling
  conditions use @eq:total-loss with a positive coefficient.

  == Backpropagation through time

  Collect all differentiable voltages, conductances, spikes, output states and
  accumulators into the state vector $bold(x)[k]$. One complete simulation step
  has the abstract form

  $
    bold(x)[k+1]=F_k(bold(x)[k],bold(u)[k];theta), quad
    L_"total"=ell(bold(x)[N_t],y).
  $ <eq:state-update>

  $F_k$ is a bundle rather than one additional simulator equation. It combines
  the discrete synapse rule in @eq:discrete-synapse and the decay factors in
  @eq:synaptic-decay-factors; the population-specific conductance updates from
  @eq:e-excitatory-conductance through @eq:i-inhibitory-conductance; the
  implemented membrane update in @eq:membrane-increment; the refractory,
  spike and reset operations in @eq:refractory-decrement and @eq:hard-spike;
  and the output update in @eq:output-candidate and @eq:output-reset. It also
  advances the accumulators used to construct @eq:class-score and
  @eq:sample-population-rate. The earlier continuous equations and intermediate
  derivation steps explain these operations but are not extra updates inside
  $F_k$.

  Here $bold(u)[k]$ denotes external input, $theta$ is the shared trainable
  parameter vector, and $ell$ includes the readout and objective defined in
  @eq:class-score–@eq:total-loss. At $Delta t_"sim"=0.1 "ms"$, a
  $200 "ms"$ presentation applies $F_k$ 2,000 times. Each copy has a different
  state but reuses the same parameters.

  Define the state sensitivity

  $
    bold(delta)[k]=nabla_(bold(x)[k]) L_"total", quad
    J_"step"[k]=frac(partial F_k, partial bold(x)[k]).
  $ <eq:bptt-definitions>

  Let $x_i[k]$ be component $i$ of the current state and $F_(k,j)$ component
  $j$ of $F_k$. The project evaluates its loss from the terminal readout and
  rate accumulators. Consequently, for $k<N_t$, the current state affects the
  loss through the next state. Applying the multivariable chain rule component
  by component gives

  $
    delta_i[k]
      &= frac(partial L_"total", partial x_i[k]) \
      &= sum_j
         frac(partial L_"total", partial x_j[k+1])
         frac(partial x_j[k+1], partial x_i[k]) \
      &= sum_j delta_j[k+1]
         frac(partial F_(k,j), partial x_i[k]), \
    bold(delta)[k]
      &= J_"step"[k]^top bold(delta)[k+1], quad
    bold(delta)[N_t]
      =nabla_(bold(x)[N_t]) ell(bold(x)[N_t],y).
  $ <eq:bptt-adjoint>

  The second line of @eq:bptt-adjoint sums every route from state component
  $x_i[k]$ to the later loss through next-state component $x_j[k+1]$. The third
  line substitutes @eq:bptt-definitions and @eq:state-update. Stacking the
  component equations produces the transpose-Jacobian vector product in the
  fourth line; the final line supplies the terminal condition from
  @eq:state-update. Because each parameter is reused, its total gradient is

  $
    nabla_theta L_"total"
      =nabla_theta^"direct" L_"total"
       +sum_(k=0)^(N_t-1)
        frac(partial F_k, partial theta)^top bold(delta)[k+1].
  $ <eq:bptt-parameter-gradient>

  @eq:bptt-adjoint and @eq:bptt-parameter-gradient define
  backpropagation through time (BPTT). PyTorch autograd evaluates these
  vector–Jacobian products and accumulates parameter gradients when backward
  propagation is requested; it need not construct the full Jacobian matrices
  explicitly.#cite(4)

  For example, the local recurrence
  $bold(g)[k+1]=beta bold(g)[k]+bold(s)[k]W$ gives

  $
    nabla_W L_"total"
      =sum_k bold(s)[k]^top
        nabla_(bold(g)[k+1]) L_"total".
  $ <eq:synaptic-weight-gradient>

  @eq:synaptic-weight-gradient shows temporal credit assignment
  directly: every use of the shared weight contributes according to the later
  loss sensitivity of the conductance it created.

  == Surrogate spike derivatives

  BPTT requires derivatives of the update operations, but the hard spike in
  @eq:hard-spike has derivative zero away from threshold and is
  undefined at threshold. Direct use of that derivative would block useful
  learning signals.

  Surrogate-gradient learning retains the binary forward spike while assigning
  a smooth backward derivative.#cite(2) For
  $xi=V_"candidate"-V_"th"$, the chosen proxy is

  $
    p(xi)=frac(beta xi, 1+beta abs(xi)), quad
    frac(partial s, partial xi) approx
      frac(beta, (1+beta abs(xi))^2).
  $ <eq:surrogate-derivative>

  The approximate equality in @eq:surrogate-derivative denotes an assigned
  backward rule rather than the true derivative of the Heaviside function. For
  hidden neurons, $xi$ is measured in millivolts and $beta$ in reciprocal
  millivolts. The standard classifiers use $beta=1$.

  Let $op("sg")(z)$ denote stop-gradient: it returns $z$ forward but has zero
  derivative. The simulator constructs

  $
    s=op("sg")(H(xi))+p(xi)-op("sg")(p(xi)).
  $ <eq:surrogate-stop-gradient>

  @eq:surrogate-stop-gradient equals $H(xi)$ numerically, while
  autograd differentiates only $p(xi)$.#cite(5) Boolean reset and refractory
  decisions remain non-differentiable. Resetting hidden voltage to a constant
  cuts that voltage-state path, while the floating spike value carries the
  surrogate derivative into downstream synapses and the readout.

  == Why voltage gradients can explode

  Expanding @eq:bptt-adjoint across many steps produces repeated
  Jacobian products:

  $
    bold(delta)[k]
      =J_"step"[k]^top J_"step"[k+1]^top dots
       J_"step"[N_t-1]^top bold(delta)[N_t]
       +"direct terms".
  $ <eq:repeated-jacobians>

  Products in @eq:repeated-jacobians can shrink useful sensitivities
  or amplify them. This project combines a 2,000-step graph with voltage,
  conductance, spike and output-state memory. PING adds a closed route by which
  a voltage perturbation can return to the population from which it began:

  $
    delta bold(V)_E arrow.r delta bold(s)_E
      arrow.r delta bold(g)_(E arrow.r I)
      arrow.r delta bold(V)_I arrow.r delta bold(s)_I
      arrow.r delta bold(g)_(I arrow.r E)
      arrow.r delta bold(V)_E.
  $ <eq:ping-gradient-route>

  The local gain of this route follows from the membrane equations rather than
  from an extra model quantity. Choose $X=E$ for excitatory conductance or
  $X=I$ for inhibitory conductance. Differentiating
  @eq:effective-equilibrium and the decay $a$ in
  @eq:membrane-increment gives

  $
    frac(partial V_infinity, partial g_X)
      =frac(E_X-V_infinity, g_"tot"), quad
    frac(partial a, partial g_X)
      =-frac(Delta t_"sim", C_m)a.
  $ <eq:conductance-derivatives>

  Applying the product rule to
  $Delta V=(V_infinity-V_m)(1-a)$ gives the conductance-to-voltage response

  $
    R_X:=frac(partial Delta V, partial g_X)
      =frac((E_X-V_infinity)(1-a), g_"tot")
       +(V_infinity-V_m)frac(Delta t_"sim", C_m)a.
  $ <eq:conductance-voltage-response>

  The response $R_X$ in @eq:conductance-voltage-response has units of
  millivolts per microsiemens. Let $R_(E arrow.r I)$ and
  $R_(I arrow.r E)$ be diagonal matrices of the appropriate responses at I and
  E neurons. Let $S_E$ and $S_I$ be diagonal matrices containing the local
  surrogate derivatives from @eq:surrogate-derivative. One complete
  backward traversal of @eq:ping-gradient-route then has return
  Jacobian

  $
    J_"loop"
      =R_(I arrow.r E) W_(I arrow.r E)^top S_I
       R_(E arrow.r I) W_(E arrow.r I)^top S_E.
  $ <eq:loop-jacobian>

  The transposes in @eq:loop-jacobian convert the simulator's
  presynaptic-row, postsynaptic-column weight storage into maps acting on column
  perturbations.

  At a fixed local linearization, a sensitivity returning through the loop
  $n$ times contains $J_"loop"^n delta bold(V)_E$. If the spectral radius
  $rho(J_"loop")>1$, some modes grow geometrically. A singular value above one
  permits amplification over one traversal, while time-varying, non-normal
  products can amplify transiently even if each instantaneous eigenvalue has
  magnitude below one.#cite(3) Inhibitory polarity can reverse a perturbation's
  sign without making its magnitude small.

  Recurrence therefore does not guarantee exploding gradients. It creates the
  return operator in @eq:loop-jacobian, which BPTT may traverse many
  times. Loop-disabled COBA lacks that particular operator, but it still carries
  gradients through membrane, conductance, spike and readout state across the
  same long presentation.

  Global gradient clipping cannot by itself repair this internal propagation.
  It acts only after BPTT has formed the complete parameter gradient. Internal
  sensitivities may already be enormous, non-finite, or dominated by the
  amplifying paths in @eq:repeated-jacobians before clipping occurs.

  == Voltage-gradient damping

  The project therefore intervenes inside the backward recurrence while
  preserving the physical forward trajectory. For dimensionless divisor
  $d_"grad">0$, define

  $
    alpha_"grad"=frac(1,d_"grad"), quad
    cal(D)_alpha(z)=alpha z+(1-alpha)op("sg")(z).
  $ <eq:gradient-scaling-operator>

  Forward evaluation of @eq:gradient-scaling-operator gives
  $cal(D)_alpha(z)=z$, but its backward derivative is

  $
    frac(partial cal(D)_alpha(z), partial z)=alpha.
  $ <eq:gradient-scaling-derivative>

  The hidden E and I membrane updates replace @eq:membrane-increment
  by

  $
    V_"candidate"[k+1]
      =V_m[k]+cal(D)_(alpha_"grad")(Delta V[k]).
  $ <eq:damped-voltage-update>

  @eq:gradient-scaling-operator and @eq:damped-voltage-update leave
  voltages and spikes unchanged forward. Backward, each selected
  conductance-to-voltage response is multiplied by $alpha_"grad"$. The PING
  return term in @eq:loop-jacobian consequently becomes

  $
    J_("loop,damped")=alpha_"grad"^2 J_"loop".
  $ <eq:damped-loop-jacobian>

  The standard paired classifiers use $d_"grad"=1000$, so
  $alpha_"grad"=10^(-3)$ and this particular complete loop term is scaled by
  $10^(-6)$. This does not scale the entire state Jacobian because
  @eq:gradient-scaling-operator is inserted only on the $Delta V$ branch of
  @eq:damped-voltage-update. For one hidden neuron, the three local derivatives
  entering its candidate voltage are therefore

  $
    frac(partial V_"candidate", partial (V_m,g_E,g_I))
      =lr(1+alpha_"grad" frac(partial Delta V,partial V_m),
          alpha_"grad" frac(partial Delta V,partial g_E),
          alpha_"grad" frac(partial Delta V,partial g_I)).
  $ <eq:damped-voltage-jacobian-blocks>

  Thus the conductance-to-voltage blocks are multiplied by
  $alpha_"grad"$, but the direct identity route from $V_m[k]$ to
  $V_"candidate"[k+1]$ retains coefficient one. The conductance decay and
  spike-injection blocks, the refractory-counter update, and the local output
  recurrence contain no damping operator and are not directly rescaled.
  Downstream composite derivatives can nevertheless inherit attenuation when
  their path passes through a damped hidden-voltage increment. There is
  therefore no scalar $alpha_"grad"$ for which the complete step Jacobian is
  simply $alpha_"grad" J_"step"$.

  Holding conductances fixed before bounds and reset,
  @eq:damped-voltage-jacobian-blocks and @eq:membrane-increment give

  $
    frac(partial V_"candidate"[k+1], partial V_m[k])
      =1-alpha_"grad"(1-a).
  $ <eq:damped-voltage-memory>

  Without damping, @eq:damped-voltage-memory reduces to $a$. Strong
  damping moves the direct old-voltage derivative toward one while attenuating
  sensitivities entering through the membrane increment. It can therefore
  suppress recurrent-drive amplification without simply erasing membrane
  memory.

  Voltage-gradient damping is an altered backward rule, not a biophysical
  process or the exact derivative of the forward simulator. It deliberately
  biases temporal credit assignment and may suppress useful sensitivities along
  with unstable ones. The divisor is therefore an empirical training choice.
  Standard classifiers apply it to hidden E and I membrane increments, before
  voltage bounds, spike detection and reset; output-neuron updates are
  undamped.#cite(1)

  == Gradient clipping, AdamW, and Dale projection

  After BPTT has accumulated the batch gradient $bold(g)=nabla_theta
  L_"total"$, the trainer computes its global Euclidean norm

  $
    ||bold(g)||_2=sqrt(sum_p sum_i g_(p,i)^2).
  $ <eq:global-gradient-norm>

  With clipping threshold $c=1$, it replaces the gradient by

  $
    tilde(bold(g))=bold(g) min(1,frac(c,||bold(g)||_2)).
  $ <eq:gradient-clipping>

  A non-finite norm causes the optimizer step to be skipped. Otherwise AdamW
  forms first- and second-moment estimates

  $
    bold(m)[n]=beta_1 bold(m)[n-1]+(1-beta_1)tilde(bold(g))[n], quad
    bold(v)[n]=beta_2 bold(v)[n-1]+(1-beta_2)tilde(bold(g))[n]^2.
  $ <eq:adam-moments>

  The square in @eq:adam-moments is elementwise. Bias correction gives

  $
    hat(bold(m))[n]=frac(bold(m)[n],1-beta_1^n), quad
    hat(bold(v))[n]=frac(bold(v)[n],1-beta_2^n).
  $ <eq:adam-bias-correction>

  For learning rate $eta$, decoupled weight decay $lambda_"wd"$, and numerical
  stabilizer $epsilon$, AdamW applies

  $
    theta^[n+1/2]
      =(1-eta lambda_"wd")theta^[n]
       -eta frac(hat(bold(m))[n],sqrt(hat(bold(v))[n])+epsilon).
  $ <eq:adamw-update>

  The standard training recipe used $eta=0.0004$, $lambda_"wd"=0$, minibatches
  of 256 presentations and 50 epochs. With zero weight decay,
  @eq:adamw-update is numerically the Adam update for the same hyperparameters.

  Dale-constrained trainable connection weights are finally projected onto the
  non-negative orthant:

  $
    theta^[n+1]=max(0,theta^[n+1/2]).
  $ <eq:dale-projection>

  The canonical input and readout matrices are trainable, while recurrent
  E–I matrices are fixed unless an experiment explicitly makes them trainable.
  Thus one optimization cycle evaluates the forward dynamics, computes
  @eq:total-loss, applies surrogate-gradient BPTT with the internal rule in
  @eq:damped-voltage-update, clips by @eq:gradient-clipping,
  updates by @eq:adamw-update, and finishes with
  @eq:dale-projection.

  == Scope and interpretation

  The dynamical discretisation is a numerical approximation to the continuous
  hybrid system, although @eq:synapse-decay and
  @eq:exponential-euler-voltage exactly solve their respective subproblems under
  the stated within-step assumptions. Surrogate spike derivatives and
  voltage-gradient damping introduce additional backward rules: they make
  learning possible and tractable but do not turn BPTT into the exact derivative
  of a smooth physical neuron.

  The loop derivation isolates one interpretable amplification route rather than
  the full state Jacobian, and it does not imply that gamma oscillation itself
  causes gradient explosion. A paired empirical comparison varies loop
  engagement and damping independently:
  #link("/exp112/")[exp112] — #link("/exp112/")[_COBA–PING Gradient-Damping Comparison._]

  #journal-references((
    (text: [Pinglab contributors. _SNNsim and the paired COBA/PING training
      specification._ Software snapshot (2026):
      #link("https://github.com/eoinmurray/pinglab/blob/a1e7361f8afc20f84e20764a03922a931dc0ffec/tools/snnsim/models.py")[membrane dynamics and learning rules],
      #link("https://github.com/eoinmurray/pinglab/blob/a1e7361f8afc20f84e20764a03922a931dc0ffec/tools/snnsim/train.py#L668-L779")[training loop and optimizer], and
      #link("https://github.com/eoinmurray/pinglab/blob/a1e7361f8afc20f84e20764a03922a931dc0ffec/experiments/exp022/recipe.py#L109-L138")[paired training choices].]),
    (text: [E. O. Neftci, H. Mostafa and F. Zenke. “Surrogate Gradient Learning
      in Spiking Neural Networks.” _arXiv_ (2019).],
      doi: "10.48550/arXiv.1901.09948"),
    (text: [R. Pascanu, T. Mikolov and Y. Bengio.
      #link("https://proceedings.mlr.press/v28/pascanu13.html")[“On the Difficulty
      of Training Recurrent Neural Networks.”] _Proceedings of the 30th
      International Conference on Machine Learning_, 1310–1318 (2013).]),
    (text: [PyTorch contributors.
      #link("https://docs.pytorch.org/docs/stable/notes/autograd.html")[“Autograd mechanics”],
      #link("https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html")[“Tensor.backward”], and
      #link("https://docs.pytorch.org/docs/stable/optim.html")[“torch.optim.”]
      _PyTorch documentation_ (accessed 15 September 2026).]),
    (text: [PyTorch contributors.
      #link("https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html")[“Tensor.detach.”]
      _PyTorch documentation_ (accessed 15 September 2026).]),
  ))
]

#let body = [
  #show list: it => enum(..it.children.map(item => item.body))
  #journal-article("exp113", inputs, report-body, dataset: false)
]
