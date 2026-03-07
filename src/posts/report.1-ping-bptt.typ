// ── Document setup ──────────────────────────────────────────────────
#set document(
  title: "Training a PING Network with BPTT",
  author: "Eoin Murray",
  date: datetime(year: 2026, month: 3, day: 6),
)

#set page(
  paper: "a4",
  margin: (x: 2.4cm, y: 2.8cm),
  numbering: "1",
  number-align: center
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1.")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => {
  v(1.2em)
  block(text(weight: "bold", size: 14pt, it))
  v(0.4em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  block(text(weight: "bold", size: 12pt, it))
  v(0.3em)
}

// ── Figure caption styling ──────────────────────────────────────────
#show figure: set block(above: 1.8em, below: 1.8em)
#set figure(gap: 0.8em)
#set figure.caption(separator: [. #h(0.3em)])
#show figure.caption: it => {
  set text(size: 9.5pt)
  set par(justify: true, leading: 0.5em)
  it
}

// ── Title block ─────────────────────────────────────────────────────
#align(center)[
  #v(1.5em)
  #text(size: 20pt, weight: "bold")[Training a PING Network]
  #v(0.8em)
  #text(size: 12pt)[Eoin Murray]
  #v(0.3em)
  #text(size: 10pt)[March 6#super[th], 2026]
  #v(0.5em)
]

#outline(indent: 1.5em, depth: 2)

#v(1fr)
#block(
  width: 100%,
  inset: 12pt,
  radius: 3pt,
  stroke: 0.5pt + luma(160),
  fill: luma(245),
)[
  #set text(size: 8.5pt, fill: luma(80))
  #set par(justify: true, leading: 0.45em)
  *AI disclosure.* This work was produced with the assistance of AI tools (Claude, Anthropic) for code generation, idea exploration, and report drafting. All content --- including experimental design, analysis, and written material --- was directed, reviewed, and certified by the author.
]

#pagebreak()

// ── 1. The PING network equations ───────────────────────────────────
= The PING network equations

== The core model

We start with a current-based leaky integrate-and-fire (LIF) equation:

$
  R C (d V)/(d t) = - (V - V_"rest") + R I (t)
$ <eq:lif-raw>

with the reset condition: if $V >= V_"thresh"$ then $V$ is reset to $V_"reset"$ and a spike is emitted.

Here $R$ is the membrane resistance, $C$ is the membrane capacitance, $V$ is the membrane potential, $t$ is time, $V_"rest"$ is the resting potential, $I(t)$ is the input current, $V_"thresh"$ is the threshold potential, and $V_"reset"$ is the reset potential.

Dividing both sides by $R$:

$
  C (d V)/(d t) = - (V - V_"rest") / R + I(t)
$

Now substitute $tau_m = R C_m$, $g_L = 1\/R$, and $E_L = V_"rest"$:

$
  C_m (d V)/(d t) = -g_L (V - E_L) + I(t)
$ <eq:lif-conductance>

Starting from this conductance form, we derive the PING network equations. The input current $I(t)$ is replaced by synaptic driving forces and an external input current. Each neuron maintains a single excitatory conductance $g_e$ and a single inhibitory conductance $g_i$, which accumulate contributions from all presynaptic sources of the corresponding type.

For *excitatory* neurons:

$
  C_m^E (d V_E)/(d t) = -g_L^E (V_E - E_L) + g_e (E_e - V_E) + g_i (E_i - V_E) + I_"ext"
$ <eq:exc>

For *inhibitory* neurons:

$
  C_m^I (d V_I)/(d t) = -g_L^I (V_I - E_L) + g_e (E_e - V_I) + I_"ext"
$ <eq:inh>

where $E_e$ is the excitatory reversal potential, $E_i$ is the inhibitory reversal potential, and $I_"ext"$ is the external input current (e.g.~from Poisson-encoded stimuli). The membrane parameters $C_m$ and $g_L$ differ between E and I populations. A voltage floor $V_"floor"$ is enforced to prevent numerical divergence.

This is integrated with forward Euler:

$
  V(t + Delta t) = V(t) + Delta t dot (d V)/(d t)
$ <eq:euler>

#v(0.4em)

== Synaptic conductances

The conductances $g_e$ and $g_i$ decay exponentially with time constants $tau_"AMPA"$ and $tau_"GABA"$ respectively. In discrete time this is implemented as exact exponential decay:

$
  g_e (t + Delta t) = g_e (t) dot exp(-Delta t \/ tau_"AMPA")
$ <eq:ge-decay>

$
  g_i (t + Delta t) = g_i (t) dot exp(-Delta t \/ tau_"GABA")
$ <eq:gi-decay>

When a presynaptic spike arrives (after any delay; see below), the contribution is added to the appropriate conductance. For E$arrow.r$E and E$arrow.r$I connections, the excitatory conductance of the postsynaptic neuron is incremented:

$
  g_e arrow.l g_e + bold(s)_E^top bold(W)
$ <eq:ge-update>

where $bold(s)_E$ is the vector of E neuron spike indicators and $bold(W)$ is the weight matrix (shape: pre $times$ post). Similarly, I$arrow.r$E connections increment the inhibitory conductance:

$
  g_i arrow.l g_i + bold(s)_I^top bold(W)_(I E)
$ <eq:gi-update>

All weights are constrained to be non-negative.

#v(0.4em)

== The PING mechanism

The PING mechanism lives in the E $arrow.r$ I $arrow.r$ E feedback loop: E neurons fire, driving I neurons via E$arrow.r$I connections. The I neurons then suppress E via I$arrow.r$E inhibitory connections. E neurons recover and fire again. Gamma-frequency oscillations ($approx$ 30--80 Hz) emerge from the interplay of the I$arrow.r$E delay and $tau_"GABA"$.

In the configurations used for our experiments, the network includes E$arrow.r$E, E$arrow.r$I, and I$arrow.r$E connections but no I$arrow.r$I connections. (The simulator supports I$arrow.r$I connections in general, but they are not used in the PING architecture.)

#v(0.4em)

== Refractory periods

After a neuron fires, it enters a refractory period during which $V$ is held at $V_"reset"$ and no further spikes can be emitted. The refractory periods $tau_"ref"^E$ and $tau_"ref"^I$ differ between populations. In discrete time the refractory period is converted to a step count:

$
  n_"ref" = ceil(tau_"ref" \/ Delta t)
$ <eq:ref>

#v(0.4em)

== Synaptic delays

To capture axonal/dendritic transmission time and synaptic processing latency --- and because they materially shape network timing --- we include synaptic delays. If a neuron spikes at time $t_k$, its contribution to the postsynaptic conductance is delivered after a delay $d$. This is implemented via a circular buffer: spikes are written into a future buffer slot at index $(t + d) mod L$, and read from the current slot at each timestep. Each connection type (E$arrow.r$E, E$arrow.r$I, I$arrow.r$E) can have its own delay.

= Oscillatory analysis

We implemented a PING simulator in Python using PyTorch for efficient tensor operations and automatic differentiation. The simulator integrates the network equations from Section 1 forward in time, producing a binary spike tensor of shape (time $times$ neurons) at each forward pass --- a 1 wherever a neuron fires and a 0 otherwise. This tensor is the raw output of the simulation, and all downstream analysis is derived from it.

To characterise whether a network is exhibiting healthy PING oscillations, we need to go beyond simply looking at individual spikes. We build up a pipeline of increasingly informative representations: first a visual spike raster, then a population firing rate that summarises collective activity over time, then a power spectrum that reveals the frequency content, and finally a signal--noise decomposition that quantifies oscillatory health.

== Spike raster

The most direct way to inspect network activity is the _spike raster plot_, where each row corresponds to a neuron and each dot marks the time at which that neuron fired. Neurons are grouped by population, so spatial structure in the plot reflects the network architecture.

#figure(
  image("../_artifacts/study.17-ping-metrics/raster_main_main_00_light.png", width: 60%),
  caption: [Spike raster from a PING network. Each row is a neuron, each dot a spike. Colours indicate neuron type (blue: E, red: I). The alternating bursts of E and I activity reflect the characteristic PING rhythm.],
) <example-raster>

In @example-raster, the PING oscillation is clearly visible: the E neurons (blue) fire in synchronised bursts, which drive the I neurons (red) to fire shortly after. The I neurons then suppress the E population through inhibition, and the cycle repeats. This alternating E--I pattern is the hallmark of a healthy PING network.

== Population firing rate

While the raster plot is informative visually, we need a quantitative summary of population activity to perform further analysis. The _population firing rate_ collapses the activity of all neurons in a population into a single time series by counting how many spikes occur in each time bin.

Concretely, we divide the simulation into bins of width $Delta t_"bin"$ (we use 5 ms by default) and count the number of spikes from the target population in each bin. Dividing by the number of neurons and the bin duration gives a firing rate in Hz:

$
  r(t_k) = n_k / (N_"pop" dot Delta t_"bin")
$ <eq:pop-rate>

where $n_k$ is the spike count in the $k$-th bin, $N_"pop"$ is the number of neurons in the population, and $Delta t_"bin"$ is expressed in seconds so that $r$ has units of Hz. If the time series is noisy, it can optionally be smoothed with a Gaussian kernel of width $sigma$ ms.

#figure(
  image("../_artifacts/study.17-ping-metrics/pop_rates_main_main_00_light.png", width: 60%),
  caption: [Population firing rates for each layer. The oscillatory modulation of the E population rate reflects the PING rhythm.],
) <example-pop-rate>

@example-pop-rate shows the population rate computed separately for each layer. The E population rate oscillates rhythmically --- rising during each synchronised burst and falling during the inhibitory pause. This oscillation is the time-domain signature of the PING mechanism.

== Power spectral density

The population rate tells us _that_ the network oscillates, but to determine _at what frequency_ and _how cleanly_, we move to the frequency domain by computing the power spectral density (PSD).

The PSD decomposes a time series into its constituent frequencies, revealing how much power (variance) is present at each frequency. Before computing the PSD, we subtract the mean firing rate to remove the _DC component_ --- this is the constant offset corresponding to the average activity level, which would otherwise produce a large peak at $f = 0$ and obscure the oscillatory peaks we care about:

$
  tilde(r)(t) = r(t) - overline(r)
$

We then apply the Fast Fourier Transform (FFT) to obtain the power at each frequency:

$
  P(f) = |cal(F){tilde(r)}|^2 / n_"bins"
$ <eq:psd>

where $n_"bins"$ is the number of time bins. The frequency axis ranges from 0 to $f_s\/2$ (the Nyquist frequency) with resolution $Delta f = f_s \/ n_"bins"$, where $f_s = 1000 \/ Delta t_"bin"$ Hz is the sampling frequency determined by the bin width.

#figure(
  image("../_artifacts/study.17-ping-metrics/psd_main_main_00_light.png", width: 60%),
  caption: [Power spectral density of the E population rate. The dominant peak at $approx$ 30 Hz corresponds to the fundamental PING frequency $f_0$; peaks at $2 f_0$, $3 f_0$, etc.~are harmonics.],
) <example-psd>

In @example-psd, the PSD shows a clear peak at approximately 30 Hz --- this is the fundamental oscillation frequency $f_0$ of the PING network. The additional peaks at $2 f_0 approx 60$ Hz, $3 f_0 approx 90$ Hz, and so on are _harmonics_: they arise because the PING oscillation is not a pure sine wave but has sharp, pulse-like bursts, which produce energy at integer multiples of the fundamental.

== Gaussian comb and signal--noise decomposition

The PSD tells us the oscillation frequency, but we also want to quantify how much of the network's activity is _oscillatory_ versus _noisy_. A healthy PING network should concentrate most of its spectral power at the fundamental and its harmonics, with little power elsewhere. To measure this, we decompose the PSD into signal and noise components using a _Gaussian comb filter_.

The idea is straightforward: we build a mask that is "on" near the fundamental frequency and each of its harmonics, and "off" everywhere else. Any power that falls under the mask is classified as signal (structured oscillatory activity), and any power outside the mask is classified as noise (irregular or aperiodic activity).

First, we identify the fundamental frequency $f_0$ as the location of the largest peak in the PSD within a search band $[f_"lo", f_"hi"]$ (default 5--80 Hz, covering the typical gamma range). We then construct a comb mask $C(f)$ consisting of $K$ Gaussian bumps ("teeth") centred at the harmonics $f_0, 2 f_0, dots, K f_0$:

$
  C(f) = min(1, sum_(k=1)^K exp(-(f - k f_0)^2 / (2 sigma_"Hz"^2)))
$ <eq:comb>

Each tooth is a Gaussian with width $sigma_"Hz"$ (default 1.75 Hz), so it captures power in a narrow band around each harmonic. The $min(1, dots)$ ensures the mask stays between 0 and 1 even where teeth overlap. @example-gaussian-comb shows the resulting comb overlaid on the PSD.

#figure(
  image("../_artifacts/study.17-ping-metrics/psd_comb_main_main_00_light.png", width: 60%),
  caption: [PSD with Gaussian comb overlay. The comb teeth (shaded) capture power at the fundamental and harmonics; the remaining power is classified as noise.],
) <example-gaussian-comb>

With the comb mask in hand, we multiply it element-wise with the PSD to extract signal and noise power:

$
  P_S = sum_f C(f) dot P(f), quad P_N = sum_f (1 - C(f)) dot P(f)
$

The signal-to-noise ratio $"SNR"_"PSD" = P_S \/ P_N$ provides a single number summarising oscillatory health: a high SNR means most of the network's activity is concentrated at the PING frequency and its harmonics, while a low SNR indicates diffuse, irregular activity. The resulting decomposition is shown in @example-psd-decomp.

#figure(
  image("../_artifacts/study.17-ping-metrics/psd_decomp_main_main_00_light.png", width: 60%),
  caption: [Signal--noise decomposition of the PSD. Power under the comb mask (green) is classified as signal; the remainder (red) as noise.],
) <example-psd-decomp>

== Differentiable oscillatory health loss

The analysis pipeline above gives us a way to _measure_ oscillatory health, but to use it as a training objective we need it to be _differentiable_ --- that is, we need to be able to compute gradients of the health metric with respect to the network weights, so that gradient descent can optimise the weights to improve oscillatory structure.

Most of the pipeline is already differentiable: binning spikes, computing the FFT, and multiplying by the comb mask are all standard tensor operations that PyTorch can differentiate through automatically. The one problematic step is finding $f_0$: the hard argmax (picking the frequency with the highest PSD value) has zero gradient almost everywhere, so it cannot guide learning.

We solve this by replacing the hard argmax with a _soft-argmax_ --- a differentiable approximation that computes a weighted average of frequencies, where the weights are determined by a temperature-scaled softmax over the PSD values in the search band:

$
  f_0 = sum_j w_j dot f_j, quad w_j = "softmax"(P(f_j) \/ (tau dot max P))
$ <eq:f0>

When the temperature $tau$ is small, the softmax concentrates its weight on the largest PSD value, closely approximating the hard argmax. But unlike the hard argmax, the soft version has smooth, non-zero gradients that allow the network to learn which frequency to oscillate at.

With this differentiable $f_0$, we build the Gaussian comb mask exactly as before (@eq:comb) and compute three sub-losses that capture different aspects of oscillatory health:

$
  cal(L)_"conc" = -ln(P_S / P_"total"), quad
  cal(L)_"freq" = (f_0 - f_"target")^2, quad
  cal(L)_"noise" = P_N / P_S
$ <eq:health-loss>

- $cal(L)_"conc"$ (spectral concentration) penalises networks where power is spread across many frequencies rather than concentrated at the harmonics. It is minimised when $P_S approx P_"total"$, i.e.~all power is under the comb.

- $cal(L)_"freq"$ (frequency deviation) penalises networks that oscillate at the wrong frequency. It is minimised when $f_0 = f_"target"$.

- $cal(L)_"noise"$ (noise ratio) penalises the absolute level of non-oscillatory activity relative to the signal.

The total health loss is a weighted combination:

$
  cal(L)_"PING" = alpha cal(L)_"conc" + beta cal(L)_"freq" + gamma cal(L)_"noise"
$

where $alpha$, $beta$, $gamma$ control the relative importance of each term. Since every operation in this pipeline --- FFT, softmax, Gaussian evaluation, element-wise products, logarithm --- is differentiable, the entire loss can be computed and back-propagated through alongside the classification loss during BPTT training.

= Training with BPTT

== Forward pass and loss computation

Training a spiking neural network to classify images requires three stages: encoding the input, simulating the network, and reading out a prediction.

*Input encoding.* Each MNIST image is a $28 times 28$ grid of pixel intensities in $[0, 1]$. We flatten this to a vector of 784 values and convert it to a spike train using _Poisson encoding_: at each simulation timestep, each input neuron independently fires a spike with probability equal to its corresponding pixel intensity. A bright pixel fires frequently; a dark pixel rarely fires. The resulting spike train is multiplied by a scale factor $kappa$ (typically 1.5--3.0) to set the injected current amplitude. Because the encoding is stochastic, every forward pass through the same image produces a slightly different spike pattern --- a form of natural data augmentation.

*Network simulation.* The encoded spike train is fed into the PING network as external current $I_"ext"$ to the input layer $E_"in"$. The simulator then integrates the network equations from Section 1 for $T\/Delta t$ discrete timesteps using forward Euler, producing a binary spike tensor $bold(S) in {0, 1}^(B times T times N)$ where $B$ is the batch size, $T$ is the number of timesteps, and $N$ is the total number of neurons.

*Readout.* To convert the output spike tensor into class logits, we use a _hybrid readout_ that combines two signals from the output layer $E_"out"$ (which has one neuron per class):

$
  bold(ell) = sum_(t=1)^T bold(s)_"out" (t) + alpha_"ro" dot 1/T sum_(t=1)^T bold(V)_"out" (t)
$ <eq:readout>

The first term counts total output spikes per neuron over the simulation window --- a discrete signal that directly reflects the network's spiking decision. The second term is the time-averaged membrane voltage, scaled by a small coefficient $alpha_"ro"$ (default 0.01). The voltage term provides a smooth, continuous gradient signal even when output neurons are not spiking, which is especially important early in training when the network has not yet learned to produce meaningful output spikes.

*Loss.* The logits $bold(ell)$ are passed to a cross-entropy loss:

$
  cal(L)_"CE" = -ln (exp(ell_y)) / (sum_c exp(ell_c))
$ <eq:ce>

where $y$ is the true class label. When PING regularisation is enabled, the total loss includes the oscillatory health term from Section 2:

$
  cal(L) = cal(L)_"CE" + lambda cal(L)_"PING"
$ <eq:total-loss>

where $lambda$ controls the regularisation strength (typically 0.01).

== Back-propagation through time

To update the synaptic weight matrices $bold(W)$, we need the gradient $partial cal(L) \/ partial bold(W)$. In a feedforward neural network, back-propagation computes this by applying the chain rule layer by layer from the output back to the input. In a _recurrent_ system like our spiking network --- where the state at each timestep depends on the state at the previous timestep --- we must unroll the computation through time and apply the chain rule across all $T$ timesteps. This is _back-propagation through time_ (BPTT).

Concretely, the forward pass builds a computation graph that spans $T$ timesteps. At each step $t$, the voltage $V(t)$ depends on $V(t-1)$, the conductances $g_e(t)$ and $g_i(t)$ depend on their previous values and any incoming spikes, and the spikes $bold(s)(t)$ depend on $V(t)$. The loss at the end depends on all the output spikes and voltages. BPTT traverses this graph in reverse, accumulating gradients from timestep $T$ back to timestep 1.

To make the gradient computation precise, we introduce compact notation. Collect each neuron's dynamical variables into a state vector:

$
  bold(h)(t) = (V(t), thin g_e (t), thin g_i (t))
$

The forward pass is then a recurrence $bold(h)(t+1) = f(bold(h)(t), bold(s)(t), bold(W))$, where the components expand to the update rules used in our simulator:

$
  V(t+1) &= V(t) + (Delta t) / C_m [-g_L (V - E_L) + g_e (E_e - V) + g_i (E_i - V) + I_"ext"] \
  g_e (t+1) &= g_e (t) dot e^(-Delta t \/ tau_"AMPA") + bold(s)_E (t - d)^top bold(W) \
  g_i (t+1) &= g_i (t) dot e^(-Delta t \/ tau_"GABA") + bold(s)_I (t - d)^top bold(W)_("IE")
$ <eq:state-update>

BPTT traverses this recurrence in reverse. Define the _adjoint_ (error signal) $bold(delta)(t) = partial cal(L) \/ partial bold(h)(t)$. Applying the chain rule one step at a time gives the adjoint recursion:

$
  bold(delta)(t) = (partial ell(t)) / (partial bold(h)(t)) + bold(delta)(t+1) dot bold(J)(t)
$ <eq:adjoint>

where $bold(J)(t) = partial bold(h)(t+1) \/ partial bold(h)(t)$ is the one-step Jacobian and $ell(t)$ is any per-timestep loss contribution. The key Jacobian entries are:

$
  (partial V(t+1)) / (partial V(t)) &= 1 + (Delta t) / C_m (-g_L - g_e - g_i) + "surrogate spike term" \
  (partial V(t+1)) / (partial g_e (t)) &= (Delta t) / C_m (E_e - V)
$ <eq:jacobian-entries>

The second entry is the excitatory driving force $(E_e - V) approx 65 "mV"$, which acts as a natural gain factor that amplifies the gradient signal flowing from voltage back through the conductance pathway. This structure is what makes the surrogate Jacobian approach (Section 3.3) effective.

Finally, the gradient of the loss with respect to a weight $W_(i j)$ accumulates contributions from every timestep, expressed in terms of the adjoint:

$
  (partial cal(L)) / (partial W_(i j)) = sum_(t=1)^T delta_(g_e)^j (t) dot s_i (t - d)
$ <eq:bptt-grad>

where $delta_(g_e)^j (t)$ is the conductance component of the adjoint for neuron $j$, $d$ is the synaptic delay, and $s_i(t-d)$ is the presynaptic spike delivered at time $t$.

In practice, PyTorch handles this automatically: the forward pass records all operations in a dynamic computation graph, and calling `loss.backward()` traverses the graph to compute all weight gradients via BPTT.

== The dead neuron problem: surrogate gradients

There is a fundamental obstacle to applying BPTT to spiking networks. The spike mechanism is a hard threshold:

$
  s(t) = cases(1 & "if" V(t) >= V_"th", 0 & "otherwise")
$

This is a step function, and step functions have zero gradient everywhere except at the threshold, where the gradient is undefined. This means that the chain rule produces $partial s \/ partial V = 0$ for almost all values of $V$, effectively blocking gradient flow through every spike event. No gradient reaches the weights, and learning cannot proceed. This is the _dead neuron problem_.

The solution, widely used in the spiking network literature, is to use a _surrogate gradient_: we keep the hard threshold in the forward pass (so the network still produces binary spikes), but replace its derivative in the backward pass with a smooth approximation. In our implementation, we use the _fast sigmoid_ surrogate:

$
  (partial s) / (partial V) approx 1 / (1 + |V - V_"th"|)^2
$ <eq:surrogate>

This function is peaked at the threshold $V_"th"$ and decays smoothly on either side. When a neuron's voltage is close to threshold, the surrogate gradient is large, signalling that a small change in voltage would likely change whether the neuron fires. When the voltage is far from threshold, the gradient is small, reflecting the fact that the spike decision is insensitive to small perturbations. This is a reasonable approximation: it preserves the qualitative structure of the true gradient (sensitive near threshold, insensitive far away) while providing the smooth, non-zero gradients that BPTT requires.

With surrogate gradients, the full back-propagation chain becomes: loss $arrow.r$ readout $arrow.r$ output spikes (surrogate) $arrow.r$ output voltages $arrow.r$ conductances $arrow.r$ hidden spikes (surrogate) $arrow.r$ hidden voltages $arrow.r$ $dots$ $arrow.r$ weights.

== Surrogate Jacobian scaling

Even with surrogate gradients enabling gradient flow, training conductance-based spiking networks presents an additional challenge: the gradients can be extremely large. The source of the problem is visible in the Jacobian entries (@eq:jacobian-entries).

Consider the voltage row of the one-step Jacobian $bold(J)(t)$. The entries that govern how the adjoint $bold(delta)(t)$ propagates through the voltage state are:

$
  bold(J)_V (t) = mat(
    underbrace(1 + (Delta t) / C_m (-g_L - g_e - g_i), approx 0.99),
    underbrace((Delta t) / C_m (E_e - V), approx 16),
    underbrace((Delta t) / C_m (E_i - V), approx -2.5)
  )
$ <eq:jacobian-row>

The scale mismatch is stark. Plugging in typical resting-state values ($Delta t = 0.25$ ms, $C_m = 1$ nF, $V approx -65$ mV, $E_e = 0$ mV, $E_i = -75$ mV):

$
  (partial V(t+1)) / (partial V(t)) approx 0.99, quad quad (partial V(t+1)) / (partial g_e (t)) = (0.25 times 65) / 1 = 16.25
$

The off-diagonal $g_e$ entry is over 16$times$ larger than the diagonal. In the adjoint recursion (@eq:adjoint), $bold(delta)(t) = dots + bold(delta)(t+1) dot bold(J)(t)$, this amplification compounds at every timestep. Over $T = 800$ steps, even modest adjoint signals flowing through the voltage$arrow.r$conductance pathway can grow by many orders of magnitude.

We address this with a _surrogate Jacobian_ technique. The idea is analogous to the surrogate gradient for spikes: we leave the forward dynamics unchanged but modify the backward pass to produce better-behaved gradients. Specifically, we scale the gradient of $d V \/ d t$ by $1 \/ xi$ (the `cm_backward_scale` parameter), as if the membrane capacitance were larger by a factor $xi$:

$
  "forward:" quad (d V)/(d t) = (dots) / C_m, quad quad "backward:" quad (partial cal(L))/(partial (d V \/ d t)) arrow.r 1/xi dot (partial cal(L))/(partial (d V \/ d t))
$ <eq:cm-scale>

This uniformly scales the voltage row of the Jacobian:

$
  tilde(bold(J))_V (t) = 1/xi dot bold(J)_V (t)
$ <eq:scaled-jacobian>

With $xi = 100$, the effective $partial V \/ partial g_e$ drops from 16.25 to 0.16 --- now comparable to the diagonal entry rather than dominating it. The adjoint recursion (@eq:adjoint) with $tilde(bold(J))$ in place of $bold(J)$ no longer amplifies exponentially through the voltage channel, while the conductance rows of $bold(J)(t)$ (simple exponential decays) remain unscaled.

This is implemented using a gradient scaling trick: $f(x) = x dot xi^(-1) + "detach"(x) dot (1 - xi^(-1))$ evaluates to $x$ in the forward pass but has gradient scaled by $xi^(-1)$ in the backward pass.

Combined with per-parameter gradient clipping (max norm 1.0), the surrogate Jacobian makes training stable even at fine timesteps ($Delta t = 0.25$--$0.5$ ms) where the computation graph spans hundreds of steps.

== Regularisation with PING health loss

The cross-entropy loss alone optimises for classification accuracy, but it places no constraint on the network's oscillatory dynamics. In principle, the network could learn to classify correctly while destroying the PING rhythm --- for example, by silencing the inhibitory population entirely or driving all neurons to fire tonically.

To encourage the network to maintain biologically plausible PING oscillations during and after training, we add the differentiable oscillatory health loss from Section 2 as a regularisation term. The PING loss is computed on the hidden excitatory population $E_"hid"$, which is the layer most directly involved in the E $arrow.r$ I $arrow.r$ E feedback loop:

$
  cal(L) = cal(L)_"CE" + lambda cal(L)_"PING" (bold(S)_(E_"hid"))
$

The weight $lambda$ (typically 0.01) is chosen to be small enough that the PING loss does not interfere with classification learning in early epochs, but large enough to gently steer the network toward oscillatory dynamics as training progresses. In practice, we observe that the CE loss dominates early in training (dropping from $approx 2.3$ to $approx 0.3$ in the first two epochs), after which the PING loss begins to have a meaningful influence on weight updates.

= Results

We trained the PING network on a 20,000-sample subset of MNIST for 10 epochs using the full pipeline described above: Poisson encoding ($kappa = 3.0$), BPTT with surrogate gradients, surrogate Jacobian scaling ($xi = 100$), and PING health regularisation ($lambda = 0.01$, $f_"target" = 30$ Hz). The simulation used $Delta t = 0.25$ ms with a 200 ms presentation window (800 timesteps per sample), yielding a computation graph of 800 steps unrolled through BPTT. Training used the Adam optimiser with a cosine learning rate schedule from $7 times 10^(-5)$ to 0, and gradient clipping at max norm 1.0. The network had 1,236,900 trainable parameters across three weight matrices: $bold(W)_(E E)$ (1,102,500), $bold(W)_(E I)$ (67,200), and $bold(W)_(I E)$ (67,200).

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/graph_light.png", width: 80%),
  caption: [Network architecture. Excitatory populations (red) $E_"in"$ (784), $E_"hid"$ (256), and $E_"out"$ (10) form the feedforward path. A global inhibitory population $I_"global"$ (64, blue) participates in the E $arrow.r$ I $arrow.r$ E PING loop with $E_"hid"$. All synaptic delays are 1.0 ms. Edge annotations show initial weight distribution parameters.],
) <fig-graph>

== Training dynamics

@fig-loss-train shows the training loss decomposed into its two components. The cross-entropy loss drops sharply during the first epoch (iterations 0--313), falling from $approx 2.3$ (chance level for 10 classes) to below 0.5. It continues to decrease steadily, reaching a final training loss of 0.14 by epoch 10. The PING health loss (scaled by $lambda = 0.01$) remains small throughout --- typically $0.01$--$0.05$ --- confirming that the regularisation does not dominate the classification objective. Occasional spikes in the PING loss correspond to batches where the network's oscillatory structure is temporarily disrupted, but these are quickly corrected.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/loss_train_light.png", width: 55%),
  caption: [Training loss breakdown over 3,130 iterations (10 epochs). Top: total loss. Middle: cross-entropy loss. Bottom: PING health loss scaled by $lambda = 0.01$.],
) <fig-loss-train>

The test loss (@fig-loss-test) shows a smooth, monotonic decrease from 2.23 after epoch 1 to 0.23 after epoch 10, with no sign of overfitting. The train--test gap is small (0.14 vs 0.23), suggesting the network generalises well despite the stochastic Poisson encoding providing natural regularisation.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/loss_test_light.png", width: 55%),
  caption: [Test cross-entropy loss per epoch. The smooth decline with no uptick indicates stable generalisation.],
) <fig-loss-test>

Training accuracy (@fig-accuracy) rises rapidly from chance ($approx 10%$) to $approx 90%$ within the first 500 iterations, then plateaus in the 90--100% range for the remainder of training. The high batch-to-batch variance is expected: each batch contains only 64 samples, and the stochastic Poisson encoding means the same image produces different spike trains on each presentation. The final test accuracy is *93.5%*.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/accuracy_light.png", width: 55%),
  caption: [Per-batch training accuracy over 10 epochs. The rapid rise and high-variance plateau reflect the stochastic Poisson encoding.],
) <fig-accuracy>

== Classification performance

The confusion matrix (@fig-confusion) shows strong diagonal dominance across all 10 digit classes. The best-recognised digits are 1 (98.6%) and 0 (96.7%), which have distinctive visual structure. The most confused digits are 7 (89.3%) and 5 (89.9%), with 7 frequently misclassified as 9 (21 samples) and 2 (12 samples) --- visually similar digits whose distinguishing features (the crossbar of 7, the curves of 2) may be difficult to resolve through Poisson-encoded spike trains.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/confusion_light.png", width: 55%),
  caption: [Confusion matrix on the 5,000-sample test set. Strong diagonal indicates good performance across all classes. Notable off-diagonal entries: 7$arrow.r$9 (21) and 4$arrow.r$9 (19).],
) <fig-confusion>

== Network dynamics

The firing rate trajectories (@fig-firing-rates) reveal how the network self-organises during training. The input layer $E_"in"$ maintains a stable rate of $approx 15$ Hz, set by the Poisson encoding and input scale. The hidden layer $E_"hid"$ undergoes a transient increase early in training (peaking at $approx 12$ Hz during the first epoch as weights are rapidly adjusted), then settles to $approx 7$--8 Hz --- a biologically plausible rate that the PING regulariser helps maintain. The output layer $E_"out"$ shows the most dramatic evolution: it initially fires at $approx 17$ Hz, drops to $approx 8$ Hz as the network learns selective responses, then recovers to $approx 12$ Hz as the winning output neurons learn to fire reliably for their corresponding digit classes.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/firing_rates_light.png", width: 55%),
  caption: [Mean firing rate per excitatory layer across training. $E_"in"$ remains stable; $E_"hid"$ settles to $approx 7$--8 Hz; $E_"out"$ shows a characteristic dip-and-recovery pattern.],
) <fig-firing-rates>

The gradient norm history (@fig-grad-norms) illustrates the challenge of training conductance-based spiking networks. During the first 600 iterations, gradient norms exhibit large transient spikes --- up to $1.6 times 10^6$ for $bold(W)_(E E)$, $2.5 times 10^4$ for $bold(W)_(E I)$, and $3.3 times 10^3$ for $bold(W)_(I E)$. These correspond to epochs where the network dynamics are still unstable and small weight changes produce large shifts in spike timing. The surrogate Jacobian scaling ($xi = 100$) and gradient clipping (max norm 1.0) prevent these spikes from destabilising training. After iteration $approx 1000$, gradient norms settle to near zero, indicating that the network has found a stable operating regime.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/grad_norms_light.png", width: 55%),
  caption: [Gradient norms per weight matrix. Large transient spikes in the first 600 iterations are tamed by surrogate Jacobian scaling and gradient clipping; the network stabilises thereafter.],
) <fig-grad-norms>

== Spiking activity and PING oscillations

@fig-raster-3 shows the full spike raster for a digit 3 presentation after training. The input layer $E_"in"$ (blue, top) fires continuously in a pattern reflecting the pixel intensities of the digit. Below, the hidden layer $E_"hid"$ shows clear rhythmic bursting: clusters of spikes appear at regular intervals of $approx 30$--40 ms, consistent with a PING oscillation in the gamma band. The inhibitory population $I_"global"$ (red, bottom) fires in short, synchronised bursts that follow each $E_"hid"$ burst by a few milliseconds --- the hallmark E$arrow.r$I$arrow.r$E PING cycle. The output layer $E_"out"$ shows sparse but selective firing, with the correct output neuron producing more spikes than the others.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_03_light.png", width: 55%),
  caption: [Spike raster for digit 3 across all layers. $E_"in"$ (blue) provides continuous input. $E_"hid"$ shows rhythmic bursting at $approx 30$ Hz. $I_"global"$ (red) fires in synchronised bursts following each $E_"hid"$ burst, maintaining the PING rhythm.],
) <fig-raster-3>

The output neuron voltage traces (@fig-voltage-3) provide a complementary view. Each of the 10 output neurons (one per class) is plotted over the 200 ms simulation window. The voltage dynamics show clear oscillatory modulation driven by the PING rhythm in the hidden layer. Neuron 3 (the correct class) repeatedly reaches threshold and fires, while other neurons are driven below rest by inhibition or remain subthreshold. This separation between the winning neuron's voltage trajectory and the others is what the hybrid readout (@eq:readout) converts into a classification decision.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/voltage_output_digit_03_light.png", width: 55%),
  caption: [Output neuron membrane voltages for digit 3. Neuron 3 (correct class) repeatedly reaches threshold ($-50$ mV, dashed red), while other neurons remain mostly subthreshold. The oscillatory envelope reflects the PING rhythm from the hidden layer.],
) <fig-voltage-3>

@fig-raster-1 shows the raster for digit 1, which has far fewer active input pixels. Despite the sparse input ($approx 93{,}000$ input spikes vs $approx 334{,}000$ for digit 3), the network still produces clear PING oscillations in $E_"hid"$ and $I_"global"$, and the correct output neuron fires selectively. This demonstrates that the PING rhythm is robust to variations in input drive. The complete set of spike rasters for all 10 digit classes is provided in @appendix-rasters.

#figure(
  image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_01_light.png", width: 55%),
  caption: [Spike raster for digit 1. Despite the sparser input (digit 1 has fewer lit pixels), the PING oscillation in $E_"hid"$ and $I_"global"$ remains intact.],
) <fig-raster-1>

== Summary

The trained PING network achieves 93.5% test accuracy on MNIST while maintaining biologically plausible gamma-band oscillations. The key techniques that enabled training --- surrogate gradients, surrogate Jacobian scaling, and PING health regularisation --- work together: surrogate gradients allow gradient flow through spikes, the surrogate Jacobian tames the large driving-force gradients in conductance-based models, and the PING regulariser gently steers the network toward oscillatory dynamics without interfering with classification learning. The network operates at physiologically reasonable firing rates (7--15 Hz) and exhibits the characteristic E$arrow.r$I$arrow.r$E PING cycle throughout the simulation window.

// ── Appendix ────────────────────────────────────────────────────────
#pagebreak()
#set heading(numbering: "A.1.")
#counter(heading).update(0)

= Spike rasters for all digit classes <appendix-rasters>

The following pages show the full spike rasters for each of the 10 MNIST digit classes (0--9) after training. Each panel shows the activity across all four populations: $E_"in"$ (blue), $E_"hid"$ (black dots), $E_"out"$ (black dots), and $I_"global"$ (red). The PING oscillation --- rhythmic alternation of $E_"hid"$ bursts followed by $I_"global"$ bursts --- is visible in every case, regardless of input density.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_00_light.png"),
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_01_light.png"),
  ),
  caption: [Spike rasters for digits 0 and 1.],
) <fig-rasters-0-1>

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_02_light.png"),
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_03_light.png"),
  ),
  caption: [Spike rasters for digits 2 and 3.],
) <fig-rasters-2-3>

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_04_light.png"),
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_05_light.png"),
  ),
  caption: [Spike rasters for digits 4 and 5.],
) <fig-rasters-4-5>

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_06_light.png"),
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_07_light.png"),
  ),
  caption: [Spike rasters for digits 6 and 7.],
) <fig-rasters-6-7>

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_08_light.png"),
    image("../_artifacts/study.19-ping-regulariser-on-mnist/raster_layers_digit_09_light.png"),
  ),
  caption: [Spike rasters for digits 8 and 9.],
) <fig-rasters-8-9>