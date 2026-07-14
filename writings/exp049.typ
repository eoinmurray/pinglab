#import "/.demolab/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Gradient descent does not preserve a trainable PING loop",
  date: "2026-06-09",
  description: "Unfreeze the recurrent conductances and Adam does not preserve or recover effective E-to-I recruitment; every tested initialisation moves toward dense E firing and weak I activity.",
  collection: "gamma-gated-sparsity",
  status: "final",
)


#let body = [
  The trained networks this entry uses are produced once in the shared training hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  #link("/exp025/")[exp025] freezes the recurrent conductances $W^(E I), W^(I E)$ at biophysical values. When they are trainable, Adam does _not_ preserve or recover effective PING from any tested initialisation (canonical, zero, or 10% of canonical): E→I recruitment weakens or remains absent, inhibitory firing collapses, and the network moves toward dense E firing at ≈ 90% accuracy, close to the frozen PING control (≈ 91%). The matrices store non-negative conductance magnitudes; inhibition is produced by the GABA reversal potential, not by a negative $W^(I E)$. PING is therefore a structural prior imposed by the frozen loop in this setup, not one gradient descent recovers on its own.

  == Method

  *Architecture.* $N_E = 1024$ excitatory, $N_I = 256$ inhibitory, mem-mean readout, Dale's law enforced. Hyperparameters match the #link("/exp025/")[exp025] PING baseline: Adam at lr $4 times 10^(-4)$, batch 256, surrogate slope 1, $W_"in" tilde cal(N)(1.2, 0.12)$ at 95% sparsity, gradient norm clipped to 1.0, $Delta t = 0.1$ ms, $T = 200$ ms, no firing-rate regulariser.

  Both recurrent matrices are non-negative conductance magnitudes. After each optimiser step they are projected onto the non-negative cone. Their physiological sign is supplied by the pathway reversal potential: I→E contributes $g_I (E_I - V)$ with $E_I = -80$ mV and is therefore inhibitory despite $W^(I E) >= 0$.

  *Sweep.* Four conditions × three seeds (42, 43, 44) on the 10% MNIST subset (5600 train / 1400 test), 50 epochs. Only the initial $(W^(E I), W^(I E))$ and the trainable-or-not flag vary:

  #table(
    columns: 3,
    [Condition], [$W^(E I), W^(I E)$ init], [Trainable?],
    [_frozen_ping_ (control)], [canonical biophysical], [no],
    [_trainable_ping_init_], [canonical biophysical], [*yes*],
    [_trainable_zero_init_], [$0$ (COBA-equivalent start)], [*yes*],
    [_trainable_small_init_], [$0.1 times$ canonical], [*yes*],
  )

  Canonical biophysical means $W^(E I) tilde cal(N)(1.0, 0.1)$ μS, $W^(I E) tilde cal(N)(2.0, 0.2)$ μS at $N_I = 256$, fan-in-normalised, so the trainer reports per-edge means of ≈ 0.0010 and ≈ 0.0078. "PING is on" means the inhibitory loop is active: E recruits I, and I paces a gamma rhythm through GABA conductance. "The loop is lost" means E→I recruitment is too weak to sustain that regime, I activity is low or absent, and E fires densely. The firing rates and rhythmicity are the cleanest read of which regime a trained network reaches.

  == Results

  The whole sweep collapses to one picture. In the (E firing rate, I firing rate) plane, a network "found PING" if it sits in the low-E / high-I corner, and "lost the loop" if it sits in the dense-E / silent-I corner.

  #figure(
    image("/artifacts/data/exp049/attractor_ei.svg", width: 100%,
      alt: "Scatter of trained networks in the E-rate / I-rate plane: the frozen control sits alone at low E and high I, every trainable condition sits along the silent-I floor at high E."),
    caption: [Each point is one trained network (4 conditions × 3 seeds). The *frozen control* (grey) sits in the PING corner, E ≈ 10 Hz and I ≈ 57 Hz, the inhibitory loop alive. *Every trainable condition* (started at canonical PING, at zero, or at 0.1× canonical) collapses to the dense-E / silent-I floor, E ≈ 40–75 Hz and I ≈ 0–7 Hz, at ≈ 90% accuracy, essentially matching the ≈ 91% control. Where the loop _starts_ makes no difference; only whether it is allowed to train. Gradient descent never reaches PING, and dropping it costs no accuracy.],
  )

  == Training curves — the loop pruned, epoch by epoch

  #figure(
    image("/artifacts/data/exp049/training_curves.svg", width: 100%,
      alt: "Four per-epoch panels (accuracy, E rate, I rate, pingness) for the four conditions; the frozen control keeps a low E rate, high I rate, and high pingness while every trainable run does the opposite."),
    caption: [Per-epoch metrics from the runs themselves, coloured by init: PING init (black), zero init (red), small-seed init (amber), frozen-PING control (grey dashed); 10% of MNIST, 50 epochs, 3 seeds each. *Accuracy*: all reach ≈ 85–90% and track each other; pruning the loop costs nothing. *E rate*: the trainable runs climb to ≈ 37–75 Hz as the loop releases the excitatory population, while the frozen control stays gated near 10 Hz. *I rate*: every trainable run drops to ≈ 0 within a few epochs (the inhibitory loop is pruned), while the frozen control's I rate _rises_ to ≈ 57 Hz as its readout trains. *Pingness*: the exp054 lobe–trough contrast holds near ≈ 0.99 for the frozen control while every trainable init collapses to ≈ 0.1–0.2. The residual floor is the metric's known low-rate inflation under shared input (#link("/exp054/")[exp054]), not a surviving rhythm: the frozen-vs-trainable gap is the signal. The loop dies early, from every start, with no accuracy penalty.],
  )

  == Phase portrait — there is only one attractor under training

  The four per-epoch panels above tell the story but separate the two state variables that matter. Putting _E rate_ and _pingness_ on perpendicular axes shows the trajectories of training directly, and the geometry rules out a misreading: it is _not_ the case that PING and COBA are two attractors of the same dynamics, with the architecture deciding the basin. Under training there is *one* attractor (the COBA corner), and the PING state only persists because freezing the loop zeroes its gradients.

  #figure(
    image("/artifacts/data/exp049/phase_portrait.svg", width: 100%,
      alt: "Training trajectories in the E-rate / pingness plane: the frozen control stays in the high-pingness corner while every trainable run sits on the pingness ≈ 0.1 floor."),
    caption: [Each trainable trajectory is the per-epoch mean across 3 seeds, alpha-ramped along the epoch axis so the direction of flow reads off the line itself; open markers are epoch 1 (the first logged epoch), filled markers are epoch 50. *Frozen PING (grey)*: sits in the upper PING basin from start to finish at pingness ≈ 0.98; drifts a little in E rate (≈ 4 → 10 Hz) as the feedforward and readout weights train, but the loop weights themselves can't move by construction. *Trainable PING init (black)*: was initialised with canonical biophysical loop weights, but by the time the first metric is logged (after epoch 1) the loop has already been pruned to pingness ≈ 0.17. The trajectory then walks rightward along the COBA floor with the others. *Trainable zero init (red)* and *small-seed init (amber)*: start at the floor and stay there, E rate climbing as the feedforward layer learns the task. Every legend entry lands at 83–91% accuracy, so the move costs nothing. The reading is sharper than the time-series panels suggest: the empty band between pingness ≈ 0.2 and ≈ 0.85 is the _whole story_, because gradient descent flies through it within one epoch and the metric never catches it mid-flight. There is no separatrix between PING and COBA under training because there is no slow trajectory through the gap. The freeze isn't preventing slow erosion; it's preventing instant collapse.],
  )

  == Accuracy–rate trajectory — same destination, different spike economy

  Figure 3 puts pingness on its own axis. Putting pingness on the _colour_ axis instead and replaying training as (E rate, accuracy) trajectories (the #link("/exp025/")[exp025] accuracy–rate frontier given a time axis) gives the third reading of the same data: every condition reaches the same final test accuracy, but along very different routes, and only one of them is doing PING while it gets there.

  #figure(
    image("/artifacts/data/exp049/acc_rate_trajectory.svg", width: 100%,
      alt: "Test accuracy versus E rate trajectories coloured by pingness: the frozen control climbs in the low-rate PING basin, every trainable run ends far right in the COBA attractor at the same accuracy."),
    caption: [Each trajectory is the per-epoch mean across 3 seeds; per-segment colour is that epoch's pingness on a viridis 0→1 scale (colourbar at right). Open markers are epoch 1, filled markers are epoch 50. The frontier _with time_ tells three things at once. *Same destination*: all four conditions land at ≈ 83–91% accuracy by epoch 50, regardless of init or whether the loop trains. *Different routes*: the frozen control climbs accuracy almost vertically with little change in E rate (≈ 4 → 10 Hz); the trainable conditions all bend right and up, the biggest one (zero init) reaching final accuracy at E ≈ 75 Hz, ≈ 8× the frozen control's rate. *Only one of them is still PING*: the frozen control's line is bright yellow because pingness stays high (≈ 1.0) the whole way; every trainable line is dark purple because pingness was already pruned to ≈ 0.1 by the first logged epoch. Reading the colour bar: most of pingness 0.2–0.85 is unused space, because no trajectory crosses it slowly enough to be logged. Reading the geometry: this is the manuscript's accuracy–rate frontier (#link("/exp025/")[exp025], §2.3) animated, and the architecture is the only thing keeping a trained network on the sparse, rhythmic side of it. The dashed divider at ≈ 17 Hz marks the two basins explicitly: frozen PING holds the left, every trainable run ends in the right, at the same accuracy.],
  )
]
