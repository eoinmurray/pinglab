#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Gradient descent prunes PING via Dale's-law clamping",
  date: "2026-06-09",
  description: "Unfreeze the recurrent inhibitory weights and Adam does not rediscover PING — it prunes the loop to a dense-E, silent-I COBA network from every initialisation.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)


#let body = [
  The trained networks this entry uses are produced once in the shared training hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  #link("/exp025/")[exp025] freezes the recurrent inhibitory weights $W^(E I), W^(I E)$ at biophysical values. Unfreeze them and Adam does _not_ rediscover PING: from any initialisation — canonical, zero, or 10% of canonical — training prunes most $W^(E I)$ entries below zero, the Dale's-law forward-pass clamp turns them into structural zeros, and the network collapses to dense E firing with I silent, at ≈ 88% accuracy (≈ 2 pp _above_ the frozen PING control). PING is a structural prior the freeze _imposes_, not one gradient descent recovers on its own.

  == Method

  *Architecture.* $N_E = 1024$ excitatory, $N_I = 256$ inhibitory, mem-mean readout, Dale's law enforced. Hyperparameters match the #link("/exp025/")[exp025] PING baseline: Adam at lr $4 times 10^(-4)$, batch 256, surrogate slope 1, $W_"in" tilde cal(N)(1.2, 0.12)$ at 95% sparsity, gradient norm clipped to 1.0, $Delta t = 0.1$ ms, $T = 200$ ms, no firing-rate regulariser.

  *Sweep.* Four conditions × three seeds (42, 43, 44), medium tier (1600 train / 400 test MNIST, 100 epochs). Only the initial $(W^(E I), W^(I E))$ and the trainable-or-not flag vary:

  #table(
    columns: 3,
    [Condition], [$W^(E I), W^(I E)$ init], [Trainable?],
    [_frozen_ping_ (control)], [canonical biophysical], [no],
    [_trainable_ping_init_], [canonical biophysical], [*yes*],
    [_trainable_zero_init_], [$0$ (COBA-equivalent start)], [*yes*],
    [_trainable_small_init_], [$0.1 times$ canonical], [*yes*],
  )

  Canonical biophysical means $W^(E I) tilde cal(N)(1.0, 0.1)$ μS, $W^(I E) tilde cal(N)(2.0, 0.2)$ μS at $N_I = 256$, fan-in-normalised — so the trainer reports per-edge means of ≈ 0.0010 and ≈ 0.0078. "PING is on" means the inhibitory loop is alive: I fires at a healthy rate and paces a gamma rhythm; "the loop is gone" means I is silenced and E fires densely (plain COBA). The firing rates are the cleanest read of which regime a trained network lands in.

  == Results

  The whole sweep collapses to one picture. In the (E firing rate, I firing rate) plane, a network "found PING" if it sits in the low-E / high-I corner, and "lost the loop" if it sits in the dense-E / silent-I corner.

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Each point is one trained network (4 conditions × 3 seeds). The *frozen control* (grey) sits in the PING corner — E ≈ 8 Hz, I ≈ 38 Hz, the inhibitory loop alive. *Every trainable condition* — started at canonical PING, at zero, or at 0.1× canonical — collapses to the same dense-E / silent-I corner (E ≈ 45 Hz, I ≈ 0), at ≈ 88% accuracy, ≈ 2 pp _above_ the control. Where the loop _starts_ makes no difference; only whether it is allowed to train. Gradient descent never reaches PING, and the task slightly prefers it gone.],
  )

  == Training curves — the loop pruned, epoch by epoch

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Per-epoch metrics from the runs themselves, coloured by init — PING init (black), zero init (red), small-seed init (amber), frozen-PING control (grey dashed); 5% of MNIST, 100 epochs, 3 seeds each. *Accuracy* — both reach ≈88–90% and track each other; pruning the loop costs nothing (the trainable runs even edge slightly above). *E rate* — the trainable runs climb to ≈25–56 Hz as the loop releases the excitatory population, while the frozen control stays gated near 8 Hz. *I rate* — every trainable run drops to ≈0 within a few epochs (the inhibitory loop is pruned), while the frozen control's I rate _rises_ to ≈45 Hz as its readout trains. *Pingness* — the exp054 lobe–trough contrast: the frozen control climbs to ≈0.95 (the rhythm sharpens) while every trainable init collapses to ≈0.1–0.2. The residual floor is the metric's known low-rate inflation under shared input (#link("/exp054/")[exp054]), not a surviving rhythm — the frozen-vs-trainable gap is the signal. The loop dies early, from every start, with no accuracy penalty.],
  )

  == Phase portrait — there is only one attractor under training

  The four per-epoch panels above tell the story but separate the two state variables that matter. Putting _E rate_ and _pingness_ on perpendicular axes shows the trajectories of training directly, and the geometry rules out a misreading: it is _not_ the case that PING and COBA are two attractors of the same dynamics, with the architecture deciding the basin. Under training there is *one* attractor — the COBA corner — and the PING state only persists because freezing the loop zeroes its gradients.

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Each trainable trajectory is the per-epoch mean across 3 seeds, alpha-ramped along the epoch axis so the direction of flow reads off the line itself; open markers are epoch 1 (the first logged epoch), filled markers are epoch 100. *Frozen PING (grey)* — sits in the upper PING basin from start to finish at pingness ≈ 0.86 → 0.94; drifts a little in E rate (≈ 4 → 10 Hz) as the feedforward and readout weights train, but the loop weights themselves can't move by construction. *Trainable PING init (black)* — was initialised with canonical biophysical loop weights, but by the time the first metric is logged (after epoch 1) the loop has already been pruned: pingness ≈ 0.17 from the start. The trajectory then walks rightward along the COBA floor with the others. *Trainable zero init (red)* and *small-seed init (amber)* — start at the floor and stay there, E rate climbing as the feedforward layer learns the task. Every legend entry lands at 87–89% accuracy, so the move costs nothing. The reading is sharper than the time-series panels suggest: the empty band between pingness ≈ 0.2 and ≈ 0.85 is the _whole story_ — gradient descent flies through it within one epoch and the metric never catches it mid-flight. There is no separatrix between PING and COBA under training because there is no slow trajectory through the gap. The freeze isn't preventing slow erosion; it's preventing instant collapse.],
  )

  == Accuracy–rate trajectory — same destination, different spike economy

  Figure 3 puts pingness on its own axis. Putting pingness on the _colour_ axis instead and replaying training as (E rate, accuracy) trajectories — the #link("/exp025/")[exp025] accuracy–rate frontier given a time axis — gives the third reading of the same data: every condition reaches the same final test accuracy, but along very different routes, and only one of them is doing PING while it gets there.

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Each trajectory is the per-epoch mean across 3 seeds; per-segment colour is that epoch's pingness on a viridis 0→1 scale (colourbar at right). Open markers are epoch 1, filled markers are epoch 100. The frontier _with time_ tells three things at once. *Same destination* — all four conditions land at ≈ 87–89% accuracy by epoch 100, regardless of init or whether the loop trains. *Different routes* — the frozen control climbs accuracy almost vertically with little change in E rate (≈ 4 → 10 Hz); the trainable conditions all bend right and up, the biggest one (zero init) reaching the final accuracy at E ≈ 57 Hz, ≈ 6× the frozen control's rate. *Only one of them is still PING* — the frozen control's line is bright yellow because pingness stays high (≈ 0.9) the whole way; every trainable line is dark purple because pingness was already pruned to ≈ 0.1 by the first logged epoch. Reading the colour bar: most of pingness 0.2–0.85 is unused space, because no trajectory crosses it slowly enough to be logged. Reading the geometry: this is the manuscript's accuracy–rate frontier (#link("/exp025/")[exp025], §2.3) animated — and the architecture is the only thing keeping a trained network on the sparse, rhythmic side of it. The dashed divider at ≈ 17 Hz marks the two basins explicitly: frozen PING holds the left, every trainable run ends in the right, at the same accuracy.],
  )
]
