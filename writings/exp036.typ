#let meta = (
  title: "W^EI sets the cliff; W^IE shapes the basin",
  date: "2026-05-30",
  description: "Maps the (wEI, wIE) coupling plane: wEI moves the recruitment cliff, wIE shapes the basin above it, and trained networks land in two distinct basins.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)


#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  Maps the (wEI, wIE) coupling plane: which weight controls the recruitment cliff,
  and where do _trained_ networks land when initialised across it? _wEI_ moves the
  cliff (E→I drive is the recruitment knob); _wIE_ shapes the basin above engagement
  but does not move the cliff. Training from each grid point lands in two distinct
  basins — a low-E PING corner and a high-E silent-I stretched-COBA corner.

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.1 ms],
    [Trial duration $T$], [200 ms],
    [MNIST samples (80/20 stratified split of 2000)],
    [1600 train / 400 test (≈ 2.9% of the 70k-sample MNIST corpus)],
    [Epochs], [10],
  )

  PING and COBA baseline definitions, training recipe, and the spike-budget
  regulariser are in #link("/exp025/")[exp025]. The recruitment cliff at $f^*$ is
  introduced in #link("/exp025/")[exp025]. This entry asks: which part of the E↔I
  coupling architecture controls the cliff's position, and what happens when we
  train the network from coupling initialisations spanning both sides of it? Three
  experiments.

  *Coupling sweep (inference-time).* If $f^*$ is set by the E↔I coupling, altering
  $W^(E I)$ or $W^(I E)$ at inference should move it. On the trained PING baseline
  ($theta_u =$ off), multiply either $W^(E I)$ or $W^(I E)$ by a scalar in
  ${0.25, 0.5, 1.0, 2.0}$ and re-run the $W_"in"$ scale sweep on top. All other
  weights frozen.

  *Training grid.* A stronger test is to train one network per grid point with
  $W^(E I)$ and $W^(I E)$ initialised to specified values from scratch, under heavy
  spike penalty: do the resulting solutions cluster into a PING-like region and a
  COBA-like region? 5×5 grid over $W^(E I), W^(I E) in {0, 0.25, 0.5, 1.0, 2.0}$
  (mean initialisation, std fixed at $0.1 times$ mean). One PING-architecture
  training run per cell at $theta_u = 0.2$ from epoch 0, $W_"in" = 0.6$ (a
  compromise between COBA's 0.3 init and PING's 1.2 — gives the no-loop cells a fair
  shot at converging), *100 training epochs* per cell, seed 42. All other
  hyperparameters from the standard PING recipe. 25 trainings on Modal A100 in
  parallel. Reporting per-cell best-epoch accuracy.

  *Diagonal slice.* The grid varies $W^(E I)$ and $W^(I E)$ independently, but the
  codebase's standard PING recipe ties them via $W^(I E) = "ei_ratio" dot W^(E I)$
  with $"ei_ratio" = 2$. Restricting to that line gives a finer-grained view: one
  knob ($W^(E I)$) at 10 values ${0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0}$,
  $W^(I E)$ tracking at twice the value, *three seeds (42, 43, 44) per value, 100
  training epochs*. Best-epoch accuracy reported. Same recipe as the grid otherwise
  ($theta_u = 0.2$, $W_"in" = 0.6$).

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Inference-time $W_"in"$ scale sweep on trained PING ($theta_u =$ off) with
      either $W^(E I)$ or $W^(I E)$ scaled. Top: accuracy. Bottom: I rate. Left:
      $W^(E I)$ sweep, $W^(I E)$ fixed. Right: $W^(I E)$ sweep, $W^(E I)$ fixed.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      Each dot is one cell of the 5×5 $(W^(E I), W^(I E))$ training grid, plotted at
      its best-epoch test accuracy vs final hidden-E rate. Colour encodes the third
      quantity — hidden-I rate — which is what discriminates the clusters. Labels
      are $(W^(E I), W^(I E))$.
    ],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [
      $W^(E I)$ scanned over ${0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0}$
      with $W^(I E) = 2 W^(E I)$. Error bars are mean ± std of the best-epoch test
      accuracy over 3 seeds (42, 43, 44), 100 training epochs each; faint dots show
      individual seed peaks. *Left:* test accuracy. *Middle:* hidden-E rate
      (last-epoch). *Right:* hidden-I rate (last-epoch). Grey dotted vertical at
      $W^(E I) = 1$ marks the canonical recipe baseline.
    ],
  )
]
