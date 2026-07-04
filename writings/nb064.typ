#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "PING needs gradient dampening, COBA does not",
  date: "2026-07-02",
  description: "Gradient dampening is specifically what the recurrent E-I-E loop needs to train; the feedforward COBA control does not need it at all.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let run = json("/artifacts/data/nb064/numbers.json")

#let body = [
  == Introduction

  Surrogate-gradient training of the spiking network applies a voltage-gradient dampening factor (the _--v-grad-dampen_ knob; theory in #link("/ar006/")[ar006]) that scales down the gradient flowing through the leaky-integrate-and-fire membrane update. It is easy to read this as a generic BPTT stabiliser. This entry tests a sharper claim: dampening is specifically what the recurrent E→I→E loop needs, and the feedforward control does not need it at all.

  Concretely — with $d$ the dampening factor (larger = more dampening; $d = 1$ is none):

  - *COBA* (loop off, $alpha_(E I) = 0$) trains to the same accuracy at $d = 1$ as at the canonical $d = 1000$.
  - *PING* (loop on, $alpha_(E I) = 1$) fails to train at $d = 1$ — its backpropagated gradient explodes through the inhibitory feedback and accuracy sits at chance — and only trains once dampening is applied.

  If both halves hold, dampening is not a global trick but a loop-specific requirement, and the manuscript's §3-para-1 mechanism claim can be stated as a controlled result rather than a caveat.

  == Method

  *Grid.* Two architectures (COBA, PING) × a dampening ladder $d in {1, 10, 100, 1000}$ × one seed = 8 networks. Each cell is otherwise the canonical #link("/nb025/")[nb025] recipe for its architecture; only $alpha_(E I)$ (loop on/off, plus its matched $w_"in"$ / readout scale) and the swept $d$ differ. One seed suffices — the effect is a qualitative dissociation (chance vs converged), not an effect size. These cells are bespoke to this validation — the low-$d$ PING cells are deliberately unstable — so they are trained in this runner rather than in the #link("/nb022/")[nb022] shared hub.

  *Run parameters.* Epochs are fixed (a convergence control, not a budget dial); the data budget is the runner's own declared scale. The scale reported is the one that actually produced the figures. Architectures: COBA (loop off) · PING (loop on); dampening ladder $d$: 1, 10, 100, 1000; seed 42; optimiser Adam, lr $4 times 10^(-4)$; mem-mean readout.

  *Readouts.* For each cell the runner reads best test accuracy (did it learn) and the per-epoch gradient norm (did BPTT stay bounded) from _metrics.json_. The predicted signature is a double dissociation: COBA's accuracy flat across the ladder while PING's rises from chance, and COBA's gradient norm bounded while PING's diverges at low $d$.

  All network construction, training and metric logging happen in _cli.py_; the notebook only shells out and reads artifacts.

  == Results

  #figure(
    image("/artifacts/data/nb064/accuracy_vs_dampen.svg", width: 100%),
    caption: [Best accuracy across the dampening ladder. COBA is flat — it trains at $d = 1$ as well as at $d = 1000$. PING climbs from chance, needing dampening to train.],
  )

  #figure(
    image("/artifacts/data/nb064/gradnorm_vs_dampen.svg", width: 100%),
    caption: [The mechanism: at low dampening PING's gradient norm blows up through the E→I→E loop, while COBA's stays bounded — which is why only PING's accuracy collapses.],
  )

  == Discussion

  If the dissociation holds, the E→I→E loop is the source of the training instability, not the spiking nonlinearity or depth per se — COBA shares both yet trains at $d = 1$. Dampening is therefore the price of the recurrent inhibitory feedback, and the manuscript can state the loop, not surrogate gradients in general, as what requires it.

  == Success criteria

  - *COBA does not need dampening*: COBA best accuracy at $d = 1$ is within 5 pp of its best across the ladder.
  - *PING needs dampening*: PING best accuracy at $d = 1$ is within 15 pp of chance and more than 5 pp below PING at $d = 1000$.

  Both are computed in the runner (_success_ block of numbers.json); the claim is validated only if both hold.

  == Next steps

  If validated, fold the result into the ar010 item-1 write-up and upgrade the §3-para-1 manuscript claim from "is consistent with" to a controlled statement. If PING trains at $d = 1$ after all (claim fails), the dampening requirement is not loop-specific and the manuscript framing stays as-is.
]
