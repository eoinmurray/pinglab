#let meta = (
  title: "Does weight decay bound the free recurrence?",
  date: "2026-07-11",
  description: "The SHD program's third stability probe: with Dale's law shown (exp062) to stabilise the net at an accuracy cost, sweep AdamW weight decay on the free signed net to ask whether decay alone can bound the runaway W_ee and remove the NaN — stabilising WITHOUT the constraint's accuracy tax.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp063/numbers.json")
#let cells = r.cells.sorted(key: c => c.wd)   // weak → strong decay
#let weakest = cells.first()
#let strongest = cells.last()

#let results-caption = [_What we see_ is written in the Reading section below.]
#let reading-body = [_(finalised once the run's numbers are in.)_]

#let body = [
  == What this checks

  The program now has a stable recipe — #link("/exp062/")[exp062] showed Dale's
  law trains the net NaN-free where the free signed net diverges — but it costs a
  few points of accuracy (53.2% vs the free net's 56.8%), the price of denying the
  signed recurrence its expressivity. So the live question is whether stability
  can be had *without* that tax: can a plain regulariser bound the free net's
  runaway dynamics instead of a hard constraint?

  This sweeps the AdamW weight decay $lambda in {0, 10^(-3), 10^(-2), 10^(-1)}$ on
  the *free* net (_--no-dales-law_), everything else the exp060 recipe at
  Δt = 1.0 ms where it diverges, single seed, full scale. It measures the plan's
  registered contrast — max recurrent-weight norm and NaN-epoch rate vs decay
  strength. A first pass at $lambda = 10^(-3)$ tamed the gradient explosion but
  did not bound W_ee or remove the NaN, so the sweep reaches up to $10^(-1)$.

  #quote(block: true)[
    *Kill criterion.* If even the strongest decay leaves W_ee growing and NaN
    present, decay is not part of the recipe — it regularises but does not
    stabilise.
  ]

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet, 256 hidden, all four recurrent blocks trainable, signed (no Dale's law)],
    [Swept variable], [AdamW weight decay $lambda in {0, 10^(-3), 10^(-2), 10^(-1)}$],
    [Integration], [$Delta t = #(r.dt_ms)$ ms, $T = 1000$ ms — the setting where the free net diverges],
    [Seeds], [1 (seed #r.seed) — a quick sweep],
    [Training], [#r.epochs epochs, lr 0.001, batch 32, surrogate-gradient BPTT],
    [Regulariser], [upper firing-rate bound ($theta_u = 100$, $s_u = 0.06$)],
    [Compute], [#r.compute],
  )

  Weight decay is the *only* thing that varies; everything else is the free net's
  diverging recipe, so any change in the stability metrics is attributable to λ
  alone.

  == Compute

  RunPod fan-out, one pod per λ, results collected off the shared network volume
  over its S3 HTTPS API (the collector introduced in #link("/exp061/")[exp061]).

  == Results

  #figure(
    image("/artifacts/data/exp063/decay_sweep.svg", width: 100%,
      alt: "Three panels against weight decay λ: NaN-epoch rate, max recurrent-weight norm, and best test accuracy."),
    caption: [
      The stability metrics against weight decay λ (weak → strong). *What we
      expect if decay stabilises.* The NaN-epoch rate and the runaway weight norm
      fall as λ grows, ideally reaching zero NaN while accuracy holds above Dale's
      law. #results-caption
    ],
  )

  == Reading

  #reading-body
]
