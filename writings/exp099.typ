#import "/.demolab/lib.typ": data-image, video, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp099")

#let meta = (
  status: "Results available",
  title: "From simplified to brainlike input in a PING network",
  created_at: "2026-08-26",
  updated_at: "2026-08-28",
  description: "A planned simulation scout comparing a Börgers–Kopell-like input regime with a richer conductance-based background.",
  collection: "demo",
  order: 13,
)

#let inputs = ("exp099",)

#let render-report(data-file) = [
  == Abstract

  We will compare simplified and richer input in the same sparse
  excitatory–inhibitory spiking network. First, we will seek stable
  pyramidal–interneuron network gamma (PING) under stationary drive with independent
  noise. We will then add private and shared conductance fluctuations, cellular
  heterogeneity, correlated afferent spikes, and slow stationary modulation to
  test whether the rhythm persists or becomes intermittent. The current probe
  implements only the richer-input condition: 400 excitatory and 100 inhibitory
  neurons receive a transient afferent bout during a two-second simulation.
  Its working media illustrate the probe; they do not establish the planned
  controlled comparison or identify which input features preserve PING.

  == Results

  === 1. Network structure

  #figure(
    data-image(data-file("exp099/network.svg"), width: 100%),
    caption: [Structural schematic of the excitatory and inhibitory populations,
      recurrent projections, and afferent inputs. This is a model diagram, not evidence.],
    kind: image, supplement: [Figure],
  )

  === 2. Richer-input probe

  #let clip = data-file("exp099/richer-input-ai-to-intermittent-ping.mp4")
  // Missing files in a selected run remain errors, not empty-run placeholders.
  #if clip != none { let _ = read(clip, encoding: none) }
  #figure(
    video(clip),
    caption: [Simulated single-seed working media, not an established comparison.
      Transmission paths are sampled for readability; the raster retains all
      recorded spikes. The view spans
      300–1,800 ms with nonuniform playback pacing. R is autocorrelation contrast;
      L is the conductance-loop score smoothed over 75 ms and normalized within
      this probe, not a probability of PING.],
    kind: image, supplement: [Figure],
  )

  == Methods

  The intended study compares a simplified Börgers–Kopell-like input regime
  with richer background activity, motivated by reciprocal excitatory–inhibitory
  synchronization.#cite(1) The procedure below describes the implemented
  richer-input probe; selecting a stable simplified reference and conducting
  the controlled comparison remain planned.

  + *Construct the recurrent circuit.* Use 400 excitatory and 100 inhibitory
    conductance-based leaky integrate-and-fire neurons with fixed excitatory–excitatory, excitatory–inhibitory,
    inhibitory–excitatory and inhibitory–inhibitory projections. Set 97.5% of
    each recurrent matrix to zero using an exact-count mask; sample the
    remaining weights from lower-clamped normal distributions with respective
    means 0.85, 0.6, 3.0 and 0.4 μS and standard deviations 0.255, 0.18, 0.9 and
    0.12 μS. Excitatory and inhibitory conductance decay times are 2 and 9 ms;
    no training or parameter selection is performed in this probe.

  + *Specify the background and afferents.* Provide shared Poisson afferents
    at 10 Hz and population-private afferents at 15 Hz, with afferent weight
    means 0.08 μS onto excitatory cells and 0.02 μS onto inhibitory cells.
    Add four excitatory/inhibitory background channels with private events at
    500 Hz and grouped shared events at 80 Hz, grouping 25 excitatory or 10
    inhibitory cells. Private/shared event amplitudes are 0.06/0.02 μS for
    excitation onto excitatory cells and 0.03/0.01 μS for the other channels.
    Apply rate and amplitude heterogeneity with mean-one,
    lower-clamped normal multipliers of standard deviation 0.1, plus stationary
    rate modulation with a 250 ms timescale and fractional spread 0.12.

  + *Apply the transient and record activity.* Simulate 2,000 ms at a 0.25 ms
    timestep with random seed 7. Raise the afferent multiplier smoothly from
    baseline at 600 ms to its peak at 850 ms and back by 1,100 ms; peak
    multipliers are 1.2 for private and 6.5 for shared afferents. Retain spikes,
    cell voltages, conductances and executed input events, keeping recurrent
    weights fixed throughout.

  + *Measure temporal organization.* Evaluate excitatory spike-autocorrelation
    lobe–trough contrast in 400 ms windows every 10 ms, using 1 ms lag bins out
    to 100 ms; undefined contrasts are recorded as zero.
    #math.equation(block: true, numbering: "(1)", $R = (a - b) / (a + b)$)
    Here $R$ is dimensionless contrast, $a$ is the smoothed autocorrelogram's
    lobe height before its first trough, and $b$ is that trough's height.
    Measure conductance-loop compactness and directional coherence in 40 ms
    windows every 5 ms over 300–1,800 ms, smooth over 75 ms, and clip the
    10th–95th-percentile normalization to zero–one. Count excitatory and
    inhibitory spikes over the whole probe; the peak-contrast summary uses
    window centres from 200 through 1,790 ms, while the plot also includes
    1,800 ms. These single-seed descriptors do not establish a causal input
    effect or classify a rhythm as PING by themselves.

  == 4. References

  #reference-list((
    (text: [Börgers, C. & Kopell, N. — _Synchronization in Networks of Excitatory
      and Inhibitory Neurons with Sparse, Random Connectivity_. Neural Computation
      15(3), 509–538, 2003.], doi: "10.1162/089976603321192059"),
  ))
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}
