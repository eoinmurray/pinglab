#let meta = (
  title: "Can trained pixel drive move a PING circuit toward irregular background activity?",
  created_at: "2026-08-24",
  updated_at: "2026-08-24",
  description: "A scout replaying one trained canonical PING checkpoint across a paired MNIST pixel-drive sweep.",
  collection: "snnlang",
  status: "ExpScout",
  order: 13,
)

#let transition-video(src) = context {
  if target() == "html" {
    html.elem("video", attrs: (src: src, controls: "", loop: "", muted: "", playsinline: "", style: "max-width:100%;width:100%"))[]
  } else {
    image("/artifacts/data/exp099/trained_pixel_transition_poster.png", width: 100%, alt: "Poster frame from the trained pixel-drive transition video.")
  }
}

#let body = [
  == 1. Abstract

  This exploratory simulation asks whether changing pixel-encoded afferent spike rate can move a trained excitatory–inhibitory circuit between PING and asynchronous-irregular-like background activity. Official run r006 replays one final-epoch canonical PING checkpoint from the gamma-gated-sparsity collection against one fixed MNIST test image. Forty paired input conditions span 0–100 Hz maximum pixel rate.

  The circuit is silent without drive, strongly rhythmic at low nonzero drive, and progressively less rhythmic as drive increases. At 2.56 Hz, E leads I by 1 ms and rhythmicity contrast is 0.923 at a 60 Hz dominant frequency. By 71.79–100 Hz, rhythmicity contrast falls to 0.064–0.076 while single-cell variability rises and sampled pairwise correlation remains modest. This supplies a clear trained pixel-drive PING-to-background-like trajectory. Played from high to low drive, it is the candidate AI-to-PING transition sought by the experiment. It does not yet establish an AI state.

  == 2. Design and scope

  Run r006 reads the final checkpoint of upstream cell `ping__canonical__seed42` from publication `ggs-production-composite-20260821-6d9c38eb`. Its SHA-256 digest is `afe3bce49a89c2dbdac4f986bc3ca65bda91db385b2d58914c9c765075d78a0f`. The checkpoint remains in its managed upstream location; exp099 neither copies nor retrains it.

  The trained network contains 1,024 excitatory and 256 inhibitory neurons with 784 pixel afferents. The input is official MNIST test image 0, label 7. Pixel intensity scales independent Poisson afferents up to maximum rate $D$. Forty values of $D$ span 0–100 Hz. All conditions use the same underlying random draws from seed 9900, so increasing $D$ adds spikes without replacing the paired input realization.

  Each condition lasts 400 ms at a 0.1 ms timestep; the first 100 ms is excluded. The scout records E and I rates, median E inter-spike-interval coefficient of variation, sampled E pairwise spike-count correlation, population rhythmicity contrast, dominant frequency, spectral peak prominence, and E–I lag. These are separate observables, not a mechanical AI label or composite score.

  == 3. Expected patterns

  The trained recurrent weights and structured pixel afferents could support three broad outcomes. The circuit might remain silent until a PING rhythm appears; it might pass between weakly coordinated irregular activity and PING; or increasing drive might over-recruit inhibition and dissolve an existing rhythm. PING requires more than a spectral peak: the rhythm should be prominent and E population activity should lead I in the canonical recurrent loop. Candidate background activity should show weakened population rhythmicity alongside irregular single-cell firing and limited shared timing.

  == 4. Observed transition

  #figure(
    image("/artifacts/data/exp099/trained_pixel_transition.svg", width: 100%, alt: "Raster and metrics across a trained MNIST pixel-drive sweep from silence through PING toward irregular background-like activity."),
    caption: [Official run r006. The fixed trained network and fixed MNIST image are replayed while only maximum pixel rate $D$ changes. The raster and metrics expose the continuous transition rather than assigning categorical state labels.],
  )

  #transition-video("/artifacts/data/exp099/trained_pixel_transition.mp4")

  At $D=0$, the network is silent. At $D=2.56$ Hz, E and I fire at 7.57 and 39.51 Hz per neuron. E leads I by 1 ms, rhythmicity contrast is 0.923, and the dominant frequency is 60 Hz. This is the clearest low-drive PING condition.

  Increasing drive weakens population locking rather than strengthening it. Rhythmicity contrast falls to 0.458 at 25.64 Hz, 0.212 at 51.28 Hz, and 0.064 at 71.79 Hz. At 100 Hz, E rate reaches 40.58 Hz, E ISI CV reaches 0.834, sampled E pairwise correlation is 0.189, and rhythmicity contrast remains low at 0.076. The inhibitory rate reaches 376.98 Hz, which is a warning that this endpoint is strongly driven and not automatically a biologically plausible background state.

  == 5. Lifecycle and prior evidence

  Exp099 remains one continuing ExpScout. Run r002 mapped candidate activity in a broader balanced topology; r003 expressed that search in simulator-native recurrent weights; r004 demonstrated a silence-to-PING trajectory in an untrained repository-native PING circuit. Run r005 first replayed the trained checkpoint, but its reported lag sign was inverted. Run r006 corrects that analysis convention while preserving the same simulations and scientific configuration, and is the current official evidence.

  == 6. Conclusion and next action

  The trained checkpoint and structured pixel input materially change the drive response: low nonzero drive supports PING, whereas stronger drive erodes rhythmicity and produces a more irregular, weakly coordinated candidate background. The video therefore shows PING-to-background-like activity as $D$ rises, or candidate AI-to-PING activity when traversed in reverse.

  Calling the high-drive endpoint AI would overstate the evidence. The result uses one trained cell, one image, one Poisson realization, and a 300 ms analysis window; E ISI CV remains below one and inhibitory firing becomes extremely high. The next scientific action is a bounded replication across fixed additional images and seeds, testing whether a contiguous high-drive region repeatedly combines low rhythmicity, low correlation, irregular spiking, and tolerable population rates before assigning an AI interpretation.
]
