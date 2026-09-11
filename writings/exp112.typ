#import "templates/article-layout.typ": journal-article
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "templates/methods.typ": journal-methods, method-card
#import "templates/result-card.typ": journal-result-card, with-result-sections
#import "/.demolab/lib.typ": data-image
#let data-file = data-file.with(article: "exp112")

#let meta = (
  tags: ("txt", "v36.0.0"),
  title: "COBA–PING Gradient-Damping Comparison",
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11",
  description: "A controlled comparison of COBA and PING training under weak and strong voltage-gradient damping on a paired MNIST subset.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp112",)

#let render-report(data-file) = [
  == Results

  #with-result-sections[
    #journal-result-card(
      title: "Training trajectories",
      orientation: [The four conditions crossed E/I-loop engagement with voltage-gradient damping divisors of 1 and 1000.],
      visual: figure(
        data-image(
          data-file("exp112/training-comparison.png"),
          width: 100%,
          alt: "Validation accuracy and gradient-norm trajectories for COBA and PING at voltage-gradient damping divisors 1 and 1000.",
        ),
        caption: [Validation accuracy and mean pre-clipping gradient norm across 50 epochs. Colour distinguishes COBA from PING; line style distinguishes the two dimensionless damping divisors. All conditions used the same 1,080 optimization images and 120 validation images.],
        kind: image,
        supplement: [Figure],
      ),
    )

    #journal-result-card(
      title: "Final test accuracy",
      orientation: [Final-epoch parameters were evaluated on the complete official MNIST test partition.],
      visual: figure(
        data-image(
          data-file("exp112/test-accuracy.png"),
          width: 85%,
          alt: "Official MNIST test accuracy for the four architecture and voltage-gradient-damping conditions.",
        ),
        caption: [Accuracy over the same 10,000 official MNIST test images for each final-epoch network. Colour distinguishes architecture; hatching marks damping divisor 1000. This single-seed comparison provides no across-training-replicate uncertainty estimate.],
        kind: image,
        supplement: [Figure],
      ),
    )

    #journal-result-card(
      title: "Final-epoch spike rasters",
      orientation: [We replayed the same illustrative digit-0 test image through every final-epoch network.],
      visual: figure(
        data-image(
          data-file("exp112/final-epoch-rasters.png"),
          width: 100%,
          alt: "Four spike rasters for the same digit-0 MNIST test image, one for each COBA or PING and damping condition.",
        ),
        caption: [E and I spikes during the same 200 ms digit-0 presentation. Panels A–D correspond respectively to COBA at divisors 1 and 1000, then PING at divisors 1 and 1000; the horizontal rule separates 1,024 E from 256 I neurons. This paired single trial is illustrative rather than an estimate of typical activity.],
        kind: image,
        supplement: [Figure],
      ),
    )
  ]

  #journal-methods(body: (
    method-card([Pair the MNIST partitions], [We sampled 1,200 images without replacement from the official 60,000-image MNIST training partition using seed 42, then made one shared stratified 1,080/120 optimization/validation split using the same seed. All conditions used identical image membership, order and stochastic encoding streams. The complete 10,000-image official test partition remained outside training and checkpoint selection.]),
    method-card([Construct the four conditions], [We crossed a COBA condition, implemented by setting recurrent E/I coupling to zero, with an active PING loop and crossed both architectures with voltage-gradient damping divisors 1 and 1000. All four networks had identical tensor shapes and shared seed 42.]),
    method-card([Train the classifiers], [We trained each network for 50 epochs with AdamW, learning rate 0.0004, zero weight decay, global gradient clipping at 1, and no firing-rate penalty. Each 200 ms presentation used a 0.1 ms timestep, 25 Hz maximum-pixel Poisson encoding, 1,024 excitatory and 256 inhibitory hidden neurons, and an output LIF readout scored by mean pre-reset voltage.]),
    method-card([Compare retained outcomes], [We retained epoch-wise validation accuracy, cross-entropy, firing rates, pre-clipping gradient norms, skipped updates and non-finite forward batches. We evaluated every final-epoch network on the same official test partition and compared the complete two-by-two condition set without aggregating across independent training replicates.]),
    method-card([Record the illustrative rasters], [For every final checkpoint, we selected digit 0, sample 0 within that class, from the official test partition and replayed its 200 ms Poisson encoding with the common evaluation seed. We retained all excitatory and inhibitory spike events and verified the raw test index and source-image digest across conditions.]),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}
#let meta = meta + (assets: input-assets("exp112", inputs))
#let body = journal-article("exp112", inputs, report-body)
