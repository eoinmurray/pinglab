#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Empirical input-rate calibration for variable-rate PING training",
  date: "2026-08-10",
  description: "Direct-simulation decoder calibration of the input-rate range for later variable-rate PING training.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp080/numbers.json")
#let d = r.decision
#let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
#let body = [
  == Abstract

  This experiment asks which input-rate interval preserves enough digit
  information after synaptic and membrane filtering to justify using it in a
  variable-rate PING training run. We trained a nonlinear decoder on freshly
  simulated MNIST features spanning eight rates, then evaluated the frozen
  decoder on held-out images at each rate. The selected interval is
  #d.recommendation.floor_hz to #d.recommendation.ceiling_hz Hz. Its lower edge
  is the first tested rate at which all three decoders meet or exceed
  #pct(r.parameters.useful_accuracy) held-out accuracy. The result calibrates
  this feature representation and decoder; it does not measure PING-network
  accuracy.

  == Methods

  #enum(
    [*Partition the MNIST dataset.* The official MNIST training partition
      contains #r.training_dataset.image_shape.first() images. We assigned its
      first #r.parameters.train_count images to decoder training and the next
      #r.parameters.validation_count images to checkpoint selection. The remaining
      #(r.training_dataset.image_shape.first() - r.parameters.train_count - r.parameters.validation_count)
      images were not used. Final evaluation used the first
      #r.parameters.test_count images from the separate official test partition.
      Training, validation, and test images therefore did not overlap.],

    [*Simulate filtered image features.*

      Each normalized pixel intensity $x_i$ generated an independent binary event
      at every $Delta t = #r.parameters.dt_ms$ ms timestep,

      $ S_i (t) tilde "Bernoulli"(r x_i Delta t / 1000). quad "(1)" $

      Here $r$ is the maximum-pixel encoding rate in spikes/s. Conductance followed

      $ g_i (t) = exp(-Delta t / tau_"AMPA") g_i (t-Delta t) + w S_i (t), quad "(2)" $

      with $tau_"AMPA" = 2$ ms and $w = #r.parameters.probe_uS$ μS. The
      non-spiking conductance-based membrane obeyed

      $
        C_E (d v_i)/(d t) = g_"L,E" (E_L-v_i) + g_i (t) (E_e-v_i), quad "(3)"
      $

      with $C_E=1$ nF, $g_"L,E"=0.05$ μS, $E_L=-65$ mV, and $E_e=0$ mV.
      Conductance began at zero and voltage at $E_L$. The decoder feature was the
      baseline-subtracted voltage averaged over the complete
      #r.parameters.presentation_ms ms presentation,

      $ z_i = 1/T integral_0^T (v_i (t)-E_L) dif t. quad "(4)" $

      Every training, validation, illustration, and test feature used a newly drawn
      spike train and direct evaluation of Equations 1--4. The random spikes form
      _shot noise_: each spike produces a discrete jump in conductance, followed by
      the exponential AMPA decay in Equation 2. These conductance pulses alter both
      the voltage toward which the membrane moves and the speed at which it moves
      there. The effect of a spike therefore depends on the voltage and conductance
      left by earlier spikes, rather than adding a fixed voltage increment#cite(1).

      Direct simulation is particularly important at low input rates. A finite
      presentation may contain no spikes, one spike, or a few spikes arriving at
      different times. The response statistics can consequently change during the
      presentation (_nonstationary_) and the distribution across presentations can
      be asymmetric or concentrated around a few distinct outcomes
      (_non-Gaussian_)#cite(2). Evaluating Equations 1--4 for each presentation
      preserves these effects instead of replacing them with a steady, bell-shaped
      approximation.

    ],

    [*Train a mixed-rate decoder.*

      The official MNIST training partition supplied the first
      #r.parameters.train_count images for training and the next
      #r.parameters.validation_count for validation. Every presentation sampled one
      of the eight rates #r.parameters.rates_hz.map(str).join(", ") Hz uniformly and
      independently. A 784--1024--10 ReLU decoder was trained with cross-entropy and
      Adam for #r.parameters.epochs epochs. Seeds
      #r.parameters.seeds.map(str).join(", ") defined independent initializations,
      rate assignments, and spike trains. Validation accuracy selected one
      checkpoint per seed.

      #figure(
        image("/artifacts/data/exp080/training_history.svg", width: 72%),
        caption: [Mixed-rate validation histories for the three independently
          trained nonlinear decoders. Every epoch used fresh direct feature
          simulations.],
      )

    ],

    [*Evaluate held-out accuracy and select the interval.*

      #enum(
        [*Select the held-out images.* We took the first
          #r.parameters.test_count images from the official MNIST test
          partition. These images were not used for training or checkpoint
          selection.],

        [*Simulate shared test features.* We simulated every held-out image once
          at each registered input rate. All three validation-selected decoders
          received the same simulated feature for a given image and rate, so
          differences among decoders were not caused by different spike
          realizations.],

        [*Measure each decoder's accuracy.* For each input rate and decoder, we
          divided the number of correctly classified test images by
          #r.parameters.test_count. This produced one held-out accuracy per
          decoder at each rate.],

        [*Select the interval.*

          The practical floor was the lowest tested rate at which every decoder
          met or exceeded #pct(r.parameters.useful_accuracy) accuracy,

          $
            r_"train" = min {r in cal(R): min_(s in cal(S)) A_s (r) >= 0.5}. quad "(5)"
          $

          Here $cal(R)$ is the set of tested rates, $cal(S)$ is the set of decoder
          seeds, $A_s(r)$ is held-out accuracy for decoder $s$ at rate $r$, and
          $r_"train"$ is the selected training floor. The upper edge was the highest
          tested rate. We did not interpolate between tested rates.],
      )
    ],
  )

  == Results

  === What the decoder saw

  #figure(
    image("/artifacts/data/exp080/feature_images.png"),
    caption: [An MNIST input image and its directly simulated filtered features.
      The left panel shows the normalized input. The remaining panels show
      independent spike realizations at maximum-pixel encoding rates of
      #r.parameters.rates_hz.at(2) Hz, #r.parameters.rates_hz.at(5) Hz, and
      #r.parameters.rates_hz.last() Hz. Each simulation uses a
      #r.parameters.probe_uS μS synaptic conductance and a
      #r.parameters.presentation_ms ms presentation. Greater input rate
      preserves more of the digit's spatial structure.],
  )

  Sparse presentations retained fragments of the digit rather than a uniformly
  attenuated image. Increasing rate filled in the spatial pattern and reduced
  the importance of individual event times.

  === Empirical rate selection

  #figure(
    image("/artifacts/data/exp080/psychometric.svg", width: 72%),
    caption: [Held-out nonlinear-decoder accuracy against maximum-pixel encoding
      rate. Points average the official test images and three independently
      trained decoders. The band spans the lowest and highest decoder accuracy
      at each rate. Horizontal rules mark chance and the
      #pct(r.parameters.useful_accuracy) practical criterion. The red vertical
      rule marks the first tested rate at which all three decoders meet that
      criterion, which defines the selected floor.],
  )

  All three decoders first met the practical
  #pct(r.parameters.useful_accuracy) criterion at #d.r_train_hz Hz. Mean
  accuracy at that condition was
  #pct(d.rows.filter(row => row.rate_hz == d.r_train_hz).first().accuracy).
  We therefore select #d.recommendation.floor_hz to #d.recommendation.ceiling_hz Hz
  for later variable-rate PING training.

  == Relation to prior work

  Neural decoding measures information accessible to a specified readout, not
  an absolute information content or a mechanistic account of the encoded
  population#cite(3). We therefore interpret the ANN psychometric curve only as
  a decoder-relative calibration. Prior visual-population work likewise used
  held-out decoding performance to quantify recoverable stimulus information
  from noisy neural responses#cite(4), motivating the empirical
  accuracy-versus-rate design used here. The filtered-shot-noise literature
  motivates direct simulation of the conductance and membrane dynamics#cite(1, 2).

  #reference-list((
    (
      text: [Wolff & Lindner: _Mean, Variance, and Autocorrelation of Subthreshold Potential Fluctuations Driven by Filtered Conductance Shot Noise_. Neural Computation, 2010.],
      doi: "10.1162/neco.2009.02-09-958",
    ),
    (
      text: [Brigham & Destexhe: _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
    (
      text: [Quian Quiroga & Panzeri: _Extracting Information from Neuronal Populations: Information Theory and Decoding Approaches_. Nature Reviews Neuroscience, 2009.],
      doi: "10.1038/nrn2578",
    ),
    (
      text: [Warland, Reinagel & Meister: _Decoding Visual Information From a Population of Retinal Ganglion Cells_. Journal of Neurophysiology, 1997.],
      doi: "10.1152/jn.1997.78.5.2336",
    ),
  ))
]
