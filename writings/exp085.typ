#let meta = (
  title: "How long do two PING networks take to synchronize?",
  date: "2026-08-19",
  description: "Measure how two PING rhythms form a stable phase relationship.",
  collection: "snnlang",
  status: "draft",
  order: 11,
)

#let body = [
  == Abstract

  This is a demonstration, not an investigation. We will search for parameters that produce a clear transition from phase drift to phase locking. We will then use one successful run to show the transition and measure its duration. It will not support a general claim about PING synchronization.

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *This experiment has not been run.* Each method includes its planned result.
  ]

  + *Define the network.* Use SNNLANG to build two PING networks. Each has 80 excitatory cells, 20 inhibitory cells, and an independent 128-channel spike input. Start from PING parameters that produced stable rhythms in earlier work. Set their inhibitory decay times to 4 ms and 5 ms so that the uncoupled rhythms differ. Use bidirectional excitatory projections from each network to both populations in the other network. This is the primary topology because long-range excitation can recruit both parts of each local PING loop. E-to-E-only and E-to-I-only coupling remain fallback options.

    *Result*

    - *Axes.* None.
    - *Traces.* None.
    - *Displayed elements.* Both PING networks, their E and I populations, independent inputs, internal connections, and cross-network connections.
    - *Why.* Confirm that the implemented graph matches the design.
    - *Expectation.* Two complete PING circuits with four reciprocal AMPA projections.

      #figure(
        image("/artifacts/data/exp085/network.svg", width: 100%, alt: "SNNLANG circuit view with two expanded PING components and four reciprocal AMPA connections."),
        caption: [SNNLANG circuit view of the proposed graph. PING A and PING B each contain an excitatory and inhibitory population. Four reciprocal AMPA projections share the selected weight $K$ after coupling.],
      )

  + *Confirm the uncoupled rhythms.* Run both networks with $K=0$. Confirm that each produces a stable rhythm and that their wrapped phase difference repeatedly crosses the phase range.

    *Result*

    - *Axes.* Time before coupling on the horizontal axis; wrapped phase difference from $-pi$ to $pi$ on the vertical axis.
    - *Trace.* Phase difference between Networks A and B.
    - *Why.* Confirm that the uncoupled rhythms drift.
    - *Expectation.* A repeating sawtooth as the faster rhythm passes the slower rhythm.

  + *Select coupling parameters.* Test candidate values of coupling weight $K$ and coupling delay $d$. For each pair $(K, d)$, switch on coupling and track the wrapped phase difference. Accept a pair if the phase difference enters a narrow band and stays there for a set hold time. Reject it if either PING rhythm stops or its population rate moves outside a stated tolerance from baseline. From the accepted pairs, choose the smallest $K$ that gives clear locking. Record $(K, d)$ for the demonstration.

    *Result*

    - *Example phase traces.*
      - *Axes.* Time after coupling on the horizontal axis; wrapped phase difference on the vertical axis.
      - *Traces.* One accepted pair and representative rejected pairs, including continued drift and a disrupted rhythm.
      - *Why.* Make the acceptance rule visible.
      - *Expectation.* The accepted trace stays in the locking band. Rejected traces drift, escape the band, or lose a valid rhythm.
    - *Parameter map.*
      - *Axes.* $K$ on the horizontal axis; $d$ on the vertical axis.
      - *Marks.* One point per pair, coloured by accepted or rejected, with the selected pair highlighted.
      - *Why.* Record how $(K, d)$ was chosen.
      - *Expectation.* At least one accepted region and a selected pair at its smallest successful $K$. This does not show robustness.

  + *Switch on coupling.* Set the selected delay $d$ before the run. At a fixed time, change only $K$ from zero to its selected value. Keep all other parameters unchanged. Continue until the phase difference becomes stable. Record all spikes and calculate each population firing rate.

    *Result*

    - *Axes.* Local time on the horizontal axis; normalized excitatory population rate on the vertical axis.
    - *Traces.* Networks A and B in three panels: before coupling, during transition, and after locking.
    - *Why.* Show how their volley timing changes.
    - *Expectation.* The peak offset drifts, adjusts, then becomes fixed.

  + *Measure the phase difference.* Use each excitatory population volley as a rhythm marker. Use these markers to calculate the phase difference $phi(t)$. Save the spike rasters, population rates, and volley times.

    *Result*

    - *Axes.* Time on the horizontal axis; population rate and phase on the vertical axes of aligned panels.
    - *Traces.* Both population rates, detected volley markers, and both phase estimates.
    - *Why.* Validate the phase measurement.
    - *Expectation.* Each phase trace follows its volley rhythm and resets at detected volleys.

  + *Measure phase locking.* The networks lock when their phase difference enters a narrow band and stays there for a set time. Report the locked phase difference. Measure $t_"sync"$ from coupling onset to the first sustained entry into the band.

    *Result*

    - *Axes.* Time on the horizontal axis; phase difference $phi(t)$ on the vertical axis.
    - *Trace.* $phi(t)$, with coupling onset, the locking band, and $t_"sync"$ marked.
    - *Why.* Measure the transition to phase locking.
    - *Expectation.* Phase drift before coupling, followed by sustained entry into the locking band.
]
