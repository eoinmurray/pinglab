// title: study.3-selective-routing
// date: 2026-02-21
// description: Selective routing via axonal delay scanning across PING-gated excitability windows.

#let config = json("_artifacts/study.3-selective-routing/config.json")





*Readers note:* click images to make them larger, you can then use arrow keys
to scroll through them.


= Introduction


#figure(
  image("_assets/study.3-selective-routing/ping-switch.png"),
  caption: [Switch Topology.],
)



This experiment uses the minimal routing switch:

1. Rhythmic drive excites `E_src`.
2. `E_src` sends to both targets (`E1`, `E2`).
3. A shared inhibitory population `I` receives from both targets and projects
   back to both targets.
4. We scan only `delay_ms` on `E_src -> E2`.

Why rhythmic input is required:

1. Delay only matters for routing when there is a phase reference.
2. Rhythmic drive creates repeating excitable/inhibitory windows in the shared
   gate-target system.
3. Scanning delay then moves `E2` arrivals across those windows, which enables
   branch selection.
4. Without rhythm, delay mostly changes latency, not which branch wins.

The idea is simple: as `E_src -> E2` delay moves through the oscillation cycle,
the arrival time into `E2` alternates between favorable and unfavorable phases
relative to shared inhibition, which changes which target dominates.

The baseline (with source feedforward removed) confirms that the targets are not
independently driven by `E_src`; the switching effect appears only when
feedforward is enabled and delay is scanned.

Mechanism in simple terms:

1. `I` is a shared inhibitory gate for both targets, so `E1` and `E2` compete
   through the same inhibitory rhythm.
2. `E_src -> E1` timing is fixed, but `E_src -> E2` timing is shifted by the
   scanned delay.
3. When `E_src` input arrives at `E2` during an excitable part of the shared
   cycle, `E2` wins and contributes strongly to `I`.
4. When it arrives during the inhibitory/refractory part, `E2` is suppressed,
   and `E1` dominates.
5. As delay keeps increasing, this phase relation wraps around, producing
   repeating selection regimes.

`I` does not need a different qualitative rhythm in each scan point. The key is
that the relative phase of `E2` input against the shared inhibitory cycle
changes with delay, which is enough to switch the dominant output branch.


= Results


#figure(
  grid(
    columns: 5,
    gutter: 4pt,
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_00_delay-0.000_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_01_delay-0.769_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_02_delay-1.538_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_03_delay-2.308_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_04_delay-3.077_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_05_delay-3.846_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_06_delay-4.615_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_07_delay-5.385_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_08_delay-6.154_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_09_delay-6.923_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_10_delay-7.692_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_11_delay-8.462_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_12_delay-9.231_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_13_delay-10.000_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_14_delay-10.769_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_15_delay-11.538_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_16_delay-12.308_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_17_delay-13.077_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_18_delay-13.846_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_19_delay-14.615_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_20_delay-15.385_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_21_delay-16.154_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_22_delay-16.923_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_23_delay-17.692_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_24_delay-18.462_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_25_delay-19.231_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_26_delay-20.000_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_27_delay-20.769_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_28_delay-21.538_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_29_delay-22.308_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_30_delay-23.077_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_31_delay-23.846_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_32_delay-24.615_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_33_delay-25.385_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_34_delay-26.154_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_35_delay-26.923_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_36_delay-27.692_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_37_delay-28.462_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_38_delay-29.231_dark.png"),
    image("_artifacts/study.3-selective-routing/raster_row-1_delay_scan_39_delay-30.000_dark.png"),
  ),
)



The raster scan shows clear switching windows:

1. Low delays: both targets can be active.
2. Mid delays: `E2` is strongly suppressed while `E1` remains active.
3. Later delays: the preference flips and `E2` dominates while `E1` drops.
4. Near the end of the scan, behavior wraps back toward the initial regime.

This is the expected phase-selection behavior from a delay-controlled gate.

#figure(
  image("_artifacts/study.3-selective-routing/spikes_row-2_delay_scan_targets-vs-delay_dark.png"),
)



The spike-count plot confirms the same pattern quantitatively:

1. A broad region where `E1` is high and `E2` is near zero.
2. A transition band where both are comparable.
3. A second region where `E2` is high and `E1` is low.

Why `E1` is also suppressed:

1. `I` is shared, so inhibition is coupled across both targets.
2. When `E2` becomes well-timed and starts winning, it drives `I` more strongly.
3. That stronger shared inhibition feeds back to both `E2` and `E1`.
4. In that regime, `E1` gets pushed into less excitable phases more often, so
   its spike count drops even though its source delay is fixed.

So this setup gives a simple, interpretable selective-routing mechanism: scan
one delay and the preferred output population changes.
