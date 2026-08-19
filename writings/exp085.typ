#let meta = (
  title: "How long do two PING networks take to synchronize?",
  date: "2026-08-19",
  description: "Measure how two PING rhythms form a stable phase relationship.",
  collection: "snnlang",
  status: "proposal",
  order: 11,
)

#let result-link(id, text) = context {
  if target() == "html" {
    html.elem("a", attrs: (href: "#" + id), text)
  } else {
    text
  }
}

#let result-anchor(id) = context {
  if target() == "html" {
    html.elem("span", attrs: (id: id))
  }
}

#let body = [
  == Abstract

  Two PING networks can synchronize when they exchange spikes. This experiment follows one ideal pair from phase drift to phase locking. It shows how their phase difference changes. It also measures how long locking takes.

  == Methods

  + *Define the network.* Use SNNLANG to build two PING networks. Each network has 80 excitatory cells, 20 inhibitory cells, and an independent 128-channel spike input. Set their inhibitory decay times to 4 ms and 5 ms. Connect each excitatory population to both populations in the other network. The four AMPA connections share one weight, $K$. Set $K=0$ before coupling and $K=0.08$ after coupling. Use a 0.1 ms connection delay. See #result-link("result-network", [Result 1]).

  + *Start two drifting rhythms.* Run the two networks with $K=0$. Wait until each rhythm is stable. Confirm that their wrapped phase difference repeatedly crosses the phase range. See #result-link("result-drift", [Result 2]).

  + *Switch on coupling.* At a fixed time, change $K$ from 0 to 0.08. Keep all other inputs and parameters unchanged. Continue the run until the phase difference becomes stable. Record all spikes and calculate each population firing rate. See #result-link("result-coupling", [Result 3]).

  + *Measure the phase difference.* Use each excitatory population volley as a rhythm marker. Use these markers to calculate the phase difference $phi(t)$. Save the spike rasters, population rates, and volley times. See #result-link("result-phase", [Result 4]).

  + *Measure phase locking.* The networks lock when their phase difference enters a narrow band and stays there for a set time. Report the locked phase difference. Measure $t_"sync"$ from the start of coupling to the first sustained entry into the band. See #result-link("result-locking", [Result 5]).

  == Results

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *This experiment has not been run.* The network diagram comes from the SNNLANG graph. The other items describe planned figures and expectations.
  ]

  + #result-anchor("result-network")*Show the network.* The SNNLANG circuit view expands the two PING components into their excitatory and inhibitory populations. It shows both independent inputs, both internal PING loops, and all four cross-network connections.

    #figure(
      image("/artifacts/data/exp085/network.svg", width: 100%, alt: "SNNLANG circuit view with two expanded PING components and four reciprocal AMPA connections."),
      caption: [SNNLANG circuit view of the proposed graph. PING A and PING B are expanded selectively. Each contains an excitatory and inhibitory population. Four reciprocal AMPA projections share the weight $K=0$ before coupling and $K=0.08$ after coupling.],
    )

  + #result-anchor("result-drift")*Show uncoupled phase drift.* The horizontal axis shows time before coupling. The vertical axis shows wrapped phase difference from $-pi$ to $pi$. We expect a repeating sawtooth as the faster rhythm passes the slower rhythm. This pattern shows drift, not phase locking.

  + #result-anchor("result-coupling")*Show three short time windows.* Select a few cycles before coupling, during the transition, and after locking. Give each window its own panel. The horizontal axis shows local time in milliseconds. The vertical axis shows normalized excitatory population rate. Overlay Network A and Network B. Before coupling, their peaks have no fixed offset. During the transition, the offset changes. After locking, the peaks keep a fixed offset. Use the full rate traces and spike raster only as diagnostic outputs.

  + #result-anchor("result-phase")*Check the phase measurement.* The horizontal axis shows time. Aligned vertical panels show population firing rate, detected volleys, and the phase of each network. The phase traces must follow the volley rhythm before and after coupling.

  + #result-anchor("result-locking")*Show the locking time.* The horizontal axis shows time. The vertical axis shows phase difference $phi(t)$. Mark coupling onset with a vertical line. Add a shaded band around the locked phase difference. Mark $t_"sync"$ from coupling onset to the first sustained entry into this band. This plot shows the full flow from drift to phase locking.
]
