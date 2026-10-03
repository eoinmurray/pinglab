#set document(title: "First-year report: draft table of contents")
#set page(paper: "a4", margin: (x: 22mm, y: 18mm))
#set text(font: "Libertinus Serif", size: 11pt)
#set par(leading: 0.45em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 11pt)
#show heading.where(level: 1): set block(above: 0.7em, below: 0.35em)
#let entry(n, body) = block(above: 0.42em, below: 0.42em, inset: (left: 6mm))[#n #h(2mm) #body]

#text(size: 20pt, weight: "bold")[First-year report]

#text(size: 13pt)[Draft table of contents]

#v(4mm)
*Eoin Murray* #h(6mm) CRSid: em586

Division F · PhD in Engineering

Academic years: 2025/26–2026/27 · Started January 2026
#v(3mm)

= Introduction
#entry("1.1", [Research context and motivation])
#entry("1.2", [Research problem and scope])

= Literature review
#entry("2.1", [Spiking neural networks and temporal processing])
#entry("2.2", [Neural oscillations and their computational role])
#entry("2.3", [Theories of gamma oscillations])

= Research questions and objectives
#entry("3.1", [Gamma oscillations as a substrate for reliable neuromorphic computing])
#entry("3.2", [Investigating the role of gamma oscillations in the brain])

= Methods
#entry("4.1", [Models and theoretical framework])
#entry("4.2", [Experimental tasks, datasets and simulation procedures])
#entry("4.3", [Evaluation measures and comparisons])

= Progress and preliminary results
#entry("5.1", [Comparison of simulated gamma oscillations with published findings])
#entry("5.2", [Training gamma networks using backpropagation through time (BPTT)])
#entry("5.3", [Excitatory firing-rate reduction in trained gamma networks])
#entry("5.4", [Streaming digit classification using trained gamma networks])

= Plan for the remaining PhD
#entry("6.1", [Neuromodulation of gamma parameters])
#entry("6.2", [Incorporation of learning rules during inference])
#entry("6.3", [Gamma networks for spatial decision-making tasks])
#entry("6.4", [Gamma networks for decision-making and action selection])
#entry("6.5", [Implementation of gamma networks on neuromorphic hardware])
#entry("6.6", [Timeline and milestones])

= References

= Appendices
#entry("8.1", [Approved module coursework replacing the MRE])
#entry("8.2", [RDC logbook])
