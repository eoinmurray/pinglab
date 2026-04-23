---
layout: ../layouts/MarkdownLayout.astro
title: "References"
---

# References

Reading list for this project. PDFs themselves are third-party copyrighted material and are not redistributed; follow the links to fetch them. Citation keys match the filenames under *src/papers/*.

## Temporal discretisation

- **parthasarathy-et-al** — Parthasarathy, Burghi & O'Leary, *Temporal Discretisation Shapes Optimisation Landscape in Spiking Neural Networks*. The direct prior work for [notebook 003](/notebooks/nb003/): shows that the Standard-SNN update (snnTorch / Nengo / Rockpool / SpiNNaker2 form) learns a *dt*-dependent discrete dynamical system, and that training at one *dt* and evaluating at another collapses accuracy. Also taxonomises three ways to transport a Poisson input across *dt*s (zero-pad, resample, downsample), which motivates the encoder next steps in nb003.

## Surrogate-gradient SNN training

- **eshraghian-et-al** — Eshraghian et al., *Training Spiking Neural Networks Using Lessons From Deep Learning*, Proc. IEEE 2023. [arXiv:2109.12894](https://arxiv.org/abs/2109.12894). snnTorch reference.
- **neftci-et-al** — Neftci, Mostafa & Zenke, *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE Signal Processing Magazine 2019. [arXiv:1901.09948](https://arxiv.org/abs/1901.09948).
- **burghi-et-al** — Burghi et al., costate-based SNN training (2026 draft). Cited in [Training](/training/) for principled adjoint scaling.

## PING / gamma rhythms

- **borgers-neural-computation** — Börgers, *An Introduction to Modeling Neuronal Dynamics*, Texts in Applied Mathematics 66, Springer 2017.
- **chapter-30-ping** — Kopell & Börgers, PING chapter in Börgers (2017).
- **buzsaki-and-wang** — Buzsáki & Wang, *Mechanisms of gamma oscillations*, Annu. Rev. Neurosci. 2012. [doi:10.1146/annurev-neuro-062111-150444](https://doi.org/10.1146/annurev-neuro-062111-150444).
- **fries** — Fries, *A mechanism for cognitive dynamics: neuronal communication through neuronal coherence*, Trends in Cognitive Sciences 2005. [doi:10.1016/j.tics.2005.08.011](https://doi.org/10.1016/j.tics.2005.08.011).
- **kopelletal-et-al** — Kopell et al., gamma/theta rhythm theory.
- **viriyopase-et-al** — Viriyopase et al., ING vs PING comparison.
- **segneri-et-al** — Segneri et al., PING network dynamics.

## Balanced / asynchronous state

- **renart** — Renart et al., *The asynchronous state in cortical circuits*, Science 2010. [doi:10.1126/science.1179850](https://doi.org/10.1126/science.1179850).
- **shadlen** — Shadlen & Newsome, *The variable discharge of cortical neurons*, J. Neurosci. 1998.

## Spiking MNIST / benchmarks

- **cramer-et-al** — Cramer, Stradmann, Schemmel & Zenke, *The Heidelberg spiking datasets for the systematic evaluation of spiking neural networks*, IEEE TNNLS 2022 (preprint Oct 2020). [arXiv:1910.07407](https://arxiv.org/abs/1910.07407). Source of the SHD dataset used in [notebook 004](/notebooks/nb004/) — 10,420 high-quality recordings of spoken digits 0–9 in English and German, converted through a biologically-motivated cochlea model into 700-channel spike rasters of roughly 1 s per trial.
- **yan-et-al** — Yan et al., sequential-MNIST SNN architectures (2025). Depth precedent for sMNIST.
- **yan-et-al-supplementary** — Yan et al. supplementary material.
- **xing-et-al** — Xing et al., SNN classification benchmarks.
- **lee-et-al** — Lee et al., surrogate-gradient MNIST.
- **nguyen-et-al** — Nguyen et al., SNN training tricks.
- **pes-et-al** — Pes et al., PES learning rule.
- **nandi-et-al** — Nandi et al., neuromorphic benchmarks.
