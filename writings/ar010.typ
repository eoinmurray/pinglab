#let meta = (
  title: "Literature Companion",
  date: "2026-07-02",
  description: "A literature companion to the manuscript (ar009): agent-located source pointers and self-checked quotes for each reference.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

#let body = [
  // Divider that survives both targets: a CSS-styled <hr> on the web, a drawn
  // rule in the PDF (matches the engine's own separator in lib.typ).
  #let divider = context {
    if target() == "html" { html.elem("hr") } else { line(length: 100%, stroke: 0.5pt + gray) }
  }

  A literature companion to the #link("/ar009/")[the manuscript (ar009)].

  Quotes below are agent-located pointers to each source, self-checked for verbatim match but not independently verified — confirm against the source before citing.

  #divider

  == Buzsáki & Wang (2012) — Mechanisms of Gamma Oscillations

  #link("https://doi.org/10.1146/annurev-neuro-062111-150444")[doi.org/10.1146/annurev-neuro-062111-150444]

  *Summary.* A canonical review of the cellular and network mechanisms of gamma oscillations, which it defines as a synchronous network rhythm in the 30–90 Hz band. It argues that gamma rhythmogenesis is paced by perisomatic inhibition from fast-spiking interneurons, distinguishes the interneuron-network (ING) and excitatory–inhibitory / pyramidal–interneuron (PING) generation mechanisms, and reviews how gamma is nested within slower rhythms and linked to cognition.

  *Claim it supports (§1.1).* _"Gamma oscillations in the 30–80 Hz band have been associated with attention, binding, and gating in cortical activity [1, 2, 3]."_

  *Quotes.*

  - _"We refer to periodic events in the 30–90-Hz band as gamma oscillations"_ (p. 3) — establishes the gamma frequency band the manuscript invokes.
  - _"Other synonyms referring to this band are the 40-Hz oscillation or cognitive rhythm"_ (p. 3) — gamma has historically been termed the cognitive rhythm, directly grounding the association with cognition.
  - _"Several studies have examined the relationship between cross-frequency coupling of gamma oscillations and cognitive processes"_ (p. 14) — supports the general link between gamma and cognitive function that the manuscript's sentence asserts.


  #divider


  == Fries (2015) — Rhythms for Cognition: Communication through Coherence

  #link("https://doi.org/10.1016/j.neuron.2015.09.034")[doi.org/10.1016/j.neuron.2015.09.034]

  *Summary.* A Perspective by Pascal Fries (Neuron, 2015) presenting a revised Communication through Coherence (CTC) hypothesis: communication between neuronal groups is governed by rhythmic synchronization, especially in the gamma band (30-90 Hz). Gamma creates rhythmic windows of postsynaptic excitability, so coherent inputs arrive at high-excitability moments and gain effective connectivity, while incoherent inputs arrive at random phases. Fries reviews evidence that attention entrains downstream neurons (e.g. V4) to the attended stimulus's V1 gamma rhythm, that gamma frequency rises with attention and salience, that top-down alpha-beta (8-20 Hz) controls bottom-up gamma, and that a 7-8 Hz theta rhythm underlies attentional sampling.

  *Claim it supports (§3.3).* _"This is consistent with — though not specifically diagnostic of — the gamma cycle acting as a temporal unit for classification, in the spirit of the communication-through-coherence framework [2, 35]."_

  *Quotes.*

  - _"These data suggest that the neuronal representation of visual stimulus orientation pulsates with the gamma cycle."_ (p. 224 (Gamma-Band Coherence Renders Communication Precise)) — Establishes that stimulus information (orientation) is packaged within, and read out relative to, the gamma cycle as a temporal unit.
  - _"Specifically, the results suggest a gamma-rhythmic pulsatile rate code, in which the spatial pattern of spike rates holds representations, yet only during short temporal windows in the gamma cycle."_ (p. 224 (Gamma-Band Coherence Renders Communication Precise)) — Directly supports the notion of the gamma cycle as a temporal unit carrying a neuronal representation, per the CTC framework.
  - _"Because the main thrust of the concept is that neuronal communication is subserved by neuronal synchronization, often quantified by the coherence metric, I have named the concept "Communication through Coherence," or CTC."_ (p. 220 (Communication through Coherence)) — Defines the communication-through-coherence framework named in the citing sentence.


  #divider


  == Fries, Reynolds, Rorie & Desimone (2001) — Modulation of Oscillatory Neuronal Synchronization by Selective Visual Attention

  #link("https://doi.org/10.1126/science.1055465")[doi.org/10.1126/science.1055465]

  *Summary.* Fries, Reynolds, Rorie and Desimone (Science) recorded spikes and local field potentials in macaque cortical area V4 during a selective visual-attention task where the animal attended one stimulus while ignoring a nearby distractor. Using spike-triggered averages and spike-field coherence, they quantified how attention altered synchronization across frequencies. Attending a stimulus inside a neuron's receptive field increased gamma-band (about 35 to 90 Hz) synchronization and reduced low-frequency (below about 17 Hz) synchronization. Because postsynaptic integration times are short, they argue gamma synchronization enhances the effective synaptic gain of the attended population, amplifying behaviorally relevant signals beyond firing-rate changes.

  *Claim it supports (§1.1).* _"Gamma oscillations in the 30–80 Hz band have been associated with attention, binding, and gating in cortical activity [1, 2, 3], with the original visual-cortex observation reported by [4]."_

  *Quotes.*

  - _"In summary, attention increased gamma frequency and reduced low-frequency synchronization of V4 neurons representing the behaviorally relevant stimulus."_ (p. 1562) — Summary sentence reiterating that attention enhances gamma-frequency synchronization in cortex.
  - _"Small changes in gamma-frequency synchronization with attention might lead to pronounced firing-rate changes at subsequent stages"_ (p. 1560) — Connects gamma-frequency synchronization to gating/amplification of attended signals through cortical stages.


  #divider


  == Gray, König, Engel & Singer (1989) — Oscillatory Responses in Cat Visual Cortex Exhibit Inter-Columnar Synchronization Which Reflects Global Stimulus Properties

  #link("https://doi.org/10.1038/338334a0")[doi.org/10.1038/338334a0]

  *Summary.* Gray, König, Engel and Singer (1989 Nature letter) recorded multi-unit activity from 5–7 sites in cat visual cortex (area 17). Using auto- and cross-correlation of spike trains, they found stimulus-evoked oscillatory firing at 40–60 Hz (mean 50 ± 6 Hz) that synchronized with near-zero phase difference across spatially separate cortical columns. Synchronization depended on the spatial separation of sites and similarity of orientation preferences, and was enhanced by global stimulus properties such as coherent motion and continuity. They proposed such synchrony could bind distributed features of a visual pattern into coherent representations.

  *Claim it supports (§1.1).* _"Gamma oscillations in the 30–80 Hz band have been associated with attention, binding, and gating in cortical activity [1, 2, 3], with the original visual-cortex observation reported by [4]."_

  *Quotes.*

  - _"we have shown that neurons in the cat visual cortex have oscillatory responses in the range 40--60 Hz"_ (p. 334 (abstract)) — Establishes the paper's core observation of gamma-band oscillatory responses in cat visual cortex, the referenced original visual-cortex finding.
  - _"The frequency of these oscillatory responses ranged from 40 to 60 Hz (mean, 50± 6 Hz)"_ (p. 334) — Quantifies the gamma-band frequency of the oscillatory visual-cortex responses reported in this work.


  #divider


  == Whittington, Traub, Kopell, Ermentrout & Buhl (2000) — Inhibition-Based Rhythms: Experimental and Mathematical Observations on Network Dynamics

  #link("https://doi.org/10.1016/S0167-8760(00")[doi.org/10.1016/S0167-8760(00)00173-2]00173-2)

  *Summary.* This review by Whittington, Traub, Kopell, Ermentrout and Buhl surveys inhibition-based rhythms — EEG oscillations in the 12–80 Hz range (beta and gamma) from inhibitory interneurons. Combining in vitro slice and computational models, it argues these rhythms depend on GABA\_A-mediated inhibition and collapse when it is blocked. It distinguishes two gamma mechanisms: interneuron network gamma (ING), paced by mutually connected inhibitory cells via IPSPs, and pyramidal-interneuron network gamma (PING), paced by excitatory-inhibitory feedback. It reviews how IPSP amplitude and drive set frequency, how excitation synchronises separated sites via interneuron doublets, and how prolonging IPSP decay converts gamma to beta.

  *Claim it supports (§1.1).* _"Two generation mechanisms are commonly distinguished, ING and PING [5, 6]; the present work focuses on PING."_

  *Quotes.*

  - _"Thus an oscillation is produced involving the interplay between excitatory pyramidal cells and interneurons; pyramidal interneuron network gamma (PING)."_ (p. 324, Section 4) — The paper names and defines PING as the second gamma-generation mechanism, the pyramidal-interneuron network mechanism — establishing ING and PING as the two distinguished mechanisms.
  - _"The central difference is which class of cells (on the whole) acts to pace the network."_ (p. 325) — Establishes that ING and PING are distinguished by which cell population paces the rhythm, confirming they are two distinct generation mechanisms.


  #divider


  == Williams et al. (2026) — Fast Spiking Interneurons Autonomously Generate Fast Gamma Oscillations in the Medial Entorhinal Cortex with Excitation Strength Tuning ING-PING Transitions

  #link("https://doi.org/10.1523/ENEURO.0452-25.2026")[doi.org/10.1523/ENEURO.0452-25.2026]

  *Summary.* This experimental and computational study examines how fast-spiking interneurons generate fast gamma oscillations (40-140 Hz) in mouse medial entorhinal cortex layer II/III. Using optogenetic stimulation, whole-cell electrophysiology, and NetPyNE network modeling, the authors find that fast-spiking interneurons fire robustly at gamma frequencies while excitatory cells engage in cycle skipping. Gamma-frequency inhibition persists after AMPA/kainate blockade, indicating an interneuron network gamma (ING) mechanism, and PV+ activation confirms autonomous fast gamma. Modeling shows weak excitatory drive to interneurons favors fast ING-dominated rhythms whereas stronger excitation shifts toward slower PING, revealing how excitation strength tunes the oscillatory regime.

  *Claim it supports (§1.1).* _"Two generation mechanisms are commonly distinguished, ING and PING [5, 6]; the present work focuses on PING."_

  *Quotes.*

  - _"Two primary mechanisms have been proposed: pyramidal–interneuron network gamma (PING), which relies on excitatory–inhibitory interactions"_ (p. 2) — Introduction, listing the two proposed gamma generation mechanisms
  - _"and interneuron network gamma (ING), which emerges from mutual inhibition among fast-spiking interneurons"_ (p. 2) — Introduction, defining the second of the two proposed gamma mechanisms (ING)


  #divider


  == Börgers (2017) — The PING Model of Gamma Rhythms

  #link("https://doi.org/10.1007/978-3-319-51171-9_30")[doi.org/10.1007/978-3-319-51171-9_30]

  *Summary.* Chapter 30 of Borgers's textbook on rhythmic neuronal dynamics, presenting PING (pyramidal-interneuronal network gamma) as a mechanism generating gamma-band (roughly 30-80 Hz) oscillations from coupled excitatory (E, RTM) and inhibitory (I, WB) populations. From a two-cell E-I network (only E-to-I and I-to-E), each E-cell spike triggers an I-cell response that silences the E-cells, which fire again more synchronously as inhibition decays. Larger simulations show gamma developing within about 50 ms, and analyze sparse connectivity, drive strength, the suppression boundary, and recurrent I-to-I and E-to-E.

  *Claim it supports (§5.1.6).* _"The $tau_"GABA" = 9$ ms value is the canonical PING value of [7]."_

  *Quotes.*

  - _No verbatim anchor found in the source._


  #divider


  == Cardin, Carlén, Meletis, Knoblich, Zhang, Deisseroth, Tsai & Moore (2009) — Driving Fast-Spiking Cells Induces Gamma Rhythm and Controls Sensory Responses

  #link("https://doi.org/10.1038/nature08002")[doi.org/10.1038/nature08002]

  *Summary.* Cardin et al. (Nature 2009) test causally whether fast-spiking inhibitory interneurons generate cortical gamma oscillations. Using optogenetics, they expressed channelrhodopsin-2 selectively in parvalbumin-positive fast-spiking interneurons or in excitatory regular-spiking neurons in mouse barrel cortex in vivo. Driving fast-spiking cells with 1-ms light pulses across 8 to 200 Hz selectively amplified LFP power in the gamma band (20 to 80 Hz), whereas driving excitatory cells amplified only lower frequencies, a cell-type-specific double dissociation. Gamma timing relative to sensory input shaped evoked response amplitude and precision, providing the first causal in vivo evidence for the fast-spiking-gamma hypothesis.

  *Claim it supports (§1.1).* _"Optogenetic activation of fast-spiking interneurons in intact cortical circuits drives gamma rhythms [8, 9, 10, 11]."_

  *Quotes.*

  - _"Here we show that light-driven activation of fast-spiking interneurons at varied frequencies (8–200 Hz) selectively amplifies gamma oscillations."_ (p. 1, Abstract) — States the paper's central optogenetic finding: driving FS interneurons selectively amplifies gamma.
  - _"This double dissociation of cell-type-specific state induction (gamma by FS and lower frequencies by RS) directly supports the prediction that FS-PV+ interneuron activation is sufficient and specific for induction of gamma oscillations."_ (p. 3, FS activation generates gamma oscillations) — Confirms FS interneuron activation specifically induces gamma, distinguishing it from RS-driven lower frequencies.
  - _"Synchronous FS-PV+ interneuron activity driven by periodic stimulation of light-activated channels generated gamma oscillations in a cortical network, and these gated sensory processing in a temporally specific manner."_ (p. 5, Discussion) — Summarizes that optogenetically driven FS interneuron activity generates gamma in the intact cortical circuit.


  #divider


  == Sohal, Zhang, Yizhar & Deisseroth (2009) — Parvalbumin Neurons and Gamma Rhythms Enhance Cortical Circuit Performance

  #link("https://doi.org/10.1038/nature07991")[doi.org/10.1038/nature07991]

  *Summary.* Sohal, Zhang, Yizhar and Deisseroth used optogenetics in mice to modulate fast-spiking parvalbumin (PV) interneurons and test their role in gamma (30–80 Hz) oscillations. Inhibiting PV interneurons with eNpHR suppressed gamma in vivo, while driving them (even via non-rhythmic feedback inhibition through ChR2) generated emergent gamma rhythmicity. Using dynamic clamp, they showed gamma-frequency modulation of excitatory input enhances neocortical signal transmission by increasing input–output gain, reducing response variability, and increasing mutual information between input rate and output spike rate. They argue PV interneurons and gamma oscillations together amplify signals and reduce circuit noise.

  *Claim it supports (§1.1).* _"Optogenetic activation of fast-spiking interneurons in intact cortical circuits drives gamma rhythms [8, 9, 10, 11]."_

  *Quotes.*

  - _"We find that inhibiting parvalbumin interneurons suppresses gamma oscillations in vivo, whereas driving these interneurons (even by means of non-rhythmic principal cell activity) is sufficient to generate emergent gamma-frequency rhythmicity."_ (Abstract) — Optogenetically driving PV interneurons is sufficient to generate gamma rhythmicity.
  - _"Because inhibition of PV interneurons was found to suppress gamma power, we next sought to determine whether stimulating PV cells could elicit gamma oscillations in downstream PY neurons."_ (p. 2) — Motivates the experiment where optogenetic drive of PV cells elicits gamma in downstream pyramidal neurons.


  #divider


  == Phensy et al. (2026) — Prefrontal Gamma Oscillations Engage Dynamic Cell Type-Specific Configurations to Support Flexible Behavior

  #link("https://doi.org/10.1016/j.neuron.2026.05.002")[doi.org/10.1016/j.neuron.2026.05.002]

  *Summary.* Phensy et al. (Neuron, 2026) study how prefrontal gamma oscillations support cognitive flexibility. Using dual-color genetically encoded voltage indicators and optogenetics in mice performing a rule-shift task, they measure ∼40 Hz synchrony between prefrontal parvalbumin interneurons (PVIs) and MD-projecting PFC neurons. They find PVIs synchronize with PFC→MD neurons in distinct, circuit-specific in-phase and antiphase configurations recruited by behavior: PVI gamma-synchrony rises when mice change strategies, and synchrony patterns distinguish conflict from non-conflict decisions. Optogenetic PVI inhibition disrupts learning and synchrony. They argue gamma comprises multiple cell-type- and phase-specific synchrony motifs subserving dissociable behaviors, with implications for schizophrenia.

  *Claim it supports (§1.1).* _"Optogenetic activation of fast-spiking interneurons in intact cortical circuits drives gamma rhythms [8, 9, 10, 11]."_

  *Quotes.*

  - _"Parvalbumin interneurons (PVIs) are known to regulate PFC microcircuits and generate synchronized gamma-frequency (∼40 Hz) neural oscillations"_ (p. 1, Summary) — Establishes that PVIs (fast-spiking interneurons) generate synchronized gamma-frequency oscillations in PFC.
  - _"Recent work from our lab has identified that prefrontal PVIs not only generate local γ oscillations but also synchronize γ activity across the left and right prefrontal cortices in mice performing a cognitive flexibility task"_ (p. 1, Introduction) — States that prefrontal PVIs generate local gamma oscillations, supporting that interneuron activity drives gamma rhythms.
  - _"of 40 Hz optogenetic stimulation delivered either in phase or out of phase across the hemispheres"_ (p. 2, Introduction) — Notes optogenetic 40 Hz stimulation of PVIs was used to manipulate gamma synchrony in intact circuits.


  #divider


  == Offermanns, Pöpplau & Hanganu-Opatz (2026) — Developmental Embedding of Parvalbumin Interneurons Drives Local and Crosshemispheric Prefrontal Gamma Synchrony

  #link("https://doi.org/10.1016/j.pneurobio.2025.102866")[doi.org/10.1016/j.pneurobio.2025.102866]

  *Summary.* Offermanns, Pöpplau and Hanganu-Opatz combine bilateral in vivo electrophysiology with selective optogenetic manipulations of parvalbumin-positive (PV+) and somatostatin-positive (SOM+) interneurons in the developing mouse medial prefrontal cortex. They show that crosshemispheric gamma synchrony strengthens with age alongside local gamma power, and that the inhibitory effect of PV+ interneurons emerges to operate within the classical gamma range from adolescence onwards. Using 30 and 50 Hz optogenetic stimulation, they demonstrate that only PV+, not SOM+, interneurons can operate at gamma frequencies, identifying the SOM+ to PV+ interneuron switch as a mechanism of prefrontal gamma ontogeny.

  *Claim it supports (§1.1).* _"Optogenetic activation of fast-spiking interneurons in intact cortical circuits drives gamma rhythms [8, 9, 10, 11]."_

  *Quotes.*

  - _"However, using a stimulation frequency of 50 Hz disrupted the power of endogenous gamma oscillation when stimulating SOM+ but not PV+ INs, indicating that SOM+ INs are not mechanistically involved in the generation of gamma oscillations."_ (Results) — Optogenetic 50 Hz stimulation shows only PV+ (not SOM+) interneurons drive endogenous gamma.
  - _"Collectively our data suggest that from the third postnatal week on PV+ INs functionally operate in the gamma frequency range, whereas SOM+ INs fail to participate in gamma generation, confirming PV+ INs as driving force for the emergence of prefrontal gamma oscillations."_ (Discussion) — PV+ interneurons are the driving force for prefrontal gamma oscillations.


  #divider


  == Whittington, Traub & Jefferys (1995) — Synchronized Oscillations in Interneuron Networks Driven by Metabotropic Glutamate Receptor Activation

  #link("https://doi.org/10.1038/373612a0")[doi.org/10.1038/373612a0]

  *Summary.* This 1995 Nature letter by Whittington, Traub and Jefferys reports an emergent 40-Hz gamma oscillation in networks of inhibitory interneurons coupled by GABA\_A synapses, using rat hippocampal and neocortical slices. In CA1 pyramidal cells, roughly 40-Hz inhibitory postsynaptic potential oscillations arise when metabotropic glutamate receptor activation tonically excites the interneurons. They are blocked by the GABA\_A antagonist bicuculline but survive ionotropic glutamate and GABA\_B block, showing they arise from mutual inhibition rather than recurrent excitation. Frequency depends on interneuron excitation and inhibitory potential decay kinetics, confirmed by 128-interneuron simulations. The authors propose such networks entrain pyramidal firing in vivo.

  *Claim it supports (§1.1).* _"Earlier in vitro recordings [12] and biophysical models [13] characterised interneuron-driven gamma in isolated inhibitory networks, and the synaptic mechanisms that synchronise the interneuron pool have been described [14]."_

  *Quotes.*

  - _"Here we report an emergent 40-Hz oscillation in networks of inhibitory neurons connected by synapses using GABA_A (γ-aminobutyric acid) receptors in slices of rat hippocampus and neocortex."_ (p. 612, abstract) — States the central finding — 40-Hz gamma emerging from an inhibitory interneuron network in vitro slices.
  - _"The 40-Hz oscillations are a collective behaviour of the network of interneurons in the hippocampus."_ (p. 612) — Confirms the gamma rhythm is a network property of the isolated inhibitory pool, matching 'interneuron-driven gamma in isolated inhibitory networks'.
  - _"The oscillation frequency is determined both by the net excitation of interneurons and by the kinetics of the inhibitory postsynaptic potentials between them."_ (p. 612, abstract) — Characterises the mechanism governing the interneuron-network gamma frequency, supporting 'characterised interneuron-driven gamma'.


  #divider


  == Wang & Buzsáki (1996) — Gamma Oscillation by Synaptic Inhibition in a Hippocampal Interneuronal Network Model

  #link("https://doi.org/10.1523/JNEUROSCI.16-20-06402.1996")[doi.org/10.1523/JNEUROSCI.16-20-06402.1996]

  *Summary.* Wang and Buzsáki (J. Neurosci. 1996) used simulations of Hodgkin-Huxley single-compartment model neurons coupled purely by GABA\_A inhibitory synapses (no excitatory cells) to test whether gamma (20–80 Hz) oscillations emerge in interneuronal networks. Coherent gamma synchrony arose in sparse, heterogeneous networks under specific conditions: spike afterhyperpolarization above the GABA\_A reversal potential, a sufficiently large ratio of synaptic decay time to oscillation period, modest heterogeneity given fast-spiking cells' steep frequency-current relationship, and a minimal average number of synaptic contacts per cell (insensitive to network size). Network synchronization was confined to the gamma band even as single-cell rates varied gradually.

  *Claim it supports (§1.1).* _"Earlier in vitro recordings [12] and biophysical models [13] characterised interneuron-driven gamma in isolated inhibitory networks, and the synaptic mechanisms that synchronise the interneuron pool have been described [14]."_

  *Quotes.*

  - _"Using computer simulations, we investigated the hypothesis that such rhythmic activity can emerge in a random network of interconnected GABAergic fast-spiking interneurons."_ (p. 6402 (Abstract)) — States the paper models gamma-band rhythmic activity in an isolated network of interconnected inhibitory interneurons.
  - _"We conclude that the GABAA synaptic transmission provides a suitable mechanism for synchronized gamma oscillations in a sparsely connected network of fast-spiking interneurons."_ (p. 6402 (Abstract)) — Concludes that interneuron-driven gamma synchrony arises via GABA\_A inhibition in an isolated interneuronal network, backing the 'biophysical models [13]' citation.
  - _"We found that synaptic transmission via GABAA receptors in a sparsely connected network of model interneurons can provide a mechanism for gamma frequency oscillations"_ (p. 6403 (Introduction)) — Restates the modeling finding that gamma frequency oscillations emerge in an inhibitory-only interneuronal network model.


  #divider


  == Bartos, Vida & Jonas (2007) — Synaptic Mechanisms of Synchronized Gamma Oscillations in Inhibitory Interneuron Networks

  #link("https://doi.org/10.1038/nrn2044")[doi.org/10.1038/nrn2044]

  *Summary.* This Nature Reviews Neuroscience article by Bartos, Vida and Jonas reviews the synaptic mechanisms underlying synchronized gamma (30–90 Hz) oscillations generated by inhibitory interneuron networks, focusing on the hippocampus. It argues gamma relies critically on GABA\_A-mediated inhibition, with fast-spiking, parvalbumin-expressing basket cells forming mutually connected networks playing a key role. Paired-recording experiments show synapses among these interneurons are fast, strong and shunting; computational analysis demonstrates this specialization, plus synaptic delays and network structure, makes interneuron networks robust gamma oscillators tolerant of heterogeneous drive. The article also considers gap junctions and glutamatergic principal-to-interneuron synapses, concluding both promote synchronization.

  *Claim it supports (§1.1).* _"Earlier in vitro recordings [12] and biophysical models [13] characterised interneuron-driven gamma in isolated inhibitory networks, and the synaptic mechanisms that synchronise the interneuron pool have been described [14]."_

  *Quotes.*

  - _"In conclusion, an interneuron network model based on fast, strong and shunting synapses as well as synaptic delays is an efficient gamma frequency oscillator (FIG. 6). Furthermore, such a model tolerates a substantial level of heterogeneity in the drive."_ (Box 2) — Identifies the synaptic mechanisms (fast, strong, shunting synapses plus delays) that synchronise the interneuron pool into a gamma oscillator.
  - _"In summary, both gap junctions and PN–IN synapses promote synchronization (FIG. 6)."_ (p. 54) — Summary of the synaptic and electrical mechanisms that synchronise the interneuron network.


  #divider


  == Kopell, Börgers, Pervouchine, Malerba & Tort (2010) — Gamma and Theta Rhythms in Biophysical Models of Hippocampal Circuits

  #link("https://doi.org/10.1007/978-1-4419-0996-1_15")[doi.org/10.1007/978-1-4419-0996-1_15]

  *Summary.* A book chapter (Chapter 15, Hippocampal Microcircuits, Springer 2010) by Kopell, Borgers, Pervouchine, Malerba, and Tort, reviewing biophysical (reduced Hodgkin-Huxley-type) network models of hippocampal gamma (30-90 Hz) and theta (4-12 Hz) rhythms and their interaction. Cells are single compartments with minimal currents; interneurons are fast-spiking basket cells and oriens lacunosum-moleculare (O-LM) cells. It dissects gamma mechanisms, distinguishing ING, PING, and persistent gamma, and describes when PING produces a coherent rhythm and forms cell assemblies. The emphasis is on modeling dynamical and physiological mechanisms rather than task training; equations, parameters, and simulation code appear in the appendices.

  *Claim it supports (§1.2).* _"PING has been studied extensively in biophysical [15, 16, 17] and neural-mass / mean-field models [18, 19, 20, 21], but these models are descriptive: they are not trained on a task."_

  *Quotes.*

  - _"The PING rhythm has been modeled in a variety of ways, from the very detailed (Traub et al., 1997) to the very simple (Ermentrout and Kopell, 1998)."_ (p. 426 (section "PING")) — States directly that PING has been studied/modeled extensively across a range of biophysical models, backing the 'studied extensively in biophysical models' claim.
  - _"We use simpler biophysical models; all cells have a single compartment only, and the interneurons are restricted to two types: fast-spiking (FS) basket cells and oriens lacunosum-moleculare (O-LM) cells."_ (p. 423 (Introduction)) — Establishes the chapter's approach is a biophysical model of the hippocampal gamma/theta circuit, confirming reference [15] as a biophysical-model source.
  - _"In particular, we wish to highlight the dynamical as well as physiological mechanisms associated with rhythms, and to begin to classify them by mechanisms, not just frequencies."_ (p. 423 (Introduction)) — Shows the models are descriptive/mechanistic in aim (explaining mechanisms), consistent with the manuscript's 'these models are descriptive: they are not trained on a task' framing.


  #divider


  == Viriyopase, Memmesheimer & Gielen (2016) — Cooperation and Competition of Gamma Oscillation Mechanisms

  #link("https://doi.org/10.1152/jn.00493.2015")[doi.org/10.1152/jn.00493.2015]

  *Summary.* This paper examines how two mechanisms of cortical gamma oscillations, interneuron gamma (ING) from mutually coupled inhibitory interneurons and pyramidal-interneuron gamma (PING) from coupled excitatory and inhibitory populations, interact when one network can generate both. Using single-compartment Hodgkin-Huxley-type CA1 models (type I and type II interneurons) and a reduced two-neuron phase model allowing analytical treatment, the authors study inhibitory, excitatory and gap-junction synapses. Centrally, ING and PING compete: the mechanism producing the higher frequency wins and suppresses the other. Gap junctions and inhibitory synapses cooperate or compete depending on interneuron type and coupling strengths.

  *Claim it supports (§1.2).* _"PING has been studied extensively in biophysical [15, 16, 17] and neural-mass / mean-field models [18, 19, 20, 21], but these models are descriptive: they are not trained on a task."_

  *Quotes.*

  - _"pyramidal-interneuron gamma (“PING”), which is mediated by coupled populations of excitatory pyramidal cells and inhibitory interneurons"_ (p. 232 (Introduction)) — Defines PING as one of the two major gamma mechanisms the paper studies.
  - _"In our computer simulations we use a single-compartment Hodgkin-Huxley-type model for CA1 pyramidal (E) cells"_ (p. 233 (Methods, Single-Compartment Hodgkin-Huxley-Type Model)) — Establishes that the paper studies these gamma rhythms with a biophysical (Hodgkin-Huxley-type) model, supporting the 'biophysical models' citation.
  - _"we have modeled a network of the hippocampal region CA1 using data and biologically plausible parameter values from the literature"_ (p. 233 (Results intro)) — Confirms the biophysically detailed, parameter-grounded nature of the PING/ING modeling.


  #divider


  == Brunel & Wang (2003) — What Determines the Frequency of Fast Network Oscillations with Irregular Neural Discharges? I. Synaptic Dynamics and Excitation-Inhibition Balance

  #link("https://doi.org/10.1152/jn.01095.2002")[doi.org/10.1152/jn.01095.2002]

  *Summary.* Brunel and Wang (J. Neurophysiol. 2003) present analytical and simulation work on what sets the frequency of fast (gamma to sharp-wave-ripple, ≈40–200 Hz) oscillations in networks of leaky integrate-and-fire neurons firing sparsely and irregularly under strong noise and recurrent inhibition. Their self-consistent theory predicts population oscillation frequency from synaptic time constants and network parameters, showing frequency depends more on the shortest time constants (latency, rise time) than decay time, and is largely dissociated from single-cell rates. Comparing interneuronal (I-I) and pyramidal-interneuron feedback (E-I) mechanisms, the feedback loop favors slower gamma oscillations; stronger excitation lowers frequency.

  *Claim it supports (§1.2).* _"PING has been studied extensively in biophysical [15, 16, 17] and neural-mass / mean-field models [18, 19, 20, 21], but these models are descriptive: they are not trained on a task."_

  *Quotes.*

  - _"An alternative to the interneuronal network model of fast oscillations is the feedback inhibition model: pyramidal neurons excite interneurons, which in turn send inhibition back onto pyramidal neurons"_ (p. 421, Two population networks) — The paper explicitly models the pyramidal-interneuron feedback loop (PING mechanism) as one of its two oscillation mechanisms.
  - _"Here we show how to derive quantitatively the coherent oscillation frequency for a randomly connected network of leaky integrate-and-fire neurons with realistic synaptic parameters."_ (p. 415, Abstract) — Establishes this is a biophysical spiking-neuron (LIF, realistic synapses) model of fast network oscillations, matching the 'biophysical models' citation.


  #divider


  == Wilson & Cowan (1972) — Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons

  #link("https://doi.org/10.1016/S0006-3495(72")[doi.org/10.1016/S0006-3495(72)86068-5]86068-5)

  *Summary.* Wilson and Cowan derive coupled nonlinear differential equations for a localized neural population with excitatory and inhibitory subpopulations, where E(t) and I(t) are the proportions of each firing per unit time. Using refractoriness, sigmoid response functions and coarse-graining, they reduce integral equations to a two-variable ODE system for phase-plane analysis. Numerical solutions reveal simple and multiple hysteresis (linked to short-term memory) and limit-cycle oscillations whose frequency rises with stimulus intensity. They prove theorems relating connectivity and threshold parameters to multiple stable states and limit cycles, connecting results to evoked potentials and EEG in a mean-field treatment of population dynamics.

  *Claim it supports (§5.3.10).* _"The treatment follows the Wilson–Cowan tradition for cortical-rhythm modelling [18] and the broader population-dynamics and next-generation neural-mass literature [19, 47, 48]."_

  *Quotes.*

  - _"Coupled nonlinear differential equations are derived for the dynamics of spatially localized populations containing both excitatory and inhibitory model neurons."_ (p. 1 (Abstract)) — States the model this paper introduces: a coupled E/I population-dynamics (neural-mass) system.
  - _"Limit cycles have also been used as a model for some of the characteristics of electroencephalogram (EEG) rhythms (Dewan, 1964). In this work the existence of limit cycle oscillations within the central nervous system was assumed without independent evidence. Our present results, therefore, provide a more concrete physiological basis for this approach to the study of EEG rhythms."_ (p. 19) — Directly ties the model's limit-cycle results to cortical-rhythm (EEG) modelling, backing the cortical-rhythm-tradition claim.
  - _"There is a second form of temporal behavior exhibited by our model which is potentially of greater functional significance: the limit cycle."_ (p. 17) — Establishes that the population model produces limit-cycle oscillations, the rhythmic behaviour relevant to cortical rhythms.


  #divider


  == Segneri, Bi, Olmi & Torcini (2020) — Theta-Nested Gamma Oscillations in Next Generation Neural Mass Models

  #link("https://doi.org/10.3389/fncom.2020.00047")[doi.org/10.3389/fncom.2020.00047]

  *Summary.* This paper uses next-generation neural mass models (Montbrio et al. 2015) that exactly reproduce the macroscopic dynamics of networks of quadratic integrate-and-fire neurons, to study theta-nested gamma oscillations. Two setups supporting collective gamma oscillations are examined: pyramidal-interneuronal network gamma (PING, coupled excitatory-inhibitory populations) and interneuronal network gamma (ING, a purely inhibitory population). Driving each system with a sinusoidal theta-forcing near a Hopf bifurcation produces theta-nested gamma oscillations showing phase-amplitude coupling. The authors identify perfectly phase-locked (periodic) and imperfectly locked (quasi-periodic or chaotic) states, finding locked states more common in ING, and match experimental optogenetic findings.

  *Claim it supports (§4.3).* _"Theoretical extensions include multi-layer PING architectures, an independent two-dimensional sweep of $alpha_(E I) times alpha_(I E)$, multi-rhythm and theta-nested gamma models [19, 20], and characterisation of capacity limits for single cycles and minimal assemblies [43]."_

  *Quotes.*

  - _"In both set-ups we observe the emergence of theta-nested gamma oscillations by driving the system with a sinusoidal theta-forcing in proximity of a Hopf bifurcation."_ (p. 1 (Abstract)) — States the paper produces theta-nested gamma oscillations in both PING and ING setups.
  - _"In this framework, we have examined two set-ups able to support collective gamma oscillations: namely, the pyramidal interneuronal network gamma (PING) and the interneuronal network gamma (ING)."_ (p. 1 (Abstract)) — Confirms the model covers PING and theta-nested gamma, matching the cited multi-rhythm/theta-nested gamma extension.
  - _"In this paper we want to compare the two principal mechanisms at the basis of the emergence of collective oscillatory dynamics in neural networks: namely, the PING and ING mechanisms."_ (section 2.1 (Network Models)) — Frames the paper as a mean-field/neural-mass study of PING-type oscillations.


  #divider


  == Nandi, Valla & di Volo (2024) — Bursting Gamma Oscillations in Neural Mass Models

  #link("https://doi.org/10.3389/fncom.2024.1422159")[doi.org/10.3389/fncom.2024.1422159]

  *Summary.* This paper presents an exact neural mass (mean-field) model of excitatory and inhibitory quadratic integrate-and-fire spiking neurons and shows it theoretically predicts a regime of intrinsic bursting gamma (IBG) oscillations arising from deterministic collective chaos, without any external noise source. The authors map a phase diagram containing asynchronous irregular, PING, bistable, and chaotic regimes, and reproduce these in direct spiking-network simulations. They quantify phase-amplitude coupling between fast gamma and slower theta (10 Hz) oscillations, finding IBG shows higher coupling than noise-induced bursting gamma (NiBG), indicating greater capacity for inter-regional information transfer. Results generalise to sparse networks.

  *Claim it supports (§4.3).* _"Theoretical extensions include multi-layer PING architectures, an independent two-dimensional sweep of $alpha_(E I) times alpha_(I E)$, multi-rhythm and theta-nested gamma models [19, 20], and characterisation of capacity limits for single cycles and minimal assemblies [43]."_

  *Quotes.*

  - _"In order to compare these two oscillations' types, we have estimated the PAC of the network to slower theta oscillations (10 Hz) in the two different regimes"_ (p. 2) — Model quantifies phase-amplitude coupling of gamma to slower theta oscillations, backing the theta-nested gamma citation.
  - _"IBG oscillations are distinguished by higher phase-amplitude coupling to slower theta oscillations concerning noise-induced bursting oscillations, thus indicating an increased capacity for information transfer between brain regions"_ (p. 1 (Abstract)) — Abstract states the model characterises gamma-theta phase-amplitude coupling, a multi-rhythm / theta-nested gamma phenomenon.


  #divider


  == Tahvili, Vinck & di Volo (2026) — A Mean-Field Model of Neural Networks with PV and SOM Interneurons Reveals Connectivity-Based Mechanisms of Gamma Oscillations

  #link("https://doi.org/10.1371/journal.pcbi.1014378")[doi.org/10.1371/journal.pcbi.1014378]

  *Summary.* This paper derives a biologically realistic mean-field model of a canonical three-population E-PV-SOM cortical circuit, comprising excitatory pyramidal cells, parvalbumin (PV) and somatostatin (SOM) interneurons. The model robustly generates gamma oscillations whose features match experimental observations, including the relative timing of PV and SOM activity and the effects of optogenetic perturbations.

  *Claim it supports (§1.2).* _"PING has been studied extensively in biophysical [15, 16, 17] and neural-mass / mean-field models [18, 19, 20, 21], but these models are descriptive: they are not trained on a task."_

  *Quotes.*

  - _"Mean-field models offer a powerful theoretical tool for dissecting the origin and stability of oscillations in neuronal networks."_ (p. 2) — Introduction, framing the paper as a mean-field/neural-mass modelling approach to oscillations.
  - _"it has become a standard model for describing canonical mechanisms driving oscillatory regimes in E/I networks such as Pyramidal–Interneuron Gamma (PING) and Interneuron Gamma (ING) in purely inhibidal networks"_ (p. 2) — Introduction, situating the mean-field approach as a descriptive model of PING gamma mechanisms.


  #divider


  == Eshraghian, Ward, Neftci, Wang, Lenz, Dwivedi, Bennamoun, Jeong & Lu (2023) — Training Spiking Neural Networks Using Lessons From Deep Learning

  #link("https://doi.org/10.1109/JPROC.2023.3308088")[doi.org/10.1109/JPROC.2023.3308088]

  *Summary.* A tutorial-and-perspective review on applying deep learning, gradient descent, and backpropagation to biologically plausible spiking neural networks (SNNs). It motivates spikes via the three S's (spikes, sparsity, static suppression), derives the leaky integrate-and-fire neuron in discrete recursive form, and covers spike encoding/decoding and objective/regularization functions. It surveys training methods: shadow (ANN-to-SNN) conversion, backpropagation using spike times (SpikeProp), and, as the most adopted solution to the non-differentiable dead-neuron problem, surrogate-gradient descent via backpropagation through time on the unrolled graph. It also discusses links to quantized networks, online learning (RTRL variants), and STDP, with interactive tutorials using the authors' snnTorch package.

  *Claim it supports (§1.2).* _"The parallel literature on trainable spiking neural networks uses surrogate-gradient descent for end-to-end optimisation [22, 23, 24]; the resulting networks are typically current-based and non-rhythmic."_

  *Quotes.*

  - _"Instead of computing the gradient with respect to spike times, the most commonly adopted approach over the past several years is to apply the generalised backpropagation algorithm to the unrolled computational graph (Figure 6(b))"_ (p. 26, Section 4.3 (Backpropagation Using Spikes)) — States that BPTT on the unrolled graph is the most commonly adopted training approach, supporting 'end-to-end optimisation' via backprop.
  - _"Surrogate gradients have been used in most state-of-the-art experiments that natively train an SNN"_ (p. 28, Section 4.3.1 (Surrogate Gradients)) — Directly supports the claim that trainable-SNN literature uses surrogate-gradient descent.
  - _"A major advantage of surrogate gradients is they help with overcoming the dead neuron problem."_ (p. 26, Section 4.3.1 (Surrogate Gradients)) — Establishes surrogate gradients as the method used to make spiking non-differentiability trainable end-to-end.


  #divider


  == Neftci, Mostafa & Zenke (2019) — Surrogate Gradient Learning in Spiking Neural Networks

  #link("https://doi.org/10.1109/MSP.2019.2931595")[doi.org/10.1109/MSP.2019.2931595]

  *Summary.* This tutorial and review (Neftci, Mostafa and Zenke, IEEE Signal Processing Magazine, 2019) covers training spiking neural networks. It maps spiking networks onto recurrent neural networks so standard machinery (backpropagation through time, real-time recurrent learning) applies. Two obstacles are identified: the non-differentiable binary spiking nonlinearity and the computational, memory and locality costs of optimization. It contrasts smoothed models (soft-nonlinearity, probabilistic, rate-coding, single-spike-timing) with surrogate gradient methods, which replace the spiking nonlinearity's derivative with a smooth surrogate while keeping the model unchanged. It reviews applications with benchmarks like MNIST, CIFAR10 and DVS gesture, linking machine learning, neuroscience and neuromorphic computing.

  *Claim it supports (§1.2).* _"The parallel literature on trainable spiking neural networks uses surrogate-gradient descent for end-to-end optimisation [22, 23, 24]; the resulting networks are typically current-based and non-rhythmic."_

  *Quotes.*

  - _"Finally, the use of SGs allows to efficiently train SNNs end-to-end without the need to specify which coding scheme is to be used in the hidden layers."_ (p. 13) — States directly that surrogate gradients enable end-to-end training of SNNs, backing the claim's 'surrogate-gradient descent for end-to-end optimisation.'
  - _"SG methods provide an alternative approach to overcoming the difficulties associated with the discontinuous nonlinearity."_ (p. 12 (Sec. IV-B, Surrogate gradients)) — Defines surrogate-gradient methods as the technique for training past the non-differentiable spike, the method the citing sentence attributes to this literature.
  - _"it gives an overview of existing approaches and provides an introduction to surrogate gradient methods, specifically, as a particularly flexible and efficient method to overcome the aforementioned challenges."_ (p. 1 (Abstract)) — Establishes that the paper's core subject is surrogate-gradient methods for training SNNs, supporting its inclusion in the cited literature.


  #divider


  == Deckers et al. (2025) — Advancing Spatio-Temporal Processing Through Adaptation in Spiking Neural Networks

  #link("https://doi.org/10.1038/s41467-025-60878-z")[doi.org/10.1038/s41467-025-60878-z]

  *Summary.* This Nature Communications article analyzes the dynamical, computational, and learning properties of adaptive leaky integrate-and-fire (adLIF) neurons and recurrent spiking networks built from them, all trained end-to-end with backpropagation through time using surrogate gradients. It identifies stability and parameterization problems caused by the standard Euler-Forward discretization and shows that a Symplectic-Euler discretization fixes them. The resulting SE-adLIF networks outperform LIF baselines and state-of-the-art spiking models on speech recognition (SHD, SSC), an ECG dataset, a spring-mass trajectory-prediction task, and neuromorphic audio compression, exploiting the neurons' oscillatory, frequency-selective dynamics without normalization techniques.

  *Claim it supports (§1.2).* _"The parallel literature on trainable spiking neural networks uses surrogate-gradient descent for end-to-end optimisation [22, 23, 24]; the resulting networks are typically current-based and non-rhythmic."_

  *Quotes.*

  - _"Recent advances in SNN research have shown that SNNs can be trained in a similar manner as ANNs using backpropagation through time (BPTT), leading to highly accurate models"_ (p. 1) — Introduction stating SNNs are trained end-to-end via BPTT.
  - _"We trained both the adLIF and LIF SNNs using BPTT with surrogate gradients"_ (p. 8) — Methods/Results describing surrogate-gradient training of the networks.


  #divider


  == Yan, Yang, Wu, Liu, Zhang, Li, Tan & Wu (2025) — Efficient and Robust Temporal Processing with Neural Oscillations Modulated Spiking Neural Networks

  #link("https://doi.org/10.1038/s41467-025-63771-x")[doi.org/10.1038/s41467-025-63771-x]

  *Summary.* This paper introduces Rhythm-SNN, a spiking neural network that draws on the brain's neural oscillation mechanism. It employs heterogeneous oscillatory signals to modulate spiking neurons, forcing them to activate periodically at distinct frequencies via ON/OFF states. The authors show this significantly reduces neuronal firing rates while improving temporal-processing capability, working-memory capacity, and robustness to noise and adversarial attacks. Rhythm modulation is applied across many SNN architectures and evaluated on tasks including S-MNIST, PS-MNIST, SHD, ECG, GSC, VoxCeleb1, PTB, and DVS-Gesture, plus the Intel N-DNS speech-enhancement challenge, achieving state-of-the-art accuracy with up to orders-of-magnitude lower energy cost.

  *Claim it supports (§3.5).* _"[25] imposes the oscillation as an external input to spiking neurons; [26] trains an adaptive-LIF network on speech, all parameters free, and reports that oscillatory synchronisation and cross-frequency coupling _emerge_ from end-to-end optimisation, correlating with task performance."_

  *Quotes.*

  - _"our approach utilizes external heterogeneous oscillatory signals to modulate neuronal dynamics, thereby facilitating the encoding, transmission, and integration of information across various timescales"_ (p. 9, Discussion) — Paper describes its own mechanism as external oscillatory signals modulating neuronal dynamics.
  - _"we employ heterogeneous oscillatory signals to modulate spiking neurons, enforcing them to activate periodically at distinct frequencies"_ (p. 1, Abstract) — Abstract states oscillatory signals are imposed to modulate spiking neurons at distinct frequencies.
  - _"speech recognition on Spiking Heidelberg Digits (SHD)"_ (p. 3, temporal-processing section) — SHD is one of the temporally structured tasks the paper evaluates on, supporting that [25] evaluates on SHD.


  #divider


  == Bittar & Garner (2024) — Exploring Neural Oscillations During Speech Perception via Surrogate-Gradient Spiking Neural Networks

  #link("https://doi.org/10.3389/fnins.2024.1449181")[doi.org/10.3389/fnins.2024.1449181]

  *Summary.* Bittar and Garner present a physiologically inspired, end-to-end trainable speech recognition architecture built around a multi-layered surrogate-gradient spiking neural network using adaptive-LIF neurons. Trained with gradient descent on speech (TIMIT, Librispeech, Google Speech Commands) to predict phoneme/subword sequences, the network develops neural oscillations and significant cross-frequency (phase-amplitude) couplings across delta, theta, alpha, beta and gamma bands as an emergent property, without imposing any oscillatory prior. These synchronization phenomena arise only from trained networks processing real speech and correlate with improved recognition performance. The paper also examines how spike-frequency adaptation, recurrence, delays and Dale's law shape the observed oscillatory activity.

  *Claim it supports (§3.5).* _"[25] imposes the oscillation as an external input to spiking neurons; [26] trains an adaptive-LIF network on speech, all parameters free, and reports that oscillatory synchronisation and cross-frequency coupling _emerge_ from end-to-end optimisation, correlating with task performance."_

  *Quotes.*

  - _"We present a physiologically inspired speech recognition architecture, compatible and scalable with deep learning frameworks, and demonstrate that end-to-end gradient descent training leads to the emergence of neural oscillations in the central spiking neural network."_ (p. 1 (Abstract)) — Abstract states oscillations emerge from end-to-end gradient descent training of the SNN.
  - _"The subsequent analysis of the spiking activity across our trained networks in response to speech stimuli revealed that neural oscillations, commonly associated with various cognitive processes in the brain, did emerge from training an architecture to recognize words or phonemes."_ (p. 15 (Discussion)) — Discussion confirms oscillations emerged from training on speech to recognize words/phonemes.
  - _"Our networks' ability to synchronize oscillatory activity in the last layer was also associated with improved speech recognition performance, which points to a functional role for neural oscillations in auditory processing."_ (p. 15 (Discussion)) — Synchronized oscillation correlates with improved recognition performance (task performance).


  #divider


  == Barth & Poulet (2012) — Experimental Evidence for Sparse Firing in the Neocortex

  #link("https://doi.org/10.1016/j.tins.2012.03.008")[doi.org/10.1016/j.tins.2012.03.008]

  *Summary.* Barth and Poulet (Trends in Neurosciences, 2012) review evidence that neocortical neurons fire sparsely — firing neurons are rare, silent ones common. They outline four scenarios producing apparent sparseness (trial-to-trial variability, small fixed ensembles, brain-state dependence, stimulus specificity) and review laminar recording and calcium-imaging data across sensory cortices. Granular (layer 4) and infragranular (layers 5/6) layers show highest firing output, while supragranular (layer 2/3) neurons fire much less, with many cells silent. They argue anesthesia and non-optimal stimuli inflate apparent sparseness, and that strong recurrent inhibition restricts spiking to a minority of pyramidal cells.

  *Claim it supports (§1.3).* _"Cortical pyramidal cells fire at low rates (typically below 10 Hz) under strong recurrent input [27], and the cortical metabolic budget is dominated by excitatory spike generation, with inhibitory spikes substantially less costly per spike [28, 29]."_

  *Quotes.*

  - _"Given the selective pressure that must have driven the expansion of the cerebral cortex, it is surprising that many neocortical neurons show very low firing rates."_ (p. 345, Introduction) — Establishes the paper's central premise that neocortical neurons fire at very low rates.
  - _"Strong recurrent inhibition reduces overall firing output from pyramidal cells, increasing the sparseness of response probabilities."_ (p. 351, Box 1) — Supports the 'under strong recurrent input' portion of the claim, linking recurrent inhibition to reduced pyramidal firing.


  #divider


  == Attwell & Laughlin (2001) — An Energy Budget for Signaling in the Grey Matter of the Brain

  #link("https://doi.org/10.1097/00004647-200110000-00001")[doi.org/10.1097/00004647-200110000-00001]

  *Summary.* This review by Attwell and Laughlin builds a bottom-up energy budget for signaling in mammalian grey matter, mostly rodent, estimating the ATP cost of generating and transmitting signals. Treating cells as largely glutamatergic, they partition energy among action potentials, postsynaptic ionic fluxes, presynaptic calcium entry, glutamate recycling, and resting potentials. At a 4 Hz mean firing rate, they attribute ∼47% to action potentials, ∼34% to postsynaptic receptors, ∼13% to resting potentials, and ∼3% each to presynaptic Ca2+ and recycling, most spent by the Na+/K+ pump. They argue high energy use favors sparse codes and fMRI signals track glutamatergic energy.

  *Claim it supports (§3.6).* _"Excitatory glutamatergic signalling accounts for a larger share of the cortical energy budget than inhibitory transmission [28, 29]; weighting spikes by metabolic cost recovers the order-of-magnitude figure."_

  *Quotes.*

  - _"energy expenditure on glutamatergic signaling is indeed significant, approximately 34% of the total signaling energy usage going on glutamate postsynaptic actions in rodents and perhaps 74% in humans"_ (p. 1143) — Quantifies the large share of the cortical signaling energy budget attributable to excitatory glutamatergic postsynaptic actions.


  #divider


  == Howarth, Gleeson & Attwell (2012) — Updated Energy Budgets for Neural Computation in the Neocortex and Cerebellum

  #link("https://doi.org/10.1038/jcbfm.2012.35")[doi.org/10.1038/jcbfm.2012.35]

  *Summary.* A review in the Journal of Cerebral Blood Flow & Metabolism (Howarth, Gleeson, Attwell, 2012) revising bottom-up energy budgets for neural computation in cortex and cerebellum, incorporating findings that mammalian action potentials are far more energy-efficient than earlier estimates. Using ion-flux data, they partition ATP use across resting potentials, action potentials, postsynaptic receptors, transmitter recycling, and presynaptic Ca2+ entry. For cortex they predict 50% to postsynaptic glutamate receptors, 21% to action potentials, 20% to resting potentials, 5% to release, 4% to recycling.

  *Claim it supports (§3.6).* _"Excitatory glutamatergic signalling accounts for a larger share of the cortical energy budget than inhibitory transmission [28, 29]; weighting spikes by metabolic cost recovers the order-of-magnitude figure."_

  *Quotes.*

  - _"we assume for simplicity that all neurons in the cortex are glutamatergic, as 90% of cells and synapses are glutamatergic"_ (p. 1224) — The cortical energy budget treats essentially all signaling as excitatory/glutamatergic because glutamatergic cells and synapses dominate the cortex.
  - _"In cerebral cortex, most signaling energy (50%) is used on postsynaptic glutamate receptors"_ (p. 1222 (Abstract)) — Half of cortical signaling energy is attributed to postsynaptic glutamate (excitatory) receptors, the single largest budget item.
  - _"inhibitory neurons are predicted to use 25% of the signaling energy, and excitatory neurons to use 75%"_ (p. 1227) — Explicit excitatory-vs-inhibitory split showing excitatory signalling takes the far larger share of the (cerebellar) energy budget.


  #divider


  == Ainsworth, Lee, Cunningham, Traub, Kopell & Whittington (2012) — Rates and Rhythms: A Synergistic View of Frequency and Temporal Coding in Neuronal Networks

  #link("https://doi.org/10.1016/j.neuron.2012.08.004")[doi.org/10.1016/j.neuron.2012.08.004]

  *Summary.* This Neuron review by Ainsworth, Lee, Cunningham, Traub, Kopell and Whittington examines how small populations of cortical neurons code sensory inputs and motor outputs, contrasting rate coding with temporal coding. Rather than treating these as mutually exclusive, the authors argue for a synergistic view. Reviewing in vitro and in vivo evidence, they show that cortical circuits are biased toward temporal coding via gamma-rhythm-imposed inhibition, but that rate and temporal codes coexist. They highlight coexisting gamma rhythms in sensory cortex whose population frequency tracks principal-cell spike rates, providing a mechanistic substrate for combining rate and temporal codes according to stimulus strength.

  *Claim it supports (§1.3).* _"In the architecture studied here the rhythm sets the rate, so rate and timing are not independent codes but synergistic descriptions of a single dynamics [30]."_

  *Quotes.*

  - _"The coincident expression of multiple types of gamma rhythm in sensory cortex suggests a mechanistic substrate for combining rate and temporal codes on the basis of stimulus strength."_ (p. 572) — Abstract thesis linking rate and temporal codes via gamma rhythm.


  #divider


  == Schaefer, Angelo, Spors & Margrie (2006) — Neuronal Oscillations Enhance Stimulus Discrimination by Ensuring Action Potential Precision

  #link("https://doi.org/10.1371/journal.pbio.0040163")[doi.org/10.1371/journal.pbio.0040163]

  *Summary.* This paper (Schaefer, Angelo, Spors & Margrie, PLoS Biology 2006) uses in vivo and in vitro recordings plus theoretical models to show that membrane potential oscillations dramatically improve action potential (AP) precision by removing membrane potential variance associated with jitter-accumulating spike trains. Oscillations achieve this because their trough establishes a period of hyperpolarization that prevents accumulation of AP jitter. The authors demonstrate this effect across mitral, CA1 pyramidal, and Purkinje cells, and in integrate-and-fire and conductance-based simulations.

  *Claim it supports (§3.2).* _"It is consistent with prior reports that oscillations sharpen spike-timing precision [31]."_

  *Quotes.*

  - _"both synaptically and intrinsically generated membrane potential oscillations dramatically improve action potential (AP) precision by removing the membrane potential variance associated with jitter-accumulating trains of APs"_ (p. 1010 (Abstract)) — Abstract statement of the core finding that oscillations improve AP timing precision.


  #divider


  == Nguyen & Rubchinsky (2021) — Temporal Patterns of Synchrony in a Pyramidal-Interneuron Gamma (PING) Network

  #link("https://doi.org/10.1063/5.0042451")[doi.org/10.1063/5.0042451]

  *Summary.* Nguyen and Rubchinsky study fine temporal patterning of synchronization in a conductance-based network of two synaptically connected pyramidal-interneuron gamma (PING) circuits, each with two excitatory and two inhibitory Hodgkin-Huxley-type neurons. Using Hilbert-transform phase analysis and first-return maps, they quantify a synchronization index and durations of desynchronization events. Gamma oscillations are only partially synchronized, alternating short synchronized and desynchronized intervals, and changing synaptic strengths and potassium current kinetics alters this patterning. Crucially, temporal patterning can vary independently of average synchrony strength and firing frequency, tending toward short desynchronizations, which they argue makes networks more efficient at forming transient neural assemblies.

  *Claim it supports (§2.6.1).* _"Two inference-time jitter perturbations of the inhibitory spike train, both holding the mean inhibitory rate fixed, produce opposite effects on the excitatory rate: per-cell jitter smears each burst into a continuous shunt and reduces the excitatory rate to zero, while cycle-coherent jitter preserves within-burst synchrony and raises the excitatory rate from $approx 8$ Hz to $approx 50$ Hz (Figure 10) [32]."_

  *Quotes.*

  - _"In this study, we use a conductance-based neural network exhibiting pyramidal-interneuron (PING) gamma rhythm to study the temporal patterning of synchronized neural oscillations."_ (Abstract, p. 1) — Establishes that the paper characterises temporal patterns of synchrony in a PING circuit, the general topic the manuscript cites it for.
  - _"The study shows how gamma rhythm can be partially synchronized with specific temporal patterning and how this temporal patterning of gamma synchronization is regulated by connectivity strength and other factors."_ (Significance statement, p. 1) — States that temporal patterning of PING gamma synchrony depends on connectivity, backing the manuscript's appeal to prior characterisations of temporal-synchrony patterns in PING circuits.


  #divider


  == Shadlen & Movshon (1999) — Synchrony Unbound: A Critical Evaluation of the Temporal Binding Hypothesis

  #link("https://doi.org/10.1016/S0896-6273(00")[doi.org/10.1016/S0896-6273(00)80822-3]80822-3)

  *Summary.* Shadlen and Movshon (Neuron, 1999) critically evaluate the temporal binding hypothesis — that the brain solves visual binding via a temporal code in which neurons representing an object's features synchronize their spike timing. They argue the hypothesis is conceptually incomplete and empirically weak: cortical neurons receive so much convergent input that apparent synchrony is hard to distinguish from chance, and cortex lacks a clear mechanism to decode synchronous spikes. The evidence shows correlated activity but does not compellingly link it to binding, whereas rate-modulated activity consistently tracks perceptual states. They suggest binding may instead be a high-level, rate-encoded computation.

  *Claim it supports (§3.2).* _"The asymmetry constitutes a testable prediction for biological gamma circuits and would speak to long-standing rate-vs-timing debates [33, 34]."_

  *Quotes.*

  - _"There is ample experimental evidence for correlated cortical activity but little that directly or compellingly links this activity to binding. In contrast, there is considerable evidence that the rate-modulated activity of cortical cell populations is crucial in mediating perceptual binding."_ (Conclusion) — Verdict that rate coding, not spike-timing synchrony, tracks perceptual binding.


  #divider


  == London, Roth, Beeren, Häusser & Latham (2010) — Sensitivity to Perturbations in vivo Implies High Noise and Suggests Rate Coding in Cortex

  #link("https://doi.org/10.1038/nature09086")[doi.org/10.1038/nature09086]

  *Summary.* Using in vivo whole-cell patch-clamp recordings in rat barrel cortex, London, Roth, Beeren, Hausser and Latham show that a single extra spike added to one excitatory neuron produces on average about 28 additional spikes in its postsynaptic targets, and that a single spike causes a detectable rise in local firing rate. This large amplification of perturbations implies high intrinsic noise (membrane-potential fluctuations of order 2.2-4.5 mV that carry no stimulus information), which puts severe constraints on precise spike-timing codes. The authors argue their findings are consistent with the cortex using primarily a rate code rather than precise spike timing.

  *Claim it supports (§3.2).* _"The asymmetry constitutes a testable prediction for biological gamma circuits and would speak to long-standing rate-vs-timing debates [33, 34]."_

  *Quotes.*

  - _"Our findings are thus consistent with the idea that cortex is likely to use primarily a rate code."_ (p. 123) — Abstract conclusion arguing for rate coding over timing.
  - _"our results indicate that there is a large amount of intrinsic noise in cortex, and that this noise puts severe constraints on spike timing codes"_ (p. 126) — Discussion: noise constrains precise spike-timing codes.
  - _"Our results, on the other hand, apply to slowly varying stimuli and higher-order computations, and suggest that in those cases the cortex does not rely on precise spike timing."_ (p. 126) — Discussion contrasting rate vs timing regimes.


  #divider


  == Akam & Kullmann (2012) — Efficient \"Communication through Coherence\" Requires Oscillations Structured to Minimize Interference between Signals

  #link("https://doi.org/10.1371/journal.pcbi.1002760")[doi.org/10.1371/journal.pcbi.1002760]

  *Summary.* Akam and Kullmann use a mathematical model to test the "communication through coherence" (CTC) hypothesis, which holds that selective communication between neural networks arises from coherence between firing-rate oscillation in a sending region and gain modulation in a receiving region. Modeling multiple population-coded input networks converging on a receiving network, they show selective communication can be achieved but requires the target input to differ from distractors in amplitude, phase, or frequency of oscillation. When distractors oscillate incoherently in the same frequency band, accuracy is severely degraded, and target modulation must be strong for a good signal-to-noise ratio.

  *Claim it supports (§3.3).* _"This is consistent with — though not specifically diagnostic of — the gamma cycle acting as a temporal unit for classification, in the spirit of the communication-through-coherence framework [2, 35]."_

  *Quotes.*

  - _"The 'communication through coherence' (CTC) hypothesis proposes that selective communication among neural networks is achieved by coherence between firing rate oscillation in a sending region and gain modulation in a receiving region."_ (p. 1 (Abstract)) — Paper's opening statement of the CTC framework it investigates.
  - _"We show that selective communication can indeed be achieved, although the structure of oscillatory activity in the target and distracting networks must satisfy certain previously unrecognized constraints."_ (p. 1 (Abstract)) — Main finding: CTC works but only under structural constraints on oscillations.
  - _"To selectively route the information encoded in one input network (the 'target' input) to the output of the receiving network, a top-down control signal imposes an oscillatory modulation on the target network firing rate and a coherent oscillatory gain modulation in the receiving region."_ (p. 6 (Figure 1 caption)) — Describes the coherence-based routing mechanism central to CTC.


  #divider


  == Renart, de la Rocha, Bartho, Hollender, Parga, Reyes & Harris (2010) — The Asynchronous State in Cortical Circuits

  #link("https://doi.org/10.1126/science.1179850")[doi.org/10.1126/science.1179850]

  *Summary.* This paper by Renart et al. (Science, 2010) asks why cortical neurons, despite dense connectivity and shared input, show near-zero spiking correlations. Using analysis of densely connected recurrent binary-neuron networks, confirmed with integrate-and-fire simulations and in vivo rat neocortex recordings, they show recurrent dynamics generate an asynchronous state where excitatory and inhibitory population fluctuations track each other, producing negative correlations between synaptic currents that cancel shared input. This active decorrelation arises from a dynamic balance of large excitatory and inhibitory currents, keeping mean correlations weak when inhibition is strong and fast, while permitting a wide distribution of pairwise values.

  *Claim it supports (§3.4).* _"PING is a structured alternative to the asynchronous balanced state [36, 37]."_

  *Quotes.*

  - _"Here we show theoretically that recurrent neural networks can generate an asynchronous state characterized by arbitrarily low mean spiking correlations despite substantial amounts of shared input."_ (p. 587 (abstract)) — States the paper's central claim that recurrent networks produce an asynchronous state, the state the manuscript contrasts with PING.
  - _"In this state, spontaneous fluctuations in the activity of excitatory and inhibitory populations accurately track each other, generating negative correlations in synaptic currents which cancel the effect of shared input."_ (p. 587 (abstract)) — Describes the balanced E/I mechanism underlying the asynchronous state, the 'balanced state' the manuscript names.


  #divider


  == van Vreeswijk & Sompolinsky (1996) — Chaos in Neuronal Networks with Balanced Excitatory and Inhibitory Activity

  #link("https://doi.org/10.1126/science.274.5293.1724")[doi.org/10.1126/science.274.5293.1724]

  *Summary.* A theoretical paper (van Vreeswijk & Sompolinsky, Science 1996) investigating whether irregular cortical spiking arises from balance of excitation and inhibition. They study a network of two-state neurons, sparsely connected by strong synapses (order sqrt(K) inputs cross threshold while each receives order K inputs). A balanced state emerges without fine-tuning: activities self-adjust so large excitation and inhibition nearly cancel, leaving net input whose mean and fluctuations are of order the threshold.

  *Claim it supports (§3.4).* _"PING is a structured alternative to the asynchronous balanced state [36, 37]."_

  *Quotes.*

  - _"Such a balance emerges naturally in large networks of excitatory and inhibitory neuronal populations that are sparsely connected by relatively strong synapses. The resulting state is characterized by strongly chaotic dynamics, even when the external inputs to the network are constant in time."_ (p. 1724 (abstract)) — Defines the balanced state as an emergent, unstructured (asynchronous chaotic) mechanism for irregular activity, the alternative PING is contrasted with.
  - _"Thus, the balance of the inhibition and excitation is an emergent property of the network dynamics and does not require precise tuning of the network parameters."_ (p. 1725) — States the balanced state arises without fine-tuning, characterizing it as a self-organizing asynchronous regime rather than a structured rhythmic one.
  - _"The ability of our network to react to changes in its environment on time scales that are much shorter than the update times of the individual neurons is a result of the combination of the large synaptic gain (on the order of VlK) and the asynchronous nature of the dynamics."_ (p. 1725) — Explicitly labels the dynamics of the balanced state as asynchronous, supporting the manuscript's term 'asynchronous balanced state'.


  #divider


  == Vogels, Sprekeler, Zenke, Clopath & Gerstner (2011) — Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways and Memory Networks

  #link("https://doi.org/10.1126/science.1211095")[doi.org/10.1126/science.1211095]

  *Summary.* This Science report by Vogels, Sprekeler, Zenke, Clopath and Gerstner proposes that cortical excitatory-inhibitory balance is established and maintained by experience-dependent plasticity at inhibitory synapses. In a conductance-based integrate-and-fire model, inhibitory synapses are made plastic under a Hebbian spike-timing-dependent rule (coincident spikes potentiate, presynaptic spikes alone depress, with a target-rate offset) while excitatory synapses stay fixed. This self-organizes a precise, per-channel E/I balance producing sparse firing to natural stimuli, matches auditory-cortex cotuning and receptive-field plasticity, and yields asynchronous irregular states in recurrent networks. Inhibitory plasticity can also embed hidden yet reactivatable Hebbian cell-assembly memories.

  *Claim it supports (§3.4).* _"The architectural treatment of the loop adopted here differs from the inhibitory-plasticity literature, in which the inhibitory connectivity is plastic and learns E/I balance [38, 39, 40, 41]."_

  *Quotes.*

  - _"Such a balance could be established and maintained in an experience-dependent manner by synaptic plasticity at inhibitory synapses."_ (p. 1569 (abstract)) — States the paper's core thesis: E/I balance arises from plasticity at inhibitory synapses.
  - _"we investigated the hypothesis that synaptic plasticity at inhibitory synapses plays a central role in balancing the excitatory and inhibitory inputs a cell receives."_ (p. 1569) — Frames the study around plastic inhibition learning E/I balance, matching the citing sentence's characterization.
  - _"Whereas inhibitory synapses were plastic, the efficacies of the excitatory model synapses were fixed at the beginning of a simulation and left unchanged unless otherwise noted."_ (p. 1570) — Confirms the architecture: inhibitory connectivity is the plastic element, distinct from fixed excitation.


  #divider


  == Hennequin, Agnes & Vogels (2017) — Inhibitory Plasticity: Balance, Control, and Codependence

  #link("https://doi.org/10.1146/annurev-neuro-072116-031005")[doi.org/10.1146/annurev-neuro-072116-031005]

  *Summary.* Hennequin, Agnes and Vogels (Annu. Rev. Neurosci. 2017) review inhibitory synaptic plasticity (ISP) and its computational roles. Surveying experimental GABAergic spike-timing-dependent learning rules, they describe how modelers use ISP to control network dynamics, arguing that excitation/inhibition balance is central to neuronal processing and ISP is an ideal mechanism to stabilize it. They cover how ISP tunes inhibition to balance excitation (stabilizing firing rates, enabling detailed balance, compensating for wiring heterogeneity), the role of inhibition in memory engrams, and negative feedback control (amplifying networks, short-term memory, inhibition-stabilized networks). They advocate codependent plasticity where synaptic changes depend on neighboring-synapse activity.

  *Claim it supports (§3.4).* _"The architectural treatment of the loop adopted here differs from the inhibitory-plasticity literature, in which the inhibitory connectivity is plastic and learns E/I balance [38, 39, 40, 41]."_

  *Quotes.*

  - _"These learning rules automatically and robustly tune the average inhibitory input in single cells, effectively balancing them with the excitatory inputs and stabilizing postsynaptic firing rates near a target value"_ (p. 561 (Section 3)) — States that Hebbian iSTDP rules make inhibitory synapses plastic so inhibition learns to balance excitation.
  - _"Any positive (respectively, negative) deviation in postsynaptic firing rate from the target is soon suppressed by strengthening (respectively, weakening) of inhibitory input synapses."_ (p. 562 (Section 3)) — Describes the plastic modification of inhibitory synapses that maintains E/I balance.
  - _"Through ISP, they become closely aligned—that is, exhibit detailed balance (Figure 4g) (Vogels et al. 2011)—even when perturbed by excitatory"_ (p. 563 (Section 4)) — Confirms inhibitory synaptic plasticity aligns inhibitory with excitatory weights to achieve E/I balance.


  #divider


  == Wu, Miehl & Gjorgjieva (2022) — Regulation of Circuit Organization and Function Through Inhibitory Synaptic Plasticity

  #link("https://doi.org/10.1016/j.tins.2022.10.006")[doi.org/10.1016/j.tins.2022.10.006]

  *Summary.* This Trends in Neurosciences (2022) review by Wu, Miehl, and Gjorgjieva synthesizes experimental and computational work on plasticity of inhibitory synapses onto excitatory neurons. It is organized around several roles: controlling excitation across spatiotemporal scales (network rates, single-neuron spiking, dendritic calcium), maintaining excitation/inhibition balance, controlling excitatory plasticity, and shaping structured connectivity like receptive fields and assemblies. It reviews inhibitory spike-timing-dependent plasticity rules, inhibition-stabilized networks, tight versus loose E/I balance, and interneuron-specific plasticity across PV, SST, VIP, and NDNF subtypes. It argues inhibitory plasticity is a key, flexible mechanism for circuit dynamics, with many open questions given interneuron diversity.

  *Claim it supports (§3.4).* _"The architectural treatment of the loop adopted here differs from the inhibitory-plasticity literature, in which the inhibitory connectivity is plastic and learns E/I balance [38, 39, 40, 41]."_

  *Quotes.*

  - _"Various inhibitory plasticity rules have been proposed to regulate E/I balance in computational models"_ (p. 887) — States that the inhibitory-plasticity literature uses plastic inhibition to regulate E/I balance in computational models, directly matching the cited claim.
  - _"The learning rule achieves the balance by a negative feedback mechanism, which increases inhibitory synaptic strength for high postsynaptic firing rates and decreases inhibitory strength for low firing rates to counteract deviations from a target firing rate"_ (p. 888) — Describes how the inhibitory connectivity itself is plastic (inhibitory strength changes) to learn/maintain E/I balance.
  - _"Inhibitory plasticity can establish excitation/inhibition (E/I) balance, control neuronal firing, and affect local calcium concentration"_ (p. 884 (Abstract)) — Abstract statement that inhibitory (i.e. plastic inhibitory) plasticity establishes E/I balance, supporting the characterization of this literature.


  #divider


  == Páscoa dos Santos & Verschure (2025) — Excitatory-Inhibitory Homeostasis and Bifurcation Control in the Wilson-Cowan Model of Cortical Dynamics

  #link("https://doi.org/10.1371/journal.pcbi.1012723")[doi.org/10.1371/journal.pcbi.1012723]

  *Summary.* This PLOS Computational Biology paper by Páscoa dos Santos and Verschure uses the Wilson-Cowan neural-mass model to study how different modes of excitatory-inhibitory (E-I) homeostasis shape cortical oscillations and edge-of-bifurcation dynamics. Whereas prior computational work focused solely on plasticity of inhibition, the authors derive analytical fixed points for four homeostatic modes (synaptic scaling of excitation, scaling of inhibition, and two forms of intrinsic-excitability plasticity) and their combinations.

  *Claim it supports (§3.4).* _"The architectural treatment of the loop adopted here differs from the inhibitory-plasticity literature, in which the inhibitory connectivity is plastic and learns E/I balance [38, 39, 40, 41]."_

  *Quotes.*

  - _"computational studies on E-I homeostasis have focused solely on the plasticity of inhibition, neglecting the impact of different modes of E-I homeostasis on cortical dynamics"_ (p. 1 (Abstract)) — Abstract states prior computational E-I homeostasis studies focused only on plasticity of inhibition.
  - _"while previous studies have limited E-I homeostasis to the plasticity of inhibition, we explore the wide range of mechanisms employed by cortical networks"_ (p. 1 (Author summary)) — Author summary contrasts this work with prior studies limiting E-I homeostasis to plasticity of inhibition.


  #divider


  == Kann (2016) — The Interneuron Energy Hypothesis: Implications for Brain Disease

  #link("https://doi.org/10.1016/j.nbd.2015.08.005")[doi.org/10.1016/j.nbd.2015.08.005]

  *Summary.* A review by Oliver Kann articulating the interneuron energy hypothesis: fast-spiking parvalbumin-positive (PV+) basket cells, which fire at high frequency (\>100 Hz) and synchronize excitatory principal cells to generate gamma oscillations (30-100 Hz), have exceptionally high per-cell energy demands. Evidence shows they are enriched in mitochondria and cytochrome c oxidase, depend on oxidative phosphorylation, and spend energy on ion transport, presynaptic GABA release, and GABA reuptake. This makes them highly sensitive to metabolic and oxidative stress, lowering the threshold for impaired fast oscillations.

  *Claim it supports (§3.6).* _"The argument is complicated by the substantial per-cell metabolic demands of fast-spiking interneurons, which sustain high firing rates and dense synaptic activity [42]; for this reason we treat the uniform-spike-counting reduction (approximately fivefold) as the more conservative claim."_

  *Quotes.*

  - _"The high energy expenditure of neurons during gamma oscillations is most likely caused by increased rates of action potentials and postsynaptic potentials."_ (p. 78, Section 3 (Bioenergetics of gamma oscillations)) — Establishes that high firing rates and synaptic activity are the dominant driver of the high metabolic cost during gamma oscillations.
  - _"Fast-spiking, PV+ interneurons can generate action potentials at high frequencies (\>100 Hz) for prolonged periods of time, with weak or no accommodation"_ (p. 78, Section 4 (Interneuron energy hypothesis)) — Documents the sustained high firing rates of fast-spiking interneurons that underlie their metabolic demand.
  - _"several findings indicate that fast-spiking interneurons utilize much more energy than other cortical neurons."_ (p. 78, Section 4 (Interneuron energy hypothesis)) — States directly that fast-spiking interneurons have substantially higher per-cell energy demands than other cortical neurons.


  #divider


  == Börgers, Talei Franzesi, LeBeau, Boyden & Kopell (2012) — Minimal Size of Cell Assemblies Coordinated by Gamma Oscillations

  #link("https://doi.org/10.1371/journal.pcbi.1002362")[doi.org/10.1371/journal.pcbi.1002362]

  *Summary.* A combined computational and mathematical study of how gamma-frequency (25–100 Hz) rhythms in networks of excitatory (E) and inhibitory (I) cells break down as the driven cell ensemble shrinks or as synaptic interactions weaken. The authors show that in homogeneous networks rhythms can slide gradually, but in realistically heterogeneous networks a PING rhythm either establishes rapidly or not at all, implying a soft lower bound on the number of cells that can oscillate together at gamma frequency. They connect this to experimental findings on gamma modulation by stimulus size and attention, and to their own optogenetic hippocampal-slice results.

  *Claim it supports (§4.3).* _"Theoretical extensions include multi-layer PING architectures, an independent two-dimensional sweep of $alpha_(E I) times alpha_(I E)$, multi-rhythm and theta-nested gamma models [19, 20], and characterisation of capacity limits for single cycles and minimal assemblies [43]."_

  *Quotes.*

  - _"This leads to the surprising conclusion that in a network with realistic heterogeneity, gamma rhythms based on the interaction of excitatory and inhibitory cell populations must arise either rapidly, or not at all."_ (p. 1 (Abstract)) — Central finding: heterogeneous PING gamma is all-or-nothing per assembly, bearing on minimal-assembly capacity.
  - _"For given synaptic strengths and heterogeneities, there is a (soft) lower bound on the possible number of cells in an ensemble oscillating at gamma frequency, based simply on the requirement that synaptic interactions between the two cell populations be strong enough."_ (p. 1 (Abstract)) — States the soft lower bound / minimal assembly size the citing sentence points to for capacity limits.


  #divider


  == Cramer, Stradmann, Schemmel & Zenke (2022) — The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks

  #link("https://doi.org/10.1109/TNNLS.2020.3044364")[doi.org/10.1109/TNNLS.2020.3044364]

  *Summary.* This paper introduces two open, audio-derived spike-based classification datasets, the Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC), as benchmarks for evaluating spiking neural networks. A biologically inspired conversion pipeline (a hydrodynamic basilar-membrane model plus hair-cell and bushy-cell models) turns spoken-digit and speech-command audio into spike trains; the datasets and software are released in HDF5. Baselines include conventional classifiers (SVMs, LSTMs, CNNs) and spiking networks trained with surrogate gradients and BPTT. Centrally, spike-count patterns stripped of temporal information cannot achieve high accuracy, so exploiting spike timing is essential; temporally aware classifiers perform substantially better.


  *Claim it supports (§4.2).* _"Empirical extensions include evaluation on the SHD dataset [44], which tests the gating on a spiking benchmark with intrinsic temporal structure, and tasks in which classification accuracy depends on input timing, which test the role of the gamma cycle as a temporal unit."_

  *Quotes.*

  - _"Temporal information is essential to classify the SHD and the SSC datasets with high accuracy."_ (p. 6, Figure 3 caption) — Establishes that the SHD benchmark requires temporal spike-timing structure, supporting its use as a spiking benchmark with intrinsic temporal structure.
  - _"Moreover, solving these problems with high accuracy requires taking into account spike timing."_ (p. 3, Section 2) — States that the SHD and SSC datasets depend on spike timing for accurate classification.
  - _"we found that the temporal information available in the spike times can be leveraged for better classification by suitable classifiers."_ (p. 10, Section 4) — Summarizes the finding that spike timing enables good classification on these spiking benchmarks.


  #divider


  == Tiesinga & Sejnowski (2009) — Cortical Enlightenment: Are Attentional Gamma Oscillations Driven by ING or PING?

  #link("https://doi.org/10.1016/j.neuron.2009.09.009")[doi.org/10.1016/j.neuron.2009.09.009]

  *Summary.* Tiesinga and Sejnowski (Neuron, 2009) review whether attentional gamma oscillations (30–80 Hz) arise from the interneuron-gamma (ING) or pyramidal-interneuron-gamma (PING) mechanism, linking modeling to optogenetic experiments manipulating parvalbumin fast-spiking interneurons. ING arises from mutually connected inhibitory cells alone; PING from reciprocally connected excitatory and inhibitory networks. They lay out distinct predictions for how stimulating fast-spiking versus regular-spiking neurons affects LFP, firing rate, synchrony, and phase, arguing the evidence supports PING in cortex (with ING possibly contributing in some states). They propose aperiodic optogenetic stimulation is needed to definitively distinguish the mechanisms.

  *Claim it supports (§5.2.2).* _"The restriction to the E-I loop is intended to make the rhythm unambiguously PING: excluding $W_(i i)$ rules out ING, and excluding $W_(e e)$ rules out recurrent-E driven oscillation [7, 45]."_

  *Quotes.*

  - _"second, by activation of inhibitory networks via the interneuron gamma (ING) mechanism (Figure 1C); and third, by activation of reciprocally connected networks of excitatory and inhibitory neurons via the pyramidal-interneuron gamma (PING) mechanism"_ (p. 3, Local Synchrony with the ING or PING Mechanisms) — Defines PING as arising from reciprocally connected E and I networks (the E-I loop) and ING as arising from inhibitory networks, supporting that PING requires the E-I loop while ING does not.
  - _"In the ING mechanism the E cells are followers, which can be modeled by cutting out the E to I projection. In this circuit, activation of E cells should have no effect on the gamma oscillation."_ (p. 6, New Experiments to Distinguish ING and PING) — Confirms ING does not depend on E-to-I drive (E cells are followers), so the E-I loop distinguishes PING from ING.


  #divider


  == Brunel (2000) — Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons

  #link("https://doi.org/10.1023/A:1008925309027")[doi.org/10.1023/A:1008925309027]

  *Summary.* Brunel analytically studies sparsely connected networks of excitatory and inhibitory leaky integrate-and-fire neurons. Using a mean-field/Fokker-Planck approach approximating synaptic input as a Gaussian process, he derives self-consistent equations for stationary firing rates and interspike-interval variability, then analyzes stability of asynchronous states. Four regimes emerge depending on excitation-inhibition balance, external input, and synaptic delays: synchronous regular, asynchronous regular, asynchronous irregular, and synchronous irregular, the last with fast or slow global oscillations whose frequencies are set by synaptic delay and membrane time constant. Phase diagrams are validated against simulations of 10,000 excitatory and 2,500 inhibitory neurons.

  *Claim it supports (§5.3.5).* _"The population gain functions $Phi_E$ and $Phi_I$ are the noise-driven LIF rate functions in Ricciardi–Siegert form [46]."_

  *Quotes.*

  - _"Note that the analytical expression for the mean first passage time of an IF neuron with random gaussian inputs (Ricciardi, 1977; Amit and Tsodyks, 1991) is recovered, as it should"_ (p. 188, Section 4.1) — Explicitly identifies the derived firing-rate expression as the Ricciardi mean first passage time / rate function for a noise-driven IF neuron, backing the Ricciardi-Siegert form claim.
  - _"the synaptic current of a neuron can be approximated by an average part plus a fluctuating gaussian part"_ (p. 185, Section 3) — Establishes that the rate function is noise-driven: the LIF neuron's input is a mean plus Gaussian noise, the setting for the Ricciardi-Siegert rate.


  #divider


  == Gerstner (2000) — Population Dynamics of Spiking Neurons: Fast Transients, Asynchronous States, and Locking

  #link("https://doi.org/10.1162/089976600300015899")[doi.org/10.1162/089976600300015899]

  *Summary.* Wulfram Gerstner's Population Dynamics of Spiking Neurons (Neural Computation, 2000), deriving and analyzing an integral equation for the population activity A(t) of a large, homogeneous, fully-connected pool of spike-response / integrate-and-fire neurons in the N-to-infinity limit. The central equation is exact in that limit and generalizes the classical integral equations of Wilson-Cowan (1972) and Knight (1972).

  *Claim it supports (§5.3.10).* _"The treatment follows the Wilson–Cowan tradition for cortical-rhythm modelling [18] and the broader population-dynamics and next-generation neural-mass literature [19, 47, 48]."_

  *Quotes.*

  - _"An integral equation describing the time evolution of the population activity in a homogeneous pool of spiking neurons of the integrate-and-fire type is discussed."_ (p. 43 (abstract)) — Opening sentence of the abstract establishing the paper as a population-dynamics (population-activity) work.
  - _"The relevant equation is a generalization of the integral equations of Wilson and Cowan (1972) and Knight (1972a) and has been discussed previously in Gerstner and van Hemmen (1994) and Gerstner (1995)."_ (p. 45 (Introduction)) — States the population equation is a generalization of the Wilson-Cowan integral equations, situating it in the Wilson-Cowan population-dynamics tradition.


  #divider


  == Montbrió, Pazó & Roxin (2015) — Macroscopic Description for Networks of Spiking Neurons

  #link("https://doi.org/10.1103/PhysRevX.5.021028")[doi.org/10.1103/PhysRevX.5.021028]

  *Summary.* Montbrio, Pazo and Roxin (Phys. Rev. X, 2015) derive exact macroscopic firing-rate equations for a network of heterogeneous, all-to-all coupled quadratic integrate-and-fire neurons in the thermodynamic limit. Using a Lorentzian ansatz for membrane-potential distributions, they reduce the infinite-dimensional dynamics to two ordinary differential equations for population firing rate and mean membrane potential. Unlike heuristic neural-mass models, these equations are exact and capture spike synchronization, including damped oscillations, bistability and macroscopic chaos under time-varying drive. A conformal map relates this to the Ott-Antonsen/Kuramoto description of coupled phase oscillators.

  *Claim it supports (§5.3.10).* _"The treatment follows the Wilson–Cowan tradition for cortical-rhythm modelling [18] and the broader population-dynamics and next-generation neural-mass literature [19, 47, 48]."_

  *Quotes.*

  - _"Here we provide the derivation of a set of exact macroscopic equations for a network of spiking neurons."_ (p. 1 (Abstract)) — States the paper's central contribution — exact macroscopic (population/neural-mass-type) equations for a spiking network, placing it in the population-dynamics literature.
  - _"These descriptions, called firing-rate equations (FREs), have been proven to be extremely useful in understanding general computational principles underlying functions such as memory [23, 24], visual processing [25–27], motor control [28] or decision making [29]."_ (p. 1 (Introduction)) — Frames the work within the firing-rate / population-dynamics modelling tradition it extends.
  - _"We have presented a method for deriving firing rate equations for a network of heterogeneous QIF neurons, which is exact in the thermodynamic limit."_ (p. 6 (Conclusions)) — Confirms the paper produces exact firing-rate (neural-mass-type) equations, i.e. next-generation neural-mass modelling.


  #divider


  == Zenke & Ganguli (2018) — SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks

  #link("https://doi.org/10.1162/neco_a_01086")[doi.org/10.1162/neco_a_01086]

  *Summary.* SuperSpike is a supervised learning rule for training multilayer networks of deterministic leaky integrate-and-fire (LIF) neurons to perform nonlinear transformations on spatiotemporal spike patterns. The authors derive a nonlinear voltage-based three-factor rule by replacing the non-differentiable spike train derivative with a surrogate gradient: a continuous auxiliary function (the negative side of a fast sigmoid) of the membrane potential. The rule combines a filtered presynaptic trace, a surrogate postsynaptic factor, and a per-neuron error signal, evaluated online via an eligibility trace.

  *Claim it supports (§5.4.4).* _"with slope $s = 1$ [23, 49]."_

  *Quotes.*

  - _"we use the partial derivative of the negative half of a fast sigmoid f(x) = x/(1+|x|)"_ (section 3.3.2, p. 1522) — SuperSpike defines its surrogate gradient as the derivative of a fast sigmoid of the membrane voltage.
  - _"we compute σ'(Ui) = (1 + |hi|)^-2 with hi ≡ β (Ui − ϑ), where ϑ is the neuronal firing threshold and β = (1 mV)^-1 unless mentioned otherwise"_ (section 3.3.2, p. 1522) — The surrogate derivative uses a fast sigmoid with slope parameter β set to unity (per mV).


  #divider


  == Kingma & Ba (2015) — Adam: A Method for Stochastic Optimization

  #link("https://doi.org/10.48550/arXiv.1412.6980")[doi.org/10.48550/arXiv.1412.6980]

  *Summary.* Kingma and Ba (ICLR 2015) introduce Adam, a first-order gradient-based optimizer for stochastic objectives using adaptive estimates of lower-order moments. It computes per-parameter adaptive learning rates from first- and second-moment estimates of gradients, combining AdaGrad's handling of sparse gradients with RMSProp's suitability for non-stationary settings. Adam is simple, computationally efficient, low in memory, and invariant to diagonal gradient rescaling, suiting large or noisy problems. The paper provides initialization bias-correction analysis, an O(√T) regret bound under online convex optimization, and empirical results on logistic regression and neural networks (MNIST, IMDB, CIFAR-10). It also introduces AdaMax, an infinity-norm variant.

  *Claim it supports (§5.4.4).* _"Optimisation uses Adam [50] with learning rate $4 times 10^(-4)$, batch size $256$, and 100 epochs."_

  *Quotes.*

  - _"We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments."_ (p. 1, Abstract) — Defines Adam as the first-order gradient-based stochastic optimization method the manuscript uses as its optimizer.
  - _"We propose Adam, a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement."_ (p. 1, Introduction) — Confirms Adam is the named optimization method being cited for training.
  - _"Good default settings for the tested machine learning problems are α = 0.001, β1 = 0.9, β2 = 0.999 and ϵ = 10−8."_ (p. 2, Algorithm 1) — Establishes that Adam is configured by a learning rate (stepsize α) hyperparameter, which the manuscript sets to 4e-4.


  #divider


  == Pascanu, Mikolov & Bengio (2013) — On the Difficulty of Training Recurrent Neural Networks

  #link("https://doi.org/10.48550/arXiv.1211.5063")[doi.org/10.48550/arXiv.1211.5063]

  *Summary.* Pascanu, Mikolov and Bengio analyze why recurrent neural networks are hard to train, focusing on vanishing- and exploding-gradient problems. Using backpropagation through time, they rewrite the gradient in a sum-of-products form where each temporal component contains a product of t−k Jacobian matrices, which can shrink to zero or explode. They derive that a largest eigenvalue of the recurrent weight matrix below 1/gamma suffices for vanishing gradients and above it is necessary for exploding gradients, connecting this to attractors and error-surface walls. Their remedy is gradient-norm clipping for exploding gradients plus a soft regularization constraint for vanishing gradients, validated empirically.

  *Claim it supports (§5.4.5).* _"The mechanism is the same multiplicative compounding through long unrolled recurrences that characterises the exploding-gradient pathology in standard recurrent networks [51]."_

  *Quotes.*

  - _"To understand this phenomenon we need to look at the form of each temporal component, and in particular at the matrix factors"_ (p. 2, Section 2.1 The mechanics) — Introduces that each gradient's temporal component takes the form of a product of Jacobian matrices.


  #divider


  == Cornford, Kalajdzievski, Leite, Lamarquette, Kullmann & Richards (2021) — Learning to Live with Dale's Principle: ANNs with Separate Excitatory and Inhibitory Units

  #link("https://doi.org/10.1101/2020.11.02.364968")[doi.org/10.1101/2020.11.02.364968]

  *Summary.* A paper introducing Dale's ANNs (DANNs), networks respecting Dale's principle via separate excitatory and inhibitory populations with strictly non-negative weights, where inhibitory interneurons provide subtractive and divisive inhibition. Dale's principle is usually omitted because it impairs learning, yet DANNs learn as well as standard MLPs and far better than sign-constrained columns (ColumnEi). Two insights enable this: the architecture resembles normalization schemes, motivating a weight initialization balancing excitation and inhibition; and inhibitory updates are scaled via Fisher-Information corrections so both affect the output comparably.

  *Claim it supports (§5.4.8).* _"After each optimiser step, the trainable conductance magnitudes are projected onto the non-negative cone [52, 53]; pathway identity and reversal potential determine whether their effect is excitatory or inhibitory."_

  *Quotes.*

  - _"in order to maintain the positive DANN weight constraint, if after a parameter update a weight was negative, we reset it to zero, i.e."_ (p. 8) — Describes the post-update weight-clipping (projection onto the non-negative sign cone) that the manuscript's Dale's-law clamp cites.
  - _"All of the synaptic weights are strictly non-negative, and inhibition is enforced via the activation rules for the units"_ (p. 3, Section 2.1 (Model definition), constraint 4) — Establishes that the DANN architecture enforces strictly non-negative weights, i.e. sign-constrained weights, the property the clamp maintains.


  #divider


  == Zhu et al. (2026) — Task Success in Trained Spiking Neural Network Models Coincides with Emergence of Cross-Stimulus-Modulated Inhibition

  #link("https://doi.org/10.1007/s00422-025-01030-4")[doi.org/10.1007/s00422-025-01030-4]

  *Summary.* This paper trains recurrent spiking neural networks (SNNs) constrained by mouse-neocortex connectivity to perform a binary motion-entropy change-detection task. It reports that task success coincides with the emergence of push-pull, cross-stimulus-modulated inhibition: excitatory units strengthen connections to like-modulated units while inhibitory units strengthen connections to oppositely-modulated units. The authors find this inhibitory motif fails to emerge when Dale's law is not enforced, and that jittering spike times by a few milliseconds impairs performance, underscoring the roles of structured inhibition and precise spike-time coordination in sparse cortical-like networks.

  *Claim it supports (§5.4.8).* _"The trained network preserves excitatory and inhibitory pathway identity throughout optimisation [52, 53]. In the present conductance formulation, every stored magnitude remains non-negative and the pathway reversal potential supplies the physiological sign."_

  *Quotes.*

  - _"All synaptic connections originating from e units have positive weight values, and all from i units have negative weight values. Positivity and negativity of each connection is maintained during training, although values may change"_ (p. 16, Sect. 4.1) — Model construction: enforcing Dale's law by fixing the sign of E and I connections throughout training.
  - _"Positivity and negativity of all connections, i.e. the excitatory or inhibitory identity of each neuron, was maintained throughout training consistent with Dale's law"_ (p. 3, Sect. 2.1) — Results: describing that the sign of each connection is held fixed during training per Dale's law.


  #divider


  == Welch (1967) — The Use of Fast Fourier Transform for the Estimation of Power Spectra: A Method Based on Time Averaging Over Short, Modified Periodograms

  #link("https://doi.org/10.1109/TAU.1967.1161901")[doi.org/10.1109/TAU.1967.1161901]

  *Summary.* _Not grounded — a signal-processing methods citation (Welch's method for power-spectral estimation via averaged, windowed periodograms). It is cited as the tool used to compute spectra, not for a scientific claim to anchor, so no source quote is needed._


  #divider


  == Atallah & Scanziani (2009) — Instantaneous Modulation of Gamma Oscillation Frequency by Balancing Excitation with Inhibition

  #link("https://doi.org/10.1016/j.neuron.2009.04.027")[doi.org/10.1016/j.neuron.2009.04.027]

  *Summary.* Atallah and Scanziani (Neuron 2009) report that gamma oscillations in rat hippocampal CA3 vary rapidly in amplitude and frequency cycle to cycle, with one cycle's amplitude predicting the interval to the next. Using in vivo and in vitro whole-cell recordings, they show amplitude fluctuations reflect synaptic excitation spanning over an order of magnitude, immediately and proportionally counterbalanced by inhibition. Larger inhibitory conductances produce longer hyperpolarization of pyramidal cells, delaying the next cycle, so rapid inhibitory adjustments instantaneously modulate frequency.

  *Claim it supports (§5.5.3).* _"$R$ is qualitatively consistent with the spectral-peak and population-coherence measures used elsewhere in the gamma literature [55, 56]."_

  *Quotes.*

  - _"A distinct spectral peak in LFP activity occurred at frequencies between 25 Hz and 40 Hz (mean frequency = 29.8 Hz; SD = 2.4 Hz) as observed in vivo."_ (p. 568) — A spectral-peak measure identifying the gamma band of the in vitro LFP.
  - _"Both excitatory and inhibitory synaptic currents occurred at gamma frequencies, as shown by their power spectra, and exhibited a pronounced peak in coherence with the simultaneously recorded LFP within the gamma frequency band (Figure S4)."_ (p. 568) — Population-coherence measure relating synaptic currents to the LFP within the gamma band.


  #divider


  == Xing, Shen, Burns, Yeh, Shapley & Li (2012) — Stochastic Generation of Gamma-Band Activity in Primary Visual Cortex of Awake and Anesthetized Monkeys

  #link("https://doi.org/10.1523/JNEUROSCI.5644-11.2012")[doi.org/10.1523/JNEUROSCI.5644-11.2012]

  *Summary.* This study by Xing et al. compares the dynamics of gamma-band activity (25-90 Hz) in local field potentials recorded from primary visual cortex (V1) of awake and anesthetized monkeys during visual stimulation with drifting gratings. Using time-frequency (continuous Gabor transform) analysis, the authors find that gamma activity in both brain states shares identical temporal characteristics: large variability of peak frequency, short oscillatory epochs (under 100 ms on average), and stochastic incidence and duration of oscillatory events.

  *Claim it supports (§5.5.3).* _"$R$ is qualitatively consistent with the spectral-peak and population-coherence measures used elsewhere in the gamma literature [55, 56]."_

  *Quotes.*

  - _"To quantify the peak frequency and duration of a gamma-band event in the spectrogram, we first searched for a point in the time-frequency coordinate"_ (p. 13874) — Time-frequency (spectrogram/Gabor) analysis used to identify gamma bursts, related to coherence-based measures.


  #divider


  == Rotter & Diesmann (1999) — Exact Digital Simulation of Time-Invariant Linear Systems with Applications to Neuronal Modeling

  #link("https://doi.org/10.1007/s004220050570")[doi.org/10.1007/s004220050570]

  *Summary.* Rotter and Diesmann (Biol. Cybern., 1999) present a method for exact digital simulation of time-invariant linear systems that model neuronal systems or their sub-modules. Their Exact Integration uses the matrix exponential to construct an iteration y\_{k+1} = e^{AΔ} y\_k + x\_k that propagates the exact state on a regular time grid, limited only by floating-point precision. Dynamic inputs including δ-pulse trains are incorporated exactly via auxiliary equations. They compare against approximate integrators (Euler, Adams-Bashforth, Runge-Kutta, Crank-Nicholson). For integrate-and-fire neurons with linear subthreshold integration, it yields the smallest error in subthreshold dynamics and spike timing, especially at moderate step sizes.

  *Claim it supports (§5.6.1).* _"The membrane and synaptic-conductance ODEs are integrated by an exponential-Euler scheme [57] with zero-order hold on the synaptic conductances over each step."_

  *Quotes.*

  - _"The iteration of this system according to (5) has been termed “exponential integration” (MacGregor 1987) and is widely used in the context of neuronal model simulations."_ (p. 387, Sect. 3.2.3 Piecewise-constant input) — Names the scheme the manuscript calls exponential-Euler ('exponential integration') and notes its widespread use in neuronal simulation, the basis for the citation.
  - _"The function ψ solving the equation is constant between any two successive pulses."_ (p. 387, Sect. 3.2.3 Piecewise-constant input) — Establishes the zero-order-hold property: the synaptic input is held piecewise-constant between pulses over each step, matching the claim's 'zero-order hold on the synaptic conductances over each step.'


  #divider


  == LeCun, Bottou, Bengio & Haffner (1998) — Gradient-Based Learning Applied to Document Recognition

  #link("https://doi.org/10.1109/5.726791")[doi.org/10.1109/5.726791]

  *Summary.* A paper reviewing gradient-based learning (chiefly multilayer networks trained by back-propagation) applied to document recognition, using handwritten character recognition as its central case. It argues hand-crafted feature extraction can be replaced by learning machines operating directly on pixels, and introduces convolutional networks (LeNet-5) exploiting 2-D shape invariances via local receptive fields, shared weights, and spatial subsampling. The authors construct the MNIST database of size-normalized digits and report convolutional networks outperform all other tested methods. It also develops graph transformer networks for end-to-end training of multi-module systems (segmentation, recognition, language modeling), plus deployed systems for online handwriting and bank-check reading.

  *Claim it supports (§5.7).* _"The classification task is MNIST [58], rate-encoded as in §5.4 (200 ms presentation, 25 Hz peak Poisson rate per active pixel)."_

  *Quotes.*

  - _"The resulting database was called the modified NIST, or MNIST, dataset."_ (p. 10, Section III.A (Database: The Modified NIST Set)) — Establishes that this paper is the source that constructed and named the MNIST handwritten-digit dataset.
  - _"While recognizing individual digits is only one of many problems involved in designing a practical recognition system, it is an excellent benchmark for comparing shape recognition methods."_ (p. 10, Section III (Results and Comparison with Other Methods)) — Frames MNIST digit recognition as a classification benchmark, matching the manuscript's use of it as a classification task.

]
