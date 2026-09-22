# Learning PING through continuous competitive selection

Research and proposed protocol, 20 September 2026. This is a design document; numerical settings below are proposed starting points, not executed results or literature-established optima. No experiment identity is allocated here.

1. **Scientific assessment and scope**

   The proposal is worth testing. Its strongest version asks whether a conductance-based spiking network can acquire useful pyramidal–interneuron network gamma (PING) feedback from an initially absent inhibitory return pathway, under pressure to select changing signals with limited activity and limited response time.

   Competition alone does not logically require an oscillation. A feedforward classifier can estimate each pattern's strength and let its decoder select the largest score. Feedback may instead provide nonoscillatory normalization, and a spike penalty may suppress interneurons altogether. These are substantive alternative results.

   Use four separately evaluated hypotheses:

   1. **Feedback acquisition:** training makes the E→I→E loop functionally important. A positive weight alone is insufficient.
   2. **PING acquisition:** the learned loop produces recurrent E recruitment of I, inhibitory suppression of E, and recovery toward another E volley, with rhythmic population activity under nonrhythmic drive.
   3. **Computational benefit:** the trained circuit improves the accuracy–latency–total-spike trade-off over competitive baselines.
   4. **Timing contribution:** perturbing temporal organization damages that advantage beyond what can be explained by changes in mean activity or inhibitory conductance. Establishing this requires several imperfect but complementary interventions.

   To claim specifically that **competition supplies the learning pressure**, also compare training objectives and input conditions. Merely finding PING after training on a competitive task cannot establish which aspect of the task caused its emergence.

   This is initially a model of bottom-up salience selection. The strongest stimulus is designated relevant by definition. Selecting a weaker but cued target would be a separate extension concerning selective attention.

2. **What the primary literature establishes**

   | Study | Relevant evidence | Boundary for this proposal |
   | --- | --- | --- |
   | [de Almeida, Idiart & Lisman, 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2758634/) | Feedback inhibition can select cells whose suprathreshold excitation lies sufficiently close to the maximum. Their E%-max approximation connects selectivity to feedback delay and membrane dynamics. | This is a mechanism in a specified circuit, not evidence that supervised task learning discovers the circuit. Multiple near-maximal cells may fire. |
   | [Börgers, Epstein & Kopell, 2008](https://math.bu.edu/people/nk/papers/borgers_pnas_08.pdf) | A biophysical cortical model links stimulus competition and attentional modulation to inhibitory coherence and oscillatory selection. | Its mechanism is richer than “more gamma means better selection”: strong inhibitory drive can disrupt coherence; attention-related modulation can restore it. It is not a task-training study. |
   | [Chalk, Gutkin & Denève, 2016](https://elifesciences.org/articles/13824) | Efficient spiking representations with synaptic delays exhibit a favorable regime of population oscillations, sparse participation, and tight E/I coordination. | Their efficient-coding construction does not establish learning from zero return coupling. Excessive synchrony impairs coding; gamma power is not itself an efficiency score. |
   | [Echeveste et al., 2020](https://www.nature.com/articles/s41593-020-0671-1) | Optimizing recurrent E/I circuits for fast sampling-based inference produces gamma oscillations and other cortical dynamics. The authors compare alternative optimization objectives. | Their circuit is a stochastic rate network, not the proposed conductance-based spiking network with an initially absent I→E pathway. Broad claims that computation has never produced learned gamma would be incorrect. |
   | [Bittar & Garner, 2024](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1449181/full) | End-to-end training of a speech-recognition SNN produces oscillatory organization and cross-frequency coupling. | Speech contains structured temporal scales; the study does not establish this proposal's zero-feedback start or isolate PING timing under activity-controlled competitive selection. |
   | [Kato & Ikeguchi, 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0146044) | STDP, conduction delays, and spontaneous oscillation cooperate to organize neural competition. | Plasticity is applied to E→E connections, with existing inhibition; this is different from learning the inhibitory feedback loop through classification and spike cost. |
   | [Nejat, Sherfey & Bastos, preprint, revised 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722284.2/) | Optimization of biophysical circuits reproduces observed spectral dynamics and develops weak-PING-like organization. | The available version is a preprint, and its objectives include matching spectral responses. This differs directly from learning a computational task without a spectral target. |
   | [Siegle, Pritchett & Moore, 2014](https://www.nature.com/articles/nn.3797) | Manipulating interneuron synchronization changes tactile detection, with benefits depending on stimulus strength and temporal alignment. | This motivates timing interventions and condition-specific predictions, not a universal gamma advantage or evidence of its developmental learning. |

   The defensible contribution is therefore the conjunction of **a spiking conductance model, an initially absent return pathway, an aperiodic continuous task, explicit total activity cost, and causal timing analysis**. The search located substantial adjacent precedent but did not verify an exact prior demonstration of this conjunction. That is not proof of priority.

3. **Mechanistic prediction**

   The proposed computation is repeated selective admission. More strongly driven E cells reach threshold first; they recruit I cells; inhibition curtails later activity among weaker competitors; inhibitory decay allows a subsequent opportunity to respond. This could reduce distractor spikes while repeatedly admitting evidence about the current winner.

   Writing the original paper's E%-max tolerance as a fraction \(\varepsilon\), an adapted notation here, its approximate relation is

   \[
   \varepsilon \approx d/\tau_m,
   \]

   where \(\varepsilon\) is the allowed fractional shortfall from the maximum **suprathreshold excitation**, \(d\) is feedback-inhibition delay, and \(\tau_m\) is the membrane time constant in that analysis. The displayed percentage would be \(100\varepsilon\). This is an established approximation in that model, not a general identity for conductance-based networks; in particular, \(\tau_m\) should not silently be replaced by a GABA decay constant. [Original analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC2758634/).

   Distinguish the within-cycle selection window from the recurrence period. Delayed inhibition can create a useful single selection event even without sustained gamma. Conversely, regular gamma can coexist with weak selectivity. A cycle-by-cycle relation between relative drive, firing order, and suppression is more informative than a spectral peak alone.

   The hypothesis predicts a favorable intermediate regime: enough coordination to suppress redundant activity, without recruiting the entire E population every cycle. It also predicts possible costs: an input arriving during strong inhibition may wait, and rapid changes may outpace evidence accumulation. These predictions should be evaluated, not removed by choosing favorable stimulus phases.

4. **A generative task with controllable difficulty**

   Start with a fixed dictionary of eight nonnegative spatial templates on a 16×16 input lattice. This size is a pilot choice. Equalize template sums and construct several known overlap levels. In each interval, activate four, six, or eight templates. Spatial overlap must refer to shared input channels; giving the network separate labeled channels for each component would trivialize demixing.

   A convenient proposed generator is

   \[
   \lambda_i(t)=\lambda_b+A(t)\sum_{k=1}^{K}p_k(t)u_{ki},
   \qquad \sum_k p_k(t)=1,
   \qquad N^{-1}\sum_i u_{ki}=1.
   \]

   Here \(t\) is time, \(i\) indexes the \(N=256\) input channels, \(k\) indexes the \(K=8\) template identities, \(\lambda_i\) is the channel's rate in spikes per second, \(\lambda_b\) is a background rate, \(A\) is a common signal-intensity scale in spikes per second, \(p_k\) is a dimensionless nonnegative mixture proportion, and \(u_{ki}\) is dimensionless template intensity. Inactive templates have \(p_k=0\). This normalization keeps total expected input rate independent of which templates are active when \(A\) is fixed.

   Draw conditionally independent Poisson processes for the input channels given the rate trajectories. Do not copy a common spike train across the pixels of a template. Conditional independence does not imply unconditional independence when rates co-vary with the stimulus.

   Define the target as \(y(t)=\arg\max_k p_k(t)\), where \(y\) is the strongest component's identity. Exclude exact ties from the primary metric and specify their handling separately. Control the relative top-two margin

   \[
   m(t)=\frac{p_{(1)}(t)-p_{(2)}(t)}{p_{(1)}(t)},
   \]

   where \(p_{(1)}\) and \(p_{(2)}\) are the largest and second-largest proportions. Proposed diagnostic bins are 5%, 10%, 20%, and 40%; tune task feasibility before freezing these values. Near ties may be statistically unresolved within a short window even for an excellent decoder.

   Keep the initial 150–400 ms interval range, sampled randomly. Mix winner changes, changes among losers, and common-intensity changes that leave the winner unchanged. Do not tell the network event times or reset its state at them. Include some intervals where only one or two component strengths change; otherwise every event supplies an artificial common reset.

   Use independent gain changes, initially over a fourfold range such as 0.5×–2×, with transition times independent of winner changes. Choose absolute input rates using an observer-based pilot; arbitrary rates can make the task either trivial or informationally impossible. Hold out new mixture trajectories and Poisson realizations. A new template dictionary requires retraining or an explicit transfer protocol because the output labels otherwise change meaning.

   Include an equal-rate single-pattern condition, held-out smooth transitions, and a broader duration test such as 75–800 ms. Uniformly bounded dwell times are aperiodic but still define a learnable hazard; use an additional renewal distribution or smoothly varying coefficients to assess dependence on that temporal prior. Audit the input spectra.

5. **Information and readout controls**

   Before interpreting an architectural failure, fit a causal idealized observer to the generator. It may know the templates and distribution of rates and change times, but must observe only the same past input spikes—not the hidden coefficients, actual change times, or future samples. A validated approximate Poisson filter is acceptable if it is clearly labeled approximate. Report its accuracy versus available observation time. A simple template-score or logistic decoder supplies a second, inexpensive difficulty check.

   The main network must report through real output spikes. Start with eight output groups, for example four neurons per identity. Use counts from a trailing 25 ms window as class evidence. Silence is an abstention, counted as unsuccessful for primary accuracy; tied counts have a fixed rule. The decoder must not retain an old winner indefinitely after its evidence expires.

   Test 10, 25, 50, and 100 ms windows rather than choosing whichever favors PING. The readout timescale is a substantive task constraint and a possible source of an apparent preferred frequency. Use a fixed smoothing floor to turn counts into probabilities for training, and avoid an unbounded trainable gain or an analog voltage path that could conceal useful computation while output spikes approach zero.

   External argmax is allowed for reporting accuracy, but it does not demonstrate competition inside E. Independently assign E-cell preferences using isolated-template probes on validation inputs, then measure how adding a competitor changes firing in those groups. With distributed representations, also use a decoder trained on E activity; avoid assigning preference from the final test conditions.

6. **Network and initial condition**

   A manageable pilot is 256 input channels, 256 E neurons, 64 I neurons, and 32 output neurons. Use the same conductance-based LIF equations across conditions. Fix E→E and I→I to zero initially; this isolates the two-population loop and excludes an autonomous interneuron recurrent network as the explanation. A later robustness study can relax those restrictions.

   Initialize E→I to produce modest, heterogeneous I activity across the intended drive range. Set I→E exactly to zero and train both matrices from the first update. Do not supply direct oscillatory input, a gamma loss, a gamma teacher, or a minimum gamma amplitude. The E and output forward dynamics start with no inhibitory return, although the I satellite is already active and its spikes must be counted.

   Separate two claims: acquiring **feedback function** and acquiring **all its anatomy**. The proposed architecture already specifies where E and I can connect and fixes fast inhibitory kinetics. Success would establish learned use of an allowed PING-capable circuit, not learning the entire mechanism without prior structure.

   Exact-zero initialization is compatible with projected nonnegative optimization. Avoid using a zero multiplicative gate, squared parameter, or other parameterization whose derivative makes the return pathway permanently inaccessible. An ordinary softplus parameterization cannot represent exact zero at finite parameters.

   Initially the classification gradient reaching E→I through the return route is multiplied by the zero I→E matrix. I→E can nevertheless receive a task gradient because I already spikes. This is a bootstrapping opportunity, not a guarantee. The I-spike penalty can suppress the satellite before useful feedback grows. Log task and activity gradients separately, I activity, projected updates, and the first departure from zero.

   Keep delays and synaptic constants fixed in the primary acquisition test. Proposed starts are a 0.1 ms integration step, 2 ms excitatory decay, and 6 ms inhibitory decay, with explicitly specified synaptic transmission timing. These are modeling choices, not a guaranteed gamma operating point. Measure actual E-to-I-to-E latency; a simulator's update order can otherwise make an effective delay depend on the integration step. Later vary inhibitory kinetics, heterogeneity, and transmission delays to test whether the result depends on one engineered timescale.

7. **Learning objective and optimization safeguards**

   Use the proposed objective

   \[
   \mathcal L=\frac{1}{M}\sum_{j=1}^{M}\operatorname{CE}(q_j,y_j)
   +\lambda_s\frac{S_E+S_I+S_O}{T r_{\mathrm{ref}}}.
   \]

   Here \(\mathcal L\) is the dimensionless training loss; \(M\) is the number of scored times; \(j\) indexes those times; \(q_j\) is the vector of probabilities derived from output spike counts; \(y_j\) is the target identity; CE is categorical cross-entropy; \(S_E,S_I,S_O\) are total emitted spikes in the respective populations over the full scored stream; \(T\) is its duration in seconds; \(r_{\mathrm{ref}}\) is one fixed reference total spike rate used only for normalization; and \(\lambda_s\) is a dimensionless trade-off weight. Keep \(r_{\mathrm{ref}}\) identical across models.

   Summing population mean rates without multiplying by their population sizes would not penalize total spikes. Charge activity during errors and silence-related failures as well as successes. The imposed input spikes are identical paired sensory evidence; report them separately if claiming a system-level cost. Spike counts are an activity proxy, not measured metabolic or hardware energy; synaptic fan-out and computation also cost resources.

   Continuous error penalizes slow switching without a separate reward for oscillation. Begin with uniformly weighted scored times; if transition errors are underweighted, introduce a declared transition-weighted objective as a distinct treatment. Do not quietly mask every post-change interval.

   Compare zero penalty with a predeclared penalty grid. A short common penalty ramp is a reasonable pilot remedy for early silence, but include immediate-penalty training as a robustness condition and report the ramp. Avoid requiring a sustained I-rate floor in the primary experiment: it would impose continued interneuron use.

   Use surrogate-gradient training with the same reset-gradient convention, voltage-gradient damping, clipping, and search budget across relevant models. The original [SuperSpike study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/) supports supervised learning through spiking dynamics; it does not guarantee successful recurrent PING acquisition. Keep state continuous through training chunks. If truncating backpropagation, retain state and detach gradients only; vary chunk boundaries independently of input changes and check whether conclusions survive a longer credit-assignment horizon.

   Choose checkpoints by held-out task/activity criteria, never gamma strength. Record true epoch-zero diagnostics and fixed checkpoints. If only PING-initialized models learn useful PING, the result concerns accessibility from initialization, not acquisition from the COBA start.

8. **Comparisons and pressure controls**

   | Condition | Training and purpose |
   | --- | --- |
   | Feedforward COBA | Same E and output model, with no functional I return; train input/readout weights. The practical baseline need not spend spikes on a useless I satellite. |
   | Learned loop from zero return | Main proposed condition: active E→I, zero I→E, both trainable. |
   | Fixed PING | Freeze a demonstrably PING-capable E/I loop; train input/readout parameters under the same objective. Verify its realized rhythm after learning and across input conditions. |
   | Trained nonoscillatory feedback | Optimize inhibitory feedback within a declared family whose held-out dynamics meet a preregistered nonoscillatory criterion. Explore heterogeneity/noise or dispersed timing with an equal tuning budget and characterize the resulting latency and conductance changes. |

   “Nonoscillatory” is a measured property, not a name for one parameter setting. This comparator must learn the task competitively and have overlapping activity ranges. Simply slowing inhibition until gamma disappears confounds oscillation with inhibition kinetics. If no adequate comparator is obtained, report that limitation rather than claiming timing specificity.

   Add a same-substrate loop-disabled control for mechanistic interpretation, a trainable loop initialized in a PING regime for accessibility, and a tiny-positive-return initialization for sensitivity. These are supplementary conditions, not substitutes for the exact-zero main test.

   To study learning pressure, cross competitive versus noncompetitive training with zero versus positive spike cost. Two complementary noncompetitive controls are an equal-total-rate single-template task and the same mixture streams with an objective to report all component strengths. Neither is a perfectly difficulty-matched task; record their learning curves and activity, and avoid attributing every difference to competition alone. The same-mixture control is especially useful for separating input statistics from the requested computation.

   A future cued-target task can test relevance independently of physical strength, including cases where the distractor is strongest. Keep this extension separate from the first experiment's claim.

9. **Establishing PING rather than labeling a spectrum**

   Use complementary evidence:

   1. Population activity has repeatable oscillatory structure above an appropriate nonrhythmic null. Predeclare a gamma convention, for example 30–90 Hz, but inspect the wider spectrum and report slower E/I rhythms under their actual frequencies.
   2. E volleys precede I recruitment; inhibitory conductance rises after recruitment and reduces E firing; its decay precedes recovery. Quantify distributions of these delays, not just a representative raster.
   3. Disrupting either loop arm changes the rhythmic organization. Because silencing E input to I can also remove I activity, include a mean-drive replacement control when asking specifically about coherent recruitment.
   4. The rhythm occurs in single runs under stationary, conditionally independent drive and is not solely a train of stimulus-onset transients or a feature of averaging aligned trials.
   5. Report sparse participation and variability across cycles. Individual E cells need not fire on every population cycle.

   The reciprocal synchronization mechanism is established in [Börgers & Kopell, 2003](https://zpkilpat.github.io/appm4370/papers/borgers03.pdf). The proposed diagnostic bundle adapts that mechanism to this experiment; it is not a universally standardized PING certification test.

   Short 150 ms intervals contain few gamma cycles. Add diagnostic plateaus lasting 1–2 seconds at selected mixtures and intensities to estimate spectra reliably. Keep these diagnostic plateaus separate from the main behavioral evaluation. Do not concatenate dissimilar intervals with discontinuities and interpret the resulting spectrum as a stationary oscillator.

   Record E and I population rates, conductances, representative voltage traces, and full spikes for prespecified diagnostic subsets. Use single-stream spectra, autocorrelation, E/I cross-correlation, spike-to-population phase locking with the measured cell excluded, and per-cycle participation. Validate the estimators with synthetic nonrhythmic and oscillatory signals during implementation. A maximum bin in a frequency range always exists and is not sufficient evidence. [Ray & Maunsell, 2011](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000610) demonstrate why broadband activity and genuine gamma rhythms must be distinguished.

10. **Timing interventions and their limits**

    Start with paired input realizations and identical initial state at the intervention time. Measure immediate effects and longer adaptation separately. Ordinary loop ablation establishes feedback dependence but changes both timing and the amount of inhibition.

    Use several interventions with distinct interpretations:

    1. **Controlled inhibitory replay:** record I spike trains in a reference run, then replace the return-path drive in both a faithful-replay condition and a temporally altered replay condition. Preserve each neuron's spike count and its outgoing weights. Perform initial analyses within stationary plateaus, handle kernel tails and boundaries explicitly, and verify integrated inhibitory conductance at each E target. Both conditions have an opened loop, so their difference isolates the replay manipulation within that artificial preparation. Faithful replay must recover the reference trajectory closely enough to be informative. This is an offline mechanistic counterfactual, not a deployable baseline or evidence of ongoing feedback function.
    2. **Desynchronize inhibitory cells:** use independent bounded timing perturbations or plateau-wise shifts to weaken population coherence while approximately retaining slower rate envelopes. Specify the preserved statistic. A surrogate appropriate for a statistical null is not automatically a biologically selective intervention. [Conditional jitter methods](https://journals.physiology.org/doi/abs/10.1152/jn.00633.2011) provide a principled starting point for defining what is held fixed.
    3. **Disrupt regular recurrence while retaining volleys:** perturb inter-volley intervals while retaining within-volley spike membership and spread. This distinguishes a benefit of regular cycles from a benefit of synchronous bursts. It also changes temporal alignment to the stimulus, so treat it as complementary evidence.
    4. **Modify live feedback timing:** add causal transmission delays or delay dispersion during inference. This preserves the feedback route but generally changes subsequent spikes and conductances; count every resulting spike and compare with matched-delay controls. Never imply that a live perturbation preserves activity merely because it preserves spikes already emitted.
    5. **Probe E-output timing separately:** replay the same E spike counts into the frozen spiking readout with altered cross-neuron timing. A loss here identifies downstream use of temporal organization; it does not by itself show that E competition required gamma.

    For each intervention report E, I, and output spike counts, mean and distribution of inhibitory conductance, temporal envelopes, suppression of the targeted timing statistic, and behavior. Use prespecified tolerances, provisionally 5% for population mean rates and mean inhibitory conductance in a calibrated matching condition; inspect conditional rates and per-cell distributions as well. If matching cannot be achieved, classify the result as confounded rather than excluding it silently.

    Equal inhibitory spike counts imply equal total conductance area only under specified fixed kernels/weights and suitable boundary treatment. They do not imply equal inhibitory current: conductance-based current also depends on membrane voltage. Holding every downstream effect fixed would remove the causal pathway being tested. Thus present the natural total effect of the intervention alongside the calibrated activity-matched comparison; do not claim one perfect “timing-only” manipulation.

    Acute degradation tests use by the frozen trained network. Retrain a timing-disrupted or asynchronous comparator with a comparable optimization budget to test whether another solution can recover performance. Neither acute fragility nor cross-model superiority alone proves gamma necessary for this computational task.

11. **Behavior, efficiency, and statistical units**

    The primary result should be a frontier, not one weighted score. Compare continuous accuracy, stable switching latency, and total spikes per second across penalty weights and readout windows. Compare one metric while matching the other two: fewer spikes at matched accuracy and latency, faster responses at matched accuracy and activity, or better accuracy at matched latency and activity. An improvement is Pareto-relevant when it reduces one cost without an unacceptable deterioration of the others, using declared tolerances.

    Evaluate predictions on a common temporal grid, for example every 5 ms, with a randomized grid offset per stream. This observation schedule is not an input to the network. Include transition periods in primary accuracy and also report steady-interval accuracy separately.

    Define switching latency from a genuine change in the latent winning identity to the first correct output maintained for at least 20 ms. The 20 ms hold is a proposed measurement convention, not a guarantee of correct evidence. Report nonresponses and events censored by the next switch; do not calculate latency solely among successful fast switches. Also report false switches during constant-winner intervals and the fraction correct by fixed deadlines.

    “Spikes per correct decision” needs a fixed definition of decision. For fixed, nonoverlapping scoring bins, divide **all network spikes over the scored time**, including errors and failed switches, by the number of correctly classified bins. Do not divide only spikes during successful bins. Also report spikes per second and accuracy, because changing bin width otherwise changes the ratio. Overlapping rolling predictions are correlated observations, not independent trials.

    Stratify by top-two margin, template overlap, active-component count, intensity, transition type, dwell time, and naturally occurring phase at a change. Prespecify the primary comparisons; secondary interactions are exploratory unless powered and declared beforehand.

    The unit of learning replication is an independently trained network. Pair stimulus seeds across conditions, bootstrap or model uncertainty across training seeds, and treat streams/events as nested observations. Show the fraction of trained seeds developing functional feedback and PING, including failed and silent seeds. Do not turn thousands of timesteps from one trained network into thousands of learning replicates.

12. **A staged execution plan**

    1. **Task and optimizer pilot:** check generator identifiability, causal observer performance, state continuity, initial I activity, true zero-return E dynamics, and accessible return-weight updates. Establish whether the penalty prevents bootstrapping. These checks separate an invalid task or implementation from a negative scientific result.
    2. **Small discovery study:** use three independent training seeds per core condition and a small penalty grid. Calibrate intensity, margins, dynamic range, readout, and optimization stability using training/validation streams only. Predefine a stop/revise decision if no baseline can perform or all perform near ceiling.
    3. **Freeze the confirmatory protocol:** specify generator, model family, selection rule, checkpoint criterion, rhythm definitions, intervention windows, matching tolerances, failure handling, and primary endpoints. A planning example is four main conditions × three penalties × ten training seeds = 120 trainings; actual seed count should follow pilot variability and the desired precision, not a claim that ten guarantees power.
    4. **Confirm and intervene:** evaluate held-out streams, long diagnostic plateaus, timing interventions, and initialization controls. Preserve all seed outcomes. Use the pressure controls to assess whether competitive selection changes the probability of learning functional rhythmic feedback.
    5. **Robustness:** investigate different readout windows, heterogeneity, inhibitory kinetics, timestep, longer gradient horizons, new template families with retraining, and the cued-target extension. Keep discovery-based extensions distinguishable from the confirmatory result.

    Useful figures are the circuit and generator; learning trajectories for loop strength/activity/rhythmicity; representative prespecified rasters with inhibitory conductance; accuracy–latency–spike frontiers; timing intervention effects with achieved activity matching; and phase-dependent switching/competition curves. A frontier colored by measured rhythmicity is descriptive, not causal evidence by itself.

13. **Connection to the current Pinglab implementation**

    Live source inspection found separately trainable E→I and I→E matrices in [models.py](/Users/eoin/pinglab/tools/snnsim/models.py:849), direct recurrent conductance increments in [the stepping code](/Users/eoin/pinglab/tools/snnsim/models.py:1423), and nonnegative post-update projection in [project_dales](/Users/eoin/pinglab/tools/snnsim/models.py:1653). These support the proposed initialization in principle; accessibility still needs a focused implementation check when this experiment is built.

    The current [canonical recipe](/Users/eoin/pinglab/experiments/exp022/recipe.py:110) uses the same voltage-gradient damping setting for COBA and PING. Do not copy an older remembered difference between those conditions. The existing [recurrent-training article](/Users/eoin/pinglab/writings/exp049.typ:33) reports weakened rhythmicity and persistence of jointly zero recurrence under its particular classification protocol. That is relevant motivation, not a universal negative result, and it differs from retaining initially active E→I in this proposal. The article's task, readout, and lack of a firing-rate penalty also differ from this proposal.

    Continuous mixture generation, genuinely rolling spiking decisions, population-complete activity loss, and the intervention machinery require explicit implementation and verification. This research pass did not establish that those features are already available end to end.

    Under the current [Experiment Runner Guide](/Users/eoin/pinglab/experiments/README.md), training, held-out forward evaluation, and perturbation simulations belong in compute; measurements belong in analyse; report graphics belong in present. Implementation must also follow the versioned Storage Guide. No simulation or storage migration was performed for this document.

14. **Interpretation of possible outcomes**

    | Outcome | Supported interpretation |
    | --- | --- |
    | Return weights never become useful | Acquisition failed under the tested objective and optimizer; first exclude inaccessible gradients and an uninformative task. |
    | Useful inhibition appears without PING | The task favors feedback inhibition, with no support for a gamma requirement. |
    | PING develops without a better frontier | Rhythmic feedback can be acquired, but its computational benefit is unestablished. |
    | A better frontier appears but timing interventions leave it intact | Feedback or another learned property is beneficial; timing-specific advantage is unsupported. |
    | Timing perturbation hurts only when activity or mean inhibition changes | The intervention is insufficient to distinguish timing from those effects. |
    | Gamma arises equally in noncompetitive controls | The claim that competitive selection specifically supplies the pressure is weakened. |
    | A learned loop improves the frontier and several calibrated timing interventions remove the advantage | Evidence for useful acquired PING timing within the tested architecture and task family. |
    | Only an already rhythmic initialization succeeds | PING is useful and trainable once present, but emergence from zero return has not been shown. |

    The central preregistrable question is: **Does continuous selection under a total spike budget make an initially open E/I circuit learn rhythmic inhibitory feedback, and does that feedback's temporal organization improve the attainable trade-off between accuracy, response time, and total activity?** A credible negative result is informative because the controls distinguish failed learning, useful nonoscillatory inhibition, acquired but unnecessary gamma, and a task that never required network-level selection.
