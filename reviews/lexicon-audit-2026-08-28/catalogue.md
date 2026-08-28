# Candidate lexicon: complete cross-file inventory

Generated from `candidates.txt`. Proposed names are recommendations, not adopted policy. Occurrences include prose, equations, code, metadata and comments; context labels in the evidence are heuristic. A cross-file candidate must occur outside comments, metadata and bibliography in at least two files. Counts measure textual reuse, not independent experiments or verified claims.

Every row links to all matching locations in the evidence index. Matched spelling variants and exact excerpts are in `occurrences.json`. Nested categories may overlap; counts must not be summed as independent concepts.

## Models and populations

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L001](evidence.md#l001) | PING | 30 / 424 | Use “pyramidal–interneuron network gamma (PING)” for the mechanism; qualify PING circuit/configuration separately from demonstrated PING activity. Preserve the `PING` return type and `ping` API spelling. |
| [L002](evidence.md#l002) | COBA / conductance-based model | 21 / 239 | Reserve COBA for conductance-based dynamics; call the experimental control “loop-off COBA”. PING networks here are also conductance based. |
| [L003](evidence.md#l003) | COBANet | 7 / 14 | Reserve `COBANet` for the implementation class; it is neither the loop-off condition nor proof of rhythmic activity. |
| [L004](evidence.md#l004) | Current-based / CUBA | 3 / 5 | Use current-based for the model family and distinguish it from a conductance model approximated with fixed driving forces. |
| [L005](evidence.md#l005) | LIF neuron | 18 / 63 | Use leaky integrate-and-fire (LIF); qualify conductance based, current based, output, and spiking status. Preserve `COBA_LIF` and `LIF` type names. |
| [L006](evidence.md#l006) | Leaky integrator | 7 / 12 | Use non-spiking leaky integrator (LI) for a state without spike/reset; do not equate it with a spiking output LIF just because both average voltage. |
| [L007](evidence.md#l007) | Spiking neural network | 11 / 55 | Use spiking neural network (SNN); do not use SNN as a synonym for PING or the simulator. |
| [L008](evidence.md#l008) | Excitatory neuron / E population | 23 / 82 | Use excitatory neuron for one neuron and excitatory population (E) for the set; qualify hidden E versus input/output neurons. |
| [L009](evidence.md#l009) | Inhibitory neuron / I population | 25 / 74 | Use inhibitory neuron and inhibitory population (I); do not use I for electrical current without a distinguishing subscript. |
| [L010](evidence.md#l010) | Cell (neuron, network replicate, component) | 27 / 258 | Split neuron, trained network replicate, condition–seed cell and PING component. “Cell” currently denotes all four; this is a high-priority semantic collision. |
| [L011](evidence.md#l011) | Population / pool | 34 / 269 | Prefer population for neurons; qualify training pool, sample pool and inhibitory population size. A training pool is not a neural population. |
| [L012](evidence.md#l012) | Network / circuit / motif | 40 / 389 | Use network for a full model, circuit for a connected neuronal subsystem, motif for reusable topology; attach replicate identity separately. |
| [L013](evidence.md#l013) | Hidden layer / hierarchy | 6 / 11 | State whether a layer is an E population or an E/I component. Keep layer index separate from population and network identity. |
| [L014](evidence.md#l014) | Input channels / afferents | 13 / 24 | Use input channel for encoded features and afferent for a biological/model connection; distinguish channel count from neuron count and fan-in. |
| [L015](evidence.md#l015) | Class / output neurons | 8 / 20 | Use output LIF neuron when it spikes; class output/logit for a returned score, not necessarily a neuron. |
| [L016](evidence.md#l016) | Feedforward control / loop-off | 5 / 33 | Use loop-off control with explicit remaining pathways. Disabling E↔I is not necessarily removing every recurrent pathway or the I population. |
| [L017](evidence.md#l017) | PING loop / E–I feedback | 16 / 67 | Use local E→I→E feedback loop, with enabled pathways and trainability explicit. Separate local recurrence from cross-network coupling. |
| [L018](evidence.md#l018) | Frozen / fixed / untrained | 19 / 107 | Distinguish frozen parameters, untrained initialization and frozen trained checkpoint. Frozen does not mean untrained. |
| [L019](evidence.md#l019) | E→E, E→I, I→E, I→I pathways | 21 / 151 | Standardize source→target direction in prose; add network/layer identity for inter-circuit edges. Do not infer matrix axis order from a pathway label. |
| [L020](evidence.md#l020) | Asynchronous / irregular / balanced activity | 4 / 17 | Keep asynchronous, irregular and balanced as separate properties with separate diagnostics; absence of visible bands does not establish all three. |
| [L021](evidence.md#l021) | Population-rate / mean-field / neural-mass model | 5 / 61 | Name the specific four-variable population-rate closure; retain mean-field/neural-mass as broader descriptions, not interchangeable guarantees of derivation. |
| [L022](evidence.md#l022) | Model / recipe / canonical defaults | 22 / 158 | Namespace library default, experiment reference recipe and fitted/estimated parameter. The 6 ms and 9 ms GABA references must not share an unqualified “canonical” value. |

## Membrane and synapses

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L023](evidence.md#l023) | Membrane voltage / potential | 12 / 35 | Prefer membrane voltage; specify candidate/pre-reset, retained/post-reset, or voltage above rest and physical versus dimensionless output units. |
| [L024](evidence.md#l024) | Membrane capacitance | 8 / 56 | Use C_m with receiving-population index; nF in biophysical equations. Do not reuse bare C for amplitude-fit factors in shared equations. |
| [L025](evidence.md#l025) | Leak conductance | 9 / 72 | Use g_L and population index when required; distinguish conductance from the leak decay factor or membrane time constant. |
| [L026](evidence.md#l026) | Rest / leak / reset potential | 9 / 77 | Keep E_L, initial voltage and V_reset distinct even when numerically equal. Equality of default values is not an alias of concepts. |
| [L027](evidence.md#l027) | Spike threshold | 8 / 21 | Prefer V_th for biophysical voltage threshold and an explicitly dimensionless output threshold. Do not share bare theta with parameter vectors or relative phase. |
| [L028](evidence.md#l028) | Refractory period and counter | 4 / 19 | Use tau_ref for physical duration and n_ref for an integer counter; record discretization and step gating separately. |
| [L029](evidence.md#l029) | Reversal potential | 10 / 96 | Standardize E_exc/E_inh/E_L or one consistent e/i convention; use population indices separately from synapse polarity. |
| [L030](evidence.md#l030) | Conductance / channel polarity | 23 / 208 | Use g for conductance states, not signed currents. Explicitly name synapse polarity and receiving population rather than overloading E/I positions. |
| [L031](evidence.md#l031) | Excitatory conductance / AMPA | 16 / 148 | Use g_exc and tau_AMPA (with population/pathway qualifier as needed); AMPA-like model filtering is not an additional biological measurement. |
| [L032](evidence.md#l032) | Inhibitory conductance / GABA | 21 / 181 | Use g_inh and tau_GABA; separate inhibitory event rate, integrated conductance and signed inhibitory current. |
| [L033](evidence.md#l033) | Total conductance | 4 / 25 | Use g_tot = leak plus active synaptic conductances; avoid confusing summed weights G with instantaneous total conductance g_tot. |
| [L034](evidence.md#l034) | Synaptic current / signed current | 5 / 13 | Choose and state inward-positive or outward-positive current. exp023 uses inward current while exp100 starts with outward ionic current; preserve the sign bridge. |
| [L035](evidence.md#l035) | Driving force | 5 / 43 | Distinguish signed V−E from positive force magnitude. exp033 and exp109 use opposite signs for the symbol Delta V_exc. |
| [L036](evidence.md#l036) | Membrane time constant | 6 / 54 | Use tau_m for passive membrane time and tau_eff for conductance-dependent time; population-rate relaxation times require a closure qualifier. |
| [L037](evidence.md#l037) | Effective time constant / shunting | 5 / 19 | Define tau_eff from capacitance and total conductance. “Shunt” should describe conductance effects, not merely high I spike rate. |
| [L038](evidence.md#l038) | Synaptic decay / timescale | 19 / 64 | Use synaptic decay time constant, in ms. Do not relabel it transmission delay, phase lag or oscillation period. |
| [L039](evidence.md#l039) | Conductance increment / spike kick | 8 / 23 | Prefer conductance increment per event, with units and pathway. Keep the decay-then-add ordering part of the definition. |
| [L040](evidence.md#l040) | Equilibrium voltage / stationary voltage | 4 / 27 | Use V_inf for the fixed-conductance equilibrium and a qualified operating-point voltage for evaluation at mean conductance; neither is automatically a mean trajectory voltage. |
| [L041](evidence.md#l041) | Exponential Euler | 5 / 7 | Use exponential Euler with conductances held fixed over a step; separate the exactly solved held system from discretization of the coupled spiking model. |
| [L042](evidence.md#l042) | Forward Euler | 2 / 5 | Preserve forward Euler as a distinct integrator, not a short name for exponential Euler. |
| [L043](evidence.md#l043) | Zero-order hold | 2 / 4 | Use zero-order hold on conductance; state which updated conductance is frozen and at what point in the schedule. |
| [L044](evidence.md#l044) | Reset rule | 15 / 55 | Distinguish hard voltage reset, subtractive output reset, runtime-state reset and segment-boundary reset. These are different operations. |
| [L045](evidence.md#l045) | Adaptation / trainable leak | 6 / 11 | Keep adaptive thresholds and learned membrane/leak parameters separate; explicitly identify disabled optional mechanisms in a recipe. |

## Connectivity and initialization

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L046](evidence.md#l046) | Weight matrix / projection weights | 8 / 9 | Use W for a stored matrix with unit, shape and source/target axes; do not reuse W unqualified for a scalar summed coupling. |
| [L047](evidence.md#l047) | Input weights | 9 / 41 | Standard name input projection weights, W_in; preserve implementation aliases and distinguish initializer mean from inference scaling. |
| [L048](evidence.md#l048) | Readout weights | 11 / 23 | Use readout projection weights, W_out, with decoder-specific units and shape. Direct initialization differs from fan-in-normalized synaptic initialization. |
| [L049](evidence.md#l049) | Recurrent weights | 11 / 40 | Qualify pathway, initialization and trainability; recurrent conductance weights are non-negative magnitudes under the stated constraint. |
| [L050](evidence.md#l050) | Coupling strength / loop gain | 11 / 29 | Split parent initializer mean, expected summed conductance, individual edge strength and dimensionless scale. A fitted effective phase interaction is another quantity. |
| [L051](evidence.md#l051) | Summed coupling | 8 / 50 | Prefer G_source→target for expected summed conductance; explicitly distinguish nominal parent mean from post-clamp expectation and current-valued rescaling. |
| [L052](evidence.md#l052) | Per-synapse strength | 2 / 13 | Use expected per-synapse conductance j and realized edge W_ij separately; report nS or μS explicitly. |
| [L053](evidence.md#l053) | Fan-in / source population size | 14 / 29 | Use N_pre for possible source count and K_in for retained afferent count. Distinguish dense normalization by N_pre from exact fan-in connectivity. |
| [L054](evidence.md#l054) | Weight orientation | 5 / 8 | Keep graph [target, source] and runtime [source, target] as distinct documented layouts; record the transpose explicitly instead of silently changing equations. |
| [L055](evidence.md#l055) | Lower-clamped normal | 8 / 15 | Canonical lower-clamped normal initializer. Do not rename it half-normal/truncated normal; retain Normal as an API compatibility spelling where documented. |
| [L056](evidence.md#l056) | Initializer / initialization | 22 / 91 | Prefer initialization/initializer in code-facing prose, with declared distribution, scaling and realized statistics. Choose one prose spelling rather than changing APIs. |
| [L057](evidence.md#l057) | Parent mean and standard deviation | 9 / 20 | Use mu_init and sigma_init for the parent distribution with scale units. They are not the observed mean/SD of stored positive edges. |
| [L058](evidence.md#l058) | Initial-zero fraction | 9 / 11 | Prefer initial_zero_fraction and q_zero; zeros that remain trainable are not structural sparsity or permanent pruning. |
| [L059](evidence.md#l059) | Structural sparsity / mask | 7 / 7 | Separate connectivity mask from zero-valued dense parameters and sampled display paths. State whether zeros can regrow. |
| [L060](evidence.md#l060) | Exact fan-in / exact K | 4 / 7 | Use exact fan-in K_in; distinguish exact support count from realized positive count after lower clamping and from fixed connectivity. |
| [L061](evidence.md#l061) | Sparsity compensation | 4 / 5 | Name survivor rescaling separately from fan-in normalization; specify the retained fraction and the expectation being preserved. |
| [L062](evidence.md#l062) | Dale constraint / non-negative projection | 12 / 26 | Use Dale-constrained conductance magnitudes and distinguish forward clamping from post-optimizer projection. Do not equate Dale's law with every matrix being signed identically. |
| [L063](evidence.md#l063) | Inference weight scaling | 4 / 15 | Use alpha_path for dimensionless inference multiplication; keep distinct from changing initializer strength or training weights. |
| [L065](evidence.md#l065) | Coupling ratio | 3 / 6 | Use rho_IE/EI for a dimensionless coupling ratio. Avoid bare r, already used for firing rate. |

## Time and input

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L066](evidence.md#l066) | Integration timestep | 37 / 217 | Use Delta t in mathematics and dt_ms at explicit boundaries. Do not also use Delta t for a distinct analysis-bin width. |
| [L067](evidence.md#l067) | Physical duration / presentation window | 22 / 61 | Use T_present for physical presentation duration and T_sim for a whole simulation; state ms/s at every conversion boundary. |
| [L068](evidence.md#l068) | Number of simulation steps | 18 / 40 | Use N_t for timestep count, distinct from physical T and neuron count N; tensor axes keep stable time/batch/channel names. |
| [L069](evidence.md#l069) | Continuous time versus step index | 7 / 7 | Reserve t for physical time and k for a discrete update where possible; when preserving existing notation, state the mapping t_k = k Delta t. |
| [L070](evidence.md#l070) | Readout / decision window | 4 / 14 | Distinguish T_readout from T_present even in matched-window experiments; equality is a protocol choice, not a universal identity. |
| [L071](evidence.md#l071) | Burn-in / transient exclusion | 6 / 8 | Standardize burn-in duration and measurement interval; zero burn-in must be explicit and distinct from equilibrium or steady-state evidence. |
| [L072](evidence.md#l072) | Analysis bin / lag bin | 8 / 15 | Qualify time-bin width, frequency-bin spacing and histogram-bin width. They must not inherit the simulation timestep by notation alone. |
| [L073](evidence.md#l073) | Transmission delay / causal delay | 5 / 7 | Use d_delay (ms) and n_delay (steps); distinguish extra declared delay, next-step recurrence and physiological filtering lag. |
| [L074](evidence.md#l074) | Poisson input / rate encoder | 21 / 71 | Use Poisson-rate encoding for the model and Bernoulli-per-step implementation where applicable; nominal rate is not realized count/duration. |
| [L075](evidence.md#l075) | Bernoulli event probability | 16 / 21 | Use p_event = r_input Delta t_ms / 1000; qualify pixel scaling, upper probability boundary and binary collision handling. |
| [L076](evidence.md#l076) | Maximum pixel / encoding rate | 18 / 90 | Prefer maximum-pixel encoding rate r_input,max for images and uniform channel rate r_input for uniform drive. Do not reuse the hidden-rate ceiling r_max. |
| [L077](evidence.md#l077) | Nominal versus realized rate | 6 / 15 | Keep requested generator rate and measured event rate as separate fields; especially important after jitter, union and boundary clamping. |
| [L078](evidence.md#l078) | Variable-rate / categorical input | 7 / 24 | Use categorical variable-rate encoding and name the sampling unit (presentation), rate set, probabilities and seed. |
| [L079](evidence.md#l079) | Shared / private / independent input | 7 / 30 | Distinguish private channels, shared afferents, grouped shared events and quenched conductance. Private input need not imply independent outputs. |
| [L080](evidence.md#l080) | Background / external / tonic drive | 8 / 49 | Qualify spike-rate drive, conductance drive and injected current; they have different units and are not interchangeable controls. |
| [L081](evidence.md#l081) | Shot noise / event timing | 4 / 12 | Use filtered conductance shot noise for the event-driven process; separate finite-window simulation from stationary Gaussian approximations. |
| [L082](evidence.md#l082) | MNIST and dataset identity | 24 / 81 | Keep the dataset name, official partition, selected subset and encoder separate. MNIST is not a synonym for a particular split or input tensor. |
| [L083](evidence.md#l083) | SHD / event dataset | 4 / 15 | Preserve SHD as a dataset identity; distinguish dataset timestamps, prebinned spikes and graph-timestep event binning. |
| [L084](evidence.md#l084) | Continuous stream / segment / boundary | 7 / 60 | Use continuous hidden-state stream with known decision boundaries when that is the protocol. No hidden reset does not imply blind segmentation. |
| [L085](evidence.md#l085) | Trial / presentation / sample | 33 / 369 | Define dataset sample, stochastic presentation, stream and trial explicitly. One image may have multiple presentations; samples are not necessarily independent replicates. |
| [L086](evidence.md#l086) | Random seed / stochastic stream | 37 / 332 | Namespace initialization, data split, order, encoding, intervention and display seeds. Multiple encoder draws are not independent trained networks. |

## Activity and readout

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L087](evidence.md#l087) | Spike indicator / spike train | 15 / 26 | Use s[k] for a binary discrete event and an explicitly defined impulse train for continuous time; their units differ. |
| [L088](evidence.md#l088) | Spike count | 19 / 51 | Use n_spikes or C_spike for a count; specify neuron/population, observation window and pooling. Do not label counts in Hz. |
| [L089](evidence.md#l089) | Population-mean firing rate | 17 / 52 | Use mean per-neuron population rate r_P in Hz with P = E/I, duration and sample aggregation explicit. |
| [L090](evidence.md#l090) | Per-neuron firing rate | 7 / 16 | Separate each neuron's rate, population mean, maximum-neuron rate and median-neuron rate. They are different statistics. |
| [L091](evidence.md#l091) | E/I rate source aliases | 12 / 84 | Candidate canonical field stems e_rate_hz and i_rate_hz, with split/checkpoint/aggregation metadata. Do not mutate existing schemas during a prose audit. |
| [L092](evidence.md#l092) | Total population activity / spike economy | 3 / 7 | N_E r_E + N_I r_I is total spike rate, not count; multiply by duration for count and divide by total neurons for a network mean. Costs require a separate model. |
| [L093](evidence.md#l093) | Sparsity / low activity | 21 / 37 | Qualify activity sparsity, connectivity sparsity, initial zero fraction or input-event scarcity; one word must not collapse these dimensions. |
| [L094](evidence.md#l094) | Rate ceiling / activity penalty | 11 / 37 | Use soft hidden-E population-rate ceiling and one-sided quadratic rate penalty; a soft loss is not a hard spike budget or a structural rate limit. |
| [L095](evidence.md#l095) | Rate ceiling parameter aliases | 9 / 28 | Prefer r_E,ceil (Hz) for the hidden-E ceiling and ceiling_hz in a new interface. Record legacy theta_u conversion rather than treating its raw value as Hz. |
| [L096](evidence.md#l096) | Rate penalty coefficient | 6 / 17 | Prefer lambda_rate with Hz^-2 when multiplying a squared excess rate to a dimensionless loss; avoid bare lambda shared with eigenvalues. |
| [L097](evidence.md#l097) | Rate floor / activity floor | 5 / 10 | Distinguish observed minimum, soft training target, dynamical lower bound and decoder-relative input floor. A sampled plateau is not a proved attractor. |
| [L098](evidence.md#l098) | Mean-voltage readout | 18 / 48 | Canonical “mean pre-reset output-LIF voltage readout” where applicable; keep non-spiking LI averages separate, including output dynamics and units. |
| [L099](evidence.md#l099) | Final-voltage readout | 2 / 3 | Retain final-voltage as a separate temporal reduction; it is not an alias of a whole-window mean. |
| [L100](evidence.md#l100) | Spike-count readout | 4 / 10 | Use output-neuron spike-count readout and dimensionless logits; distinguish from summing hidden spikes before a linear projection. |
| [L101](evidence.md#l101) | Spike-rate readout | 4 / 8 | Use duration-normalized output spike rate (spikes/s); distinguish legacy `rate`, which is documented as an unnormalized hidden-count projection. |
| [L102](evidence.md#l102) | Cumulative-potential readout | 4 / 5 | Keep this exact named readout separate from voltage mean and spike count; include the actual softmax accumulation and leak definition. |
| [L103](evidence.md#l103) | Legacy rate readout | 2 / 5 | Preserve `rate` as the legacy API token but describe it as hidden-spike-count linear readout; do not promise time normalization. |
| [L104](evidence.md#l104) | Class score / logit / prediction | 18 / 72 | Prefer z_c for unnormalized class score, y for true label and hat(y) for predicted label; count and voltage logits remain decoder-specific. |
| [L105](evidence.md#l105) | Softmax share / confidence | 9 / 27 | Use softmax score/share unless calibration is established; do not equate normalized scores with calibrated confidence or information content. |
| [L106](evidence.md#l106) | Classification accuracy | 22 / 329 | Record split, checkpoint role, reduction and fraction-versus-percent units. “Final accuracy” must say final epoch or selected checkpoint. |
| [L107](evidence.md#l107) | Chance / silent output | 9 / 33 | Separate nominal class chance, observed empty-input accuracy and silent-output fraction. State tie-breaking for zero scores. |
| [L108](evidence.md#l108) | Psychometric / duration–rate curve | 5 / 32 | Prefer accuracy-versus-input-rate curve unless a fitted psychometric model is actually specified; retain duration, decoder and sampling protocol. |
| [L109](evidence.md#l109) | Filtered pixel feature | 2 / 8 | Use mean depolarization feature z_feature (mV), distinct from class logits z and spike counts z_bn. |
| [L110](evidence.md#l110) | Output / observable / recording | 12 / 50 | Distinguish returned graph output, internal observable declaration and retained recording. All can be signals without having the same API role. |
| [L111](evidence.md#l111) | Raster / population trace | 23 / 157 | Define raster as event coordinates and population trace as an explicit time-bin reduction; specify summed counts versus mean per-neuron rate. |
| [L234](evidence.md#l234) | Energy / efficiency / economy | 4 / 17 | Separate spike counts/rates from modeled energy and measured wall-clock efficiency; specify E/I weighting and units before making an economy claim. |

## Training and evaluation

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L112](evidence.md#l112) | Cross-entropy / objective / loss | 11 / 42 | Use L_CE for data loss and L_total for the combined objective; state reduction, log base and whether the rate penalty is included. |
| [L113](evidence.md#l113) | Adam / AdamW | 16 / 25 | Keep optimizer names distinct, recording weight decay explicitly; source disagreement cannot be repaired by spelling normalization alone. |
| [L114](evidence.md#l114) | Learning rate | 15 / 21 | Use eta for learning rate; distinguish global optimizer rate from named per-parameter-group rates and actual optimizer state. |
| [L115](evidence.md#l115) | Batch / minibatch | 17 / 36 | Use B for examples/presentations in the current minibatch; distinguish simulated parallel trials and independent stream count from dataset size. |
| [L116](evidence.md#l116) | Epoch / update / training horizon | 23 / 201 | Use e for epoch and u_update for optimizer update; distinguish initialization snapshot, first completed epoch and final epoch. |
| [L117](evidence.md#l117) | Surrogate gradient / fast sigmoid | 9 / 33 | Specify forward spike and backward surrogate separately, including numerator normalization, threshold distance units and slope k_sg. |
| [L118](evidence.md#l118) | BPTT / recurrent derivative | 3 / 8 | Use backpropagation through time (BPTT); distinguish discrete update Jacobian from the continuous-time vector-field Jacobian. |
| [L119](evidence.md#l119) | Voltage-gradient damping | 11 / 23 | Use voltage-increment gradient damping and d_grad; preserve exact CLI/API spellings. It scales the increment path, not the complete state derivative. |
| [L120](evidence.md#l120) | Stop-gradient / detach | 4 / 13 | Distinguish stop-gradient boundary, detached runtime snapshot and straight-through gradient-scaling operator; all alter graph dependencies differently. |
| [L121](evidence.md#l121) | Gradient clipping / non-finite update | 5 / 8 | Use global gradient-norm clipping with stated norm/threshold and separate skipped-update counts; clipping is not voltage-gradient damping. |
| [L122](evidence.md#l122) | Training / validation / test split | 18 / 71 | Name optimizer-training, validation and official-test partitions explicitly. “Held-out” is insufficient where selection and final evaluation differ. |
| [L123](evidence.md#l123) | Checkpoint selection | 14 / 21 | Qualify selection objective, split, encoding draws, tie-break and invocation scope. Graph lowest-update-loss and validation selection are not aliases. |
| [L124](evidence.md#l124) | Final checkpoint / endpoint dynamics | 15 / 45 | Use final-epoch parameter checkpoint for training endpoints and final-update training checkpoint for graph update state; preserve role metadata. |
| [L125](evidence.md#l125) | Parameter checkpoint / resume checkpoint | 3 / 18 | Separate parameter-only checkpoint, training-resume checkpoint and dynamic runtime state; filenames alone do not identify selection role. |
| [L126](evidence.md#l126) | Frozen evaluation / replay / inference | 27 / 191 | Define parameter loading, stochastic input replay and state continuation separately. Replaying learned weights does not guarantee replaying the same trajectory. |
| [L127](evidence.md#l127) | Learning history / trajectory / convergence | 13 / 41 | Use finite training trajectory and operational final-window stability criterion; reserve convergence/attractor claims for the relevant evidence. |
| [L128](evidence.md#l128) | Accuracy–rate frontier | 7 / 29 | Use sampled accuracy–rate trade-off; “frontier”, matched accuracy and equivalence require stated construction or comparison criteria. |
| [L129](evidence.md#l129) | Weight decay | 9 / 13 | Keep weight decay distinct from activity regularization, hard projection and initialization sparsity; record optimizer implementation and coefficient. |
| [L130](evidence.md#l130) | Replicate / condition / sweep cell | 25 / 159 | Condition = parameter/protocol setting; replicate = independently initialized/trained network; reserve condition–seed cell for campaign bookkeeping. |
| [L131](evidence.md#l131) | Regularization versus constraint | 12 / 34 | Separate soft objective penalties, hard admissibility constraints and immutable experimental controls; use the exact implemented operation in the lexicon. |
| [L236](evidence.md#l236) | Calibration / benchmark / integration test | 10 / 30 | Distinguish decoder-relative calibration, scientific benchmark and interface test. Shared vocabulary should preserve the scope of each result. |

## Rhythms and statistics

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L132](evidence.md#l132) | Gamma frequency / spectral peak | 15 / 107 | Use estimated population spectral-peak frequency with estimator and band; f_gamma is not proof that a 5–150 Hz peak is gamma. |
| [L133](evidence.md#l133) | Gamma band / peak-search band | 4 / 8 | Keep conventional gamma band and estimator search interval as separate named ranges; do not define 5–150 Hz as the gamma band. |
| [L134](evidence.md#l134) | Gamma period / cycle duration | 7 / 33 | Prefer T_gamma (ms) with T_gamma = 1000/f_gamma when frequency is Hz; distinguish inferred period from detected individual cycle lengths. |
| [L135](evidence.md#l135) | Cycle participation | 6 / 31 | Use p_part for measured active neuron–cycle fraction and beta_rf for fitted rate–frequency slope; do not identify them without the model assumptions. |
| [L136](evidence.md#l136) | Spikes per cycle / active pair | 7 / 24 | Define the neuron–cycle observation, boundary rule and treatment of partial cycles; pooled pairs are not independent network replicates. |
| [L137](evidence.md#l137) | Inhibitory burst / volley / cycle anchor | 7 / 27 | Use inhibitory population volley for a detected multi-neuron event; state smoothing, threshold and separation. Keep single-neuron bursting distinct. |
| [L138](evidence.md#l138) | Excitatory volley | 4 / 12 | Use excitatory population volley and document detection rule. E-volley phase in coupled networks is not automatically I-midpoint cycle phase. |
| [L139](evidence.md#l139) | PSD / power spectrum / Welch | 9 / 43 | Use power spectral density with normalization, signal and one-/two-sided convention; distinguish one full-window periodogram from multi-segment averaging. |
| [L140](evidence.md#l140) | Peak interpolation / spectral aggregation | 9 / 29 | Record interpolation rule and order of averaging/peak extraction. Peak of mean spectrum, mean/median of trial peaks and binned display peaks differ. |
| [L141](evidence.md#l141) | Autocorrelation / autocorrelogram / autocovariance | 5 / 20 | Distinguish overlap/mean-square-normalized count correlation (chance 1) from centered covariance (zero baseline) and lag-product sums. |
| [L142](evidence.md#l142) | Lobe–trough contrast / rhythmicity score | 7 / 60 | Prefer R_contrast and the precise lobe/trough estimator. This is not phase concentration, spectral frequency or a calibrated probability of PING. |
| [L143](evidence.md#l143) | Contrast lobe and trough | 6 / 66 | Use A_lobe/A_trough with selection and smoothing rules; a first pre-trough maximum is not necessarily a later recurrence side lobe. |
| [L144](evidence.md#l144) | Undefined score / missing data | 26 / 106 | Keep undefined measurements distinct from observed zero and unavailable article inputs. exp099's zero fill is an estimator policy, not the universal contrast definition. |
| [L145](evidence.md#l145) | Cross-correlation lag | 9 / 43 | Qualify signed/absolute lag, units, signal order and estimator. Lag magnitude, synaptic delay and causal direction are not interchangeable. |
| [L146](evidence.md#l146) | Relative phase / phase gap | 2 / 22 | Use Delta phi = phi_A−phi_B with units radians or cycles; document wrapping interval, sign and event anchor. |
| [L147](evidence.md#l147) | Phase concentration / locking value | 2 / 23 | Prefer R_phase (mean resultant length) and identify phase samples; do not share R with autocorrelation contrast. |
| [L148](evidence.md#l148) | Phase locking / synchronization | 9 / 34 | Separate within-population spike synchrony, between-network phase concentration and locking under drift criteria. Define each diagnostic. |
| [L149](evidence.md#l149) | Phase drift / instantaneous frequency | 2 / 18 | Use signed phase drift in cycles/s or angular velocity in rad/s, with conversion explicit; an interval frequency is an estimator. |
| [L150](evidence.md#l150) | Phase slips / net windings | 2 / 15 | Distinguish whole net windings, wrap crossings and individual slip events including reversals; retain the implemented count definition. |
| [L151](evidence.md#l151) | Detuning / mean frequency | 3 / 22 | Use Delta f_detune for uncoupled frequency difference and f_mean for mean rhythm frequency; Delta f_bin denotes spectral-bin spacing separately. |
| [L152](evidence.md#l152) | Cross-network / effective interaction strength | 3 / 22 | Keep summed conductance K (μS), leak-normalized kappa (dimensionless) and fitted phase interaction epsilon (Hz) as three separate quantities. |
| [L153](evidence.md#l153) | Phase response / advance / delay | 5 / 33 | Define measured timing response and its sign; a millisecond next-volley advance is not automatically a dimensionless infinitesimal phase-response curve. |
| [L155](evidence.md#l155) | Jitter / cell-wise versus coherent | 11 / 110 | Use independent per-spike jitter and cycle-coherent volley jitter, with sigma_jitter in ms. Current “per-cell” wording can hide that offsets are per spike. |
| [L157](evidence.md#l157) | Spike deletion / insertion | 6 / 57 | Use deletion probability p_drop and nominal addition rate r_add; record target, schedule, binary collisions and whether membrane reset is unchanged. |
| [L158](evidence.md#l158) | Matched control / rate matching | 6 / 22 | State the matched quantity, tolerance and actual measured values. Matched I spike rate is not automatically matched conductance/current. |
| [L159](evidence.md#l159) | Standard deviation / SD | 18 / 43 | Use sample standard deviation with aggregation unit and denominator; do not conflate variation across trials, neurons and trained seeds. |
| [L160](evidence.md#l160) | Standard error / SEM | 9 / 19 | Use standard error of the mean across specified independent replicates; keep distinct from SD, min–max shading and confidence intervals. |
| [L161](evidence.md#l161) | Confidence interval / range envelope | 8 / 14 | Name uncertainty display exactly: SD, SEM, min–max range or CI, including estimator and replicate count. |
| [L162](evidence.md#l162) | Pooling / averaging / reduction order | 18 / 51 | Specify pooling and averaging axes/order; do not give independent-replicate status to pooled neuron–cycle pairs or copied figures. |
| [L163](evidence.md#l163) | Affine fit / slope / intercept | 3 / 30 | Use a_rf and beta_rf for affine rate–frequency fit and beta_rf,0 for through-origin fit; identify data points and weighting. |
| [L164](evidence.md#l164) | Coefficient of determination | 5 / 17 | Reserve R² for coefficient of determination with its total-sum-of-squares convention; distinguish from either R_contrast or R_phase. |
| [L165](evidence.md#l165) | Percentage / percentage point | 8 / 32 | Keep accuracy fraction, accuracy percent and percentage-point difference as separate representations; label source fields and display conversions. |
| [L166](evidence.md#l166) | Illustrative probe / population estimate | 21 / 67 | Label selected illustrative observations separately from quantitative population estimates and independent replication. |
| [L167](evidence.md#l167) | Causal effect / association / mechanism | 20 / 70 | Use explicit evidence labels: observed association, controlled effect, proposed mechanism and untested interpretation. A lexicon must not upgrade claim strength. |

## Dynamical systems and filters

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L168](evidence.md#l168) | Fixed point / equilibrium / stationary state | 7 / 35 | Distinguish equilibrium of a dynamical system, stationary distribution, held-conductance equilibrium and late-time measurement window. |
| [L169](evidence.md#l169) | Silent / non-oscillating | 13 / 32 | Reserve silent for no spiking/zero rate under a stated window. Non-oscillating and stable equilibrium need not be silent. |
| [L170](evidence.md#l170) | Hopf onset / empirical threshold | 6 / 122 | Separate eigenvalue-defined Hopf threshold from empirical grid crossing and recruitment marker; do not identify them without calibration. |
| [L171](evidence.md#l171) | Supercritical / subcritical / reversibility | 4 / 29 | Keep nonlinear criticality, reversible sampled ramps and evidence consistent with a criticality class distinct; define the actual criterion. |
| [L172](evidence.md#l172) | Jacobian / eigenvalue / stability | 4 / 28 | Use J_flow for continuous vector-field derivative and J_step for discrete state-map derivative; eigenvalue units and stability tests differ. |
| [L173](evidence.md#l173) | Angular frequency / onset frequency | 6 / 69 | Use omega_H (rad/ms) and f_H (Hz), with factor 1000/(2π); distinguish eigenfrequency at onset from finite-drive spectral peaks. |
| [L174](evidence.md#l174) | Amplitude / oscillation power | 5 / 44 | Prefer A_pp for peak-to-peak rate amplitude with units and window; it is not mean rate, autocorrelation A, accuracy A or power. |
| [L175](evidence.md#l175) | Hysteresis / attractor / basin | 6 / 27 | Give an operational ramp/initial-condition test; endpoint agreement and low rate alone do not establish absence of other attractors. |
| [L176](evidence.md#l176) | QSS / reduction / closure | 4 / 74 | Name quasi-steady-state substitution, mean-field closure and centre-manifold reduction separately; they are not the same reduction claim. |
| [L177](evidence.md#l177) | Gain / f–I curve / Siegert | 3 / 41 | Use Phi for current-to-rate gain with input/output units; a response versus Poisson input rate is not literally an injected-current f–I curve. |
| [L178](evidence.md#l178) | Effective voltage noise | 3 / 19 | Use sigma_V for the gain model's effective voltage-noise scale; it is not necessarily measured voltage SD or temporal jitter sigma. |
| [L179](evidence.md#l179) | Mean / fluctuation / expectation notation | 12 / 34 | State whether a bar means sample average or equilibrium evaluated at a mean. E[V(g)] and V(E[g]) are different quantities. |
| [L180](evidence.md#l180) | Frequency response / transfer function | 2 / 17 | Name synaptic, membrane and finite-window stages separately; G_r/H_r/A_T denote distinct transfers, not coupling or graph G. |
| [L181](evidence.md#l181) | Finite-window / stationary approximation | 4 / 60 | Keep finite-duration start-from-rest simulation and stationary small-fluctuation approximation distinct in names and provenance. |
| [L182](evidence.md#l182) | Fourier / FFT / spectrum convention | 10 / 28 | Record time base, normalization and one-/two-sided density convention; never compare amplitudes from differently normalized spectra as identical. |
| [L183](evidence.md#l183) | Numerical tolerances / exactness | 25 / 109 | Qualify algebraic exactness, floating equality, tolerance-based conformance and empirical equivalence. They are different claims. |
| [L232](evidence.md#l232) | Phase portrait / projection / trajectory | 17 / 55 | Distinguish state-space trajectory, projection onto coordinates and training-metric trajectory. A closed projected curve does not establish an autonomous two-variable system. |
| [L233](evidence.md#l233) | Stability / invariance / convergence | 16 / 55 | Name numerical stability, local equilibrium stability, gradient behavior and empirical timestep sensitivity separately; no one “stable” label certifies all. |

## Software vocabulary

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L184](evidence.md#l184) | SNNLANG / snnlang | 13 / 30 | Use SNNLANG as the project/language display name and `snnlang` or `tools.snnlang` for package identity; authoring is separate from execution. |
| [L185](evidence.md#l185) | SNNSIM / tools.snnsim | 13 / 73 | Use SNNSIM for simulator/execution tools, with exact module paths in API references; it is not the experiment orchestrator or storage system. |
| [L186](evidence.md#l186) | Graph / Network / Bundle | 39 / 484 | Preserve three objects: mutable authoring Network, immutable graph data, portable Bundle. Avoid using graph and bundle as synonyms. |
| [L187](evidence.md#l187) | Component / group | 7 / 49 | Distinguish reusable graph-building component, structural group and optimizer parameter group; their ownership and execution effects differ. |
| [L188](evidence.md#l188) | Signal / port / target | 27 / 156 | Qualify graph signal, population target port, training target label and compiler target backend. Bare target currently covers unrelated roles. |
| [L189](evidence.md#l189) | Projection / connection / synapse | 31 / 142 | Projection = graph edge with source/target/filter/weight; connection role = scheduling classification; synapse = event-to-conductance filter. |
| [L190](evidence.md#l190) | Parameter / constant / state | 33 / 246 | Separate learned/frozen parameter, declared constant and evolving state; freezing parameters does not freeze trajectory state. |
| [L191](evidence.md#l191) | Tensor shape / axis contract | 23 / 91 | Adopt an explicit axis glossary: time, batch/sample, channel/neuron, class, source, target. Preserve graph-to-runtime transformations and dtype. |
| [L192](evidence.md#l192) | Training recipe / TrainSpec | 3 / 10 | Use training recipe for declarative objectives/groups/optimizer/backward choices; do not mix with complete experiment recipe or execution request. |
| [L193](evidence.md#l193) | Execution protocol / request / ExecutionSpec | 7 / 26 | Separate typed request from resolved protocol record and graph plan. Record data/order/encoder seeds and actual resolved values. |
| [L194](evidence.md#l194) | Executor / backend / device / provider | 12 / 95 | Namespace executor (legacy/graph), implementation backend, device (CPU/CUDA/MPS) and compute provider (local/cluster/cloud); not interchangeable. |
| [L195](evidence.md#l195) | Compiler / compilation / graph plan | 11 / 61 | Distinguish SNNLANG graph compilation, graph execution planning and torch/Inductor compilation; “compiled runtime” must identify which. |
| [L196](evidence.md#l196) | Capability / validation / conformance gate | 30 / 224 | Separate schema validity, supported execution capability, numerical conformance and scientific acceptance. A passing plumbing gate is not a scientific validation. |
| [L197](evidence.md#l197) | Runtime state / simulation continuation | 6 / 26 | Runtime state = dynamic trajectory values/history and step coordinate; distinguish parameter checkpoints and training-resume state. |
| [L198](evidence.md#l198) | Checkpoint replay / training resume | 5 / 25 | Use inference parameter load, simulation continuation and exact training resume as separate operations with separate compatibility requirements. |
| [L199](evidence.md#l199) | Binding / replay / generated input | 10 / 46 | Define separate dense, event, generated-Poisson and dataset-snapshot bindings; distinguish exact replay from seeded fresh generation. |
| [L200](evidence.md#l200) | Valid-time mask | 2 / 5 | Use valid-time mask (time,batch) for duration/reduction validity; not a connectivity mask, spike-deletion mask or initial-zero mask. |
| [L201](evidence.md#l201) | Output / recording profile | 5 / 13 | Keep profile values full/observables/none exact and distinct from article input availability or scientific output selection. |
| [L202](evidence.md#l202) | Manifest / digest / hash / signature | 13 / 87 | Qualify graph digest, training digest, payload checksum, manifest digest and runtime compatibility signature; no generic hash substitutes for all identities. |
| [L203](evidence.md#l203) | Stable identifier / name / identity | 5 / 11 | Define ID namespaces and scope. A display label, filename, graph ID, run ID and content identity are different entities. |
| [L204](evidence.md#l204) | Determinism / parity / equivalence | 10 / 62 | State tested layer, hardware, stochastic stream and tolerance. Exact local parity is not global backend or campaign equivalence. |
| [L205](evidence.md#l205) | Legacy adapter / compatibility / migration | 17 / 99 | Mark historical adapters and current supported paths separately. Lexical consistency must not reactivate retired storage or imply migration readiness. |
| [L206](evidence.md#l206) | Disabled / unsupported / not implemented | 14 / 28 | Use time-scoped implementation status separate from graph enablement and experiment completion; conflicting status prose needs code verification, not renaming. |
| [L207](evidence.md#l207) | API / CLI / Python interface | 14 / 53 | Use API for programmatic interface and CLI for commands; neither term implies a files-only execution boundary. |
| [L235](evidence.md#l235) | Event collision / binary union / boundary clamping | 6 / 10 | Name event union and boundary clamping explicitly; they can reduce realized counts even when the intended perturbation only moves/adds spikes. |

## Execution and provenance

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L208](evidence.md#l208) | Experiment / article / run | 46 / 637 | Experiment = scientific design, article = written account, run = completed stage execution. An article can consume several runs; a run is not one neuron or one condition by definition. |
| [L209](evidence.md#l209) | Compute / analyse / present stages | 5 / 25 | Use exact stage IDs compute/analyse/present in operational records; analysis and presentation are prose nouns. Keep simulation/training execution kinds separate. |
| [L210](evidence.md#l210) | Campaign / bank / family | 8 / 97 | Distinguish orchestration campaign, retained model bank, scientific condition family and stage run; none is a synonym for collection. |
| [L211](evidence.md#l211) | Collection / catalogue / inventory | 44 / 83 | Qualify article collection, read-only discovery/presentation projection and payload inventory. Do not introduce an operational Pingstore catalogue. |
| [L212](evidence.md#l212) | Run record / payload / export | 13 / 64 | Keep authoritative run record, exported stage outputs and retained execution evidence distinct; source prose is not verified execution evidence. |
| [L213](evidence.md#l213) | Artifact / asset / dataset | 44 / 287 | Qualify scientific dataset, article-selected presentation input, generated output artifact and logical bundle asset. The Datasets UI labels selected runs, not necessarily raw datasets. |
| [L214](evidence.md#l214) | Snapshot / archive / backup / restore | 10 / 73 | Qualify initialization snapshot, dataset snapshot, R2 payload snapshot and complete run backup. Copy verification is not complete provenance recovery. |
| [L215](evidence.md#l215) | Immutable / completed / temporary | 7 / 20 | Keep completed validated run separate from hidden temporary output, successful process exit and immutable payload copy; do not infer status from names. |
| [L216](evidence.md#l216) | Selection / pin / publication | 35 / 116 | Qualify model checkpoint selection, article present-run selection and publication. These are distinct decisions; no selection is authorized by this audit. |
| [L217](evidence.md#l217) | Data availability / status badge | 43 / 231 | Preserve exact badges and availability meaning; they do not mean scientific completion, reviewed claims or complete article input coverage. |
| [L218](evidence.md#l218) | Scientific duration / elapsed runtime | 9 / 50 | Distinguish physical simulated duration, elapsed stage wall time, execution span including gaps and summed job time; report import time separately. |
| [L219](evidence.md#l219) | Execution origin / compute resources | 15 / 140 | Keep origin/provider, scheduler, device and executor names in their own fields; avoid deriving provenance from a run identifier. |
| [L220](evidence.md#l220) | Reused / historical / new evidence | 24 / 70 | Separate inherited observation, new analysis and new simulation. A newly rendered figure is not new experimental evidence. |

## Typst source conventions

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L221](evidence.md#l221) | Shared data and report helpers | 45 / 1081 | Reuse existing helper names; distinguish input resolver, report renderer, unavailable-input view, dataset subview and TOC wrapper. Do not globally rename locally scoped bindings. |
| [L222](evidence.md#l222) | Loaded report aliases | 24 / 61 | Prefer report/config/parameters or experiment-qualified report names in future code, while allowing small local r/c aliases. These are Typst bindings, not mathematical variables. |
| [L223](evidence.md#l223) | Numeric display helpers | 14 / 20 | Separate formatting from numerical aggregation; pct currently has different input scales. A shared helper needs an explicit fraction/percent contract first. |
| [L224](evidence.md#l224) | Document metadata / authored dates | 44 / 240 | Use the Writing Guide's exact metadata vocabulary; authored creation/update dates are not run dates or file timestamps. |
| [L225](evidence.md#l225) | Figure / schematic / measurement | 28 / 252 | Name model schematic, illustrative recording and measured result distinctly; a shared caption must preserve the source estimator and evidence limitations. |
| [L226](evidence.md#l226) | Prose spelling variants | 40 / 338 | Choose one house spelling for prose, preserve exact API identifiers and quoted titles; prefer damping as the common scientific noun while retaining dampen API tokens. |
| [L237](evidence.md#l237) | Bibliography / citation / reference | 22 / 70 | Use a single bibliographic identity per cited work while preserving source titles. Repeated reference-title vocabulary alone does not justify a scientific lexicon entry. |

## Units and mathematical conventions

| Entry | Term or quantity | Files / matches | Proposed standard and boundary |
| --- | --- | ---: | --- |
| [L227](evidence.md#l227) | Physical unit system | 23 / 294 | Give each quantity a unit and conversion boundary: biophysical ms/mV/nF/μS/nA, graph-declared units, rates Hz and dimensionless output states are not one uniform namespace. |
| [L228](evidence.md#l228) | Physical versus dimensionless voltage | 10 / 26 | Use a unit-bearing V for membrane voltage and a qualified dimensionless output state when applicable. Never add mV to a legacy output score by analogy. |
| [L229](evidence.md#l229) | Probability / indicator | 24 / 67 | Use one indicator notation and distinguish event probability, predicted softmax share and empirical fraction. The same numeric interval does not imply the same meaning. |
| [L230](evidence.md#l230) | Normal / Gaussian distribution | 14 / 27 | Distinguish parent normal, lower-clamped normal, signed normal and timing-noise distributions; document mean/variance convention and dimensions. |
| [L231](evidence.md#l231) | Mean and sample variance | 18 / 67 | Name mean, sample SD, variance and validation-error metric separately. Avoid generic “error” for variability, residual, uncertainty or implementation mismatch. |

## Supplementary local or reference-only candidates

These were considered during the audit but did not meet the cross-file substantive-use rule. Keep them scoped locally unless future writing reuses them; zero-hit proposals are explicitly visible here.

| Entry | Term | Files / matches | Recommendation |
| --- | --- | ---: | --- |
| [L064](evidence.md#l064) | Disabled projection / ablation | 1 / 3 | Use disabled projection for structural identity retained with zero execution contribution; distinguish zero initialization and removing a population. |
| [L154](evidence.md#l154) | Intermittent phase attraction | 1 / 10 | Use the operational combination of continued slipping, nonuniform phase density and phase-dependent slowing; distinguish qualitative demonstration from repeatability. |
| [L156](evidence.md#l156) | Phase shuffle / rate-matched redraw | 1 / 5 | Keep shared time permutation and independent Poisson redraw distinct; document which count and co-firing properties are preserved. |
