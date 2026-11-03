# ELEVATE TALK

Work Package 3 asks whether controlled spiking states—rest, tonic firing, bursting, and gamma rhythms—can do useful computation.

=============

Daria (3 slides) has constructed neurons with rest–spike bistability with two timescales and used it to build spike-efficient working memory controlled one parameter, w.

She is testing winner-take-all networks, time-series tasks, spatial working memory, and navigation. And reproducing attention-like selection, forgetting, and memory reset.

Next, she will add the three-timescale bursting regime, where bursting stabilizes weights updated during tonic firing, then move toward online learning, and a trip to work with me in Cambridge.

=============

Dziyana (1 slide) is developing event-based computation with excitatory–inhibitory networks.

She is defining the project direction and building core skills, and has implemented a Hodgkin–Huxley neuron with fast and slow transient conductances.

Next month, she will visit KU Leuven and use that exchange to set the project objectives.

=============

Giada (1 slide) is coordinating model, biology, and hardware.

She has combined Guillaume Drion’s architecture with a two-stage plasticity rule for learning and memory consolidation, controlled by a parameter, w, and is now scaling the network to test stability and failure modes.

A new primary-neuron culture on a chip has just started and will mature for about two weeks before stimulation.

Hardware remains on hold until the model and in vitro results are validated.

=============

Julia (1 slide) is connecting the work to behaviour and action selection.

She is studying basal-ganglia–cortical loops, cortical layer circuits, and cortical–subcortical connections. While using hierarchical reinforcement learning as a starting point and training networks previously used in robotics.

Her next step is to continue those experiments and define a focused project plan.

=============

Qi (4 slides) is working on silicon neurons. She is using circuit simulations and Kirchhoff’s current law to understand how an existing circuit produces bursts.

And she is studying "NeoCorAL" to see how bursts should drive synaptic updates.

Next, she will identify the signals a burst circuit must provide to a learning rule and compare an alternative burst-generating design.

=============

My work (2 slides) tests whether gamma-generating E/I circuits are computationally useful.

I trained Gamma networks on MNIST. They can solve tasks with sparse, rhythm-anchored activity, but trainable recurrence often weakened rhythmicity and raised firing. We recently demonstrated streaming digit classification where the PING reset mechanism helps the network clear previous state faster so its ready for the next digit.

The key question now is when task pressure preserves gamma, and whether disrupting its timing removes a computational advantage.

Also announcing snnlab!

=============

Thank you for listening.
