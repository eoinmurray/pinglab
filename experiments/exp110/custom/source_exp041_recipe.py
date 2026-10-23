"""The retained inhibitory-decay sweep recipe; training remains owned by exp022.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

TAU_GABA_REFERENCE_MS = 6.0
