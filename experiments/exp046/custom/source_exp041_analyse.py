"""Measure retained inhibitory-decay sweep evidence; never run inference or plot.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

MEASUREMENT = {
    "schema": "exp041.measurement/v1",
    "spectrum": "Welch per trial; demean; full-trial segment; density scaling; average PSD across trials",
    "gamma_peak": "5-150 Hz argmax of mean PSD with parabolic offset clamped to +/- half a bin",
    "fit_points": "six tau_GABA means across seeds; equal weight per condition",
    "fit_models": ["affine", "through_origin"],
    "r_squared": "centred total sum of squares for both fits",
    "uncertainty": "seed SEM (ddof=1)",
    "history_partition": "validation",
    "trial_peak_histogram": "diagnostic only; pooled across seeds; 1 Hz bins",
}
