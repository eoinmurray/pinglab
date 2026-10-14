"""Scientific definitions for the Scikit Digits LIF baseline."""
SLUG = "exp125"
CONFIG = {
    "schema": "exp125.recipe/v1", "seed": 42, "inputs": 64, "hidden": 128,
    "outputs": 10, "epochs": 50, "batch_size": 64, "evaluation_batch_size": 256,
    "steps": 50, "dt_ms": 1.0, "peak_input_rate_hz": 250.0,
    "beta": 0.9, "threshold": 1.0, "surrogate_slope": 25.0,
    "learning_rate": 0.001, "weight_decay": 0.0, "gradient_clip": 1.0,
    "test_fraction": 0.2, "validation_fraction_of_remainder": 0.25,
    "selection": "minimum validation cross-entropy, earliest epoch on ties",
}
