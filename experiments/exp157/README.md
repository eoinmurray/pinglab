# Mean-field gamma competition — Figure 1

E-only inputs drive A, B and C; A and B feed forward to C. Bring autonomous
frequencies to A/C/B = 34/36/38 Hz by tuning I response times only. Inherit all
weights, E response times and input schedules from an explicit nine-window
compute reference. No firing-rate retuning.

1. `uv run python experiments/exp157/compute.py --reference COMPUTE_ID` — optional
   `--run-id`. Calibrates I response times, then generates the fixed-control
   comparison and native validation evidence.
2. `uv run python experiments/exp157/analyse.py --source COMPUTE_ID` — optional
   `--run-id`; measures rhythms and rate ratios without enforcing old targets.
3. `uv run python experiments/exp157/present.py --source ANALYSE_ID` — optional
   `--run-id`; renders Figure 1 and a laid-out, weight-labelled circuit diagram
   from the saved compute bundle using SNNLab's visualization API.
4. `uv run pytest experiments/exp157/test.py -q` — scientific validation.
