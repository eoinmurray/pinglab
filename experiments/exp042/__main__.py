"""Reject retired combined execution before creating any output."""

raise SystemExit(
    "exp042 has independent stages: experiments/exp042/compute.py --source <exp022-bank>; "
    "experiments/exp042/analyse.py --source <compute-run>; "
    "experiments/exp042/present.py --source <analyse-run>. "
    "No combined execution or automatic publication."
)
