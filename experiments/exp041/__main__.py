"""Reject retired combined execution before creating any output."""

raise SystemExit(
    "exp041 has independent stages: experiments/exp041/compute.py --source <exp022-bank>; "
    "experiments/exp041/analyse.py --source <compute-run>; "
    "experiments/exp041/present.py --source <analyse-run>. "
    "No combined execution or automatic publication."
)
