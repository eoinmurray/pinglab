"""Reject retired combined execution before creating any output."""

raise SystemExit(
    "exp044 has independent stages: experiments/exp044/compute.py --source <exp022-bank>; "
    "experiments/exp044/analyse.py --source <compute-run>; "
    "experiments/exp044/present.py --source <analyse-run>. "
    "No combined execution or automatic publication."
)
