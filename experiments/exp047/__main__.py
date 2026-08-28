"""Retire the combined simulation/publication runner without creating output."""

raise SystemExit(
    "exp047 has independent stages: experiments/exp047/compute.py; "
    "experiments/exp047/analyse.py --source <compute-run>; "
    "experiments/exp047/present.py --source <analyse-run>. "
    "No combined execution or automatic publication."
)
