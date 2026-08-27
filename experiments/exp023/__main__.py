"""The combined runner is retired; fail before creating any output."""

raise SystemExit(
    "exp023 has independent stages: experiments/exp023/compute.py; "
    "experiments/exp023/analyse.py --source <compute-run>; "
    "experiments/exp023/present.py --source <analyse-run>. "
    "No combined execution or automatic publication."
)
