"""Reject the retired combined runner without creating any run or output."""

raise SystemExit(
    "exp024 now has independent stages: run experiments/exp024/analyse.py "
    "--source <exp022-compute-run>, then experiments/exp024/present.py "
    "--source <exp024-analyse-run>. No training or automatic publication."
)
