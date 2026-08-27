"""Retired combined runner; use the independent exp024 stage commands."""

if __name__ == "__main__":
    raise SystemExit(
        "exp024 now has independent stages: run experiments/exp024/analyse.py "
        "--source <exp022-compute-run>, then experiments/exp024/present.py "
        "--source <exp024-analyse-run>. No automatic publication."
    )
