"""Retired combined runner; execute the independent exp081 stages."""

if __name__ == "__main__":
    raise SystemExit(
        "exp081 requires independent stages: run experiments/exp081/compute.py, then experiments/exp081/analyse.py --source <compute-run-id>, then experiments/exp081/present.py --source <analyse-run-id>. No automatic execution or publication."
    )
