"""Reject combined execution without dispatching any stage."""

if __name__ == "__main__":
    raise SystemExit(
        "exp099 requires independent stages: run experiments/exp099/compute.py, then experiments/exp099/analyse.py --source <compute-run-id>, then experiments/exp099/present.py --source <analyse-run-id>. No automatic execution or publication."
    )
