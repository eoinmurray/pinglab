"""Reject combined execution; stages must be invoked independently."""

raise SystemExit(
    "exp080 requires independent stages: experiments.exp080.compute; experiments.exp080.analyse --source <compute-run-id>; experiments.exp080.present --source <analyse-run-id>. No automatic execution or publication."
)
