"""Reject the retired combined entrypoint without creating output."""

raise SystemExit(
    "exp086 requires an explicit stage: compute, analyse --source, or present --source"
)
