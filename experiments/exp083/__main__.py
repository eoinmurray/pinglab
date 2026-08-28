"""Require an explicit stage; never dispatch upstream work implicitly."""

raise SystemExit(
    "Choose an exp083 stage: compute, analyse --source <compute-run-id>, "
    "or present --source <analyse-run-id>. Nothing is published automatically."
)
