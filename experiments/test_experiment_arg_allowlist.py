"""Gate: experiment runners expose only META flags — never science parameters.

The demolab invariant is "the runner is the recipe": a committed experiment must
run the same way every time, so its physics/training parameters (weights, rates,
τ_GABA, lr, epochs, seeds, dt, …) are hardcoded literals in the runner, not
overridable CLI flags. Only a small set of *meta* flags — which don't change the
science, just how/whether it runs or re-renders — may be accepted.

This scans each explicit compute, analyse and present module for the flags it accepts (argparse
add_argument + bare `"--flag" in sys.argv` checks) and fails on anything outside
the meta allowlist. A new science flag lights up here until it is either hardcoded
or (if genuinely meta) added to ALLOWED with justification.

Scope note: helper modules are excluded.
"""

import re
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent

from experiments.helpers.cli import ALL_META_FLAGS, parse_meta

# Synced with helpers/cli.py — the closed meta vocabulary (+ legacy wipe/replot).
ALLOWED_EXACT = set(ALL_META_FLAGS) | {"--no-wipe-dir", "--wipe-dir", "--replot"}

# Exp022's bank-array workflow controls one explicitly reserved compute run.
# Keep the exception local so ordinary experiment runners cannot acquire
# per-cell worker controls accidentally.
EXP022_BANK_META = {
    "--bank",
    "--bank-create",
    "--bank-finalize",
    "--bank-list",
    "--bank-status",
    "--bank-train-cell",
    "--bank-validate",
    "--execution-origin",
    "--recover-stale",
    "--retry-only",
    "--tier",
}

# Canonical staged runners only; helper and dispatch modules are skipped.
RUNNERS = sorted(
    path for path in EXPERIMENTS.glob("exp[0-9][0-9][0-9]/*.py")
    if path.stem in {"compute", "analyse", "present"}
)
STAGE_META = {
    "--source",
    "--frequency-source",  # explicit second analysis input
    "--run-id",
    "--import-source",
    "--diagnostics",
    "--retained-presentation",
    "--shard-index",  # scheduler-owned partition of a committed recipe
}

REMOVED_RUNNER_FLAGS = {
    "exp022/compute.py": {"--hpc", "--hpc-root", "--plumbing"},
    "exp023/present.py": {"--metadata-source"},
    "exp054/analyse.py": {"--theory-source"},
    "exp082/analyse.py": {"--showcase-source"},
    "exp099/compute.py": {
        "--baseline-hz",
        "--capacitance-nf",
        "--inhibitory-scale",
        "--leak-us",
        "--no-i-external",
        "--recurrent-scale",
        "--seed",
    },
    "exp099/present.py": {"--view-end-ms", "--view-start-ms"},
    "exp110/present.py": {
        "--exp037-source",
        "--exp041-source",
        "--exp044-source",
        "--exp046-source",
    },
    "exp037/compute.py": {"--collect"},
    "exp042/compute.py": {"--collect"},
    "exp082/compute.py": {"--collect"},
}


def _accepted_flags(src: str) -> set[str]:
    argparse_flags = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    argv_flags = set(re.findall(r'"(--[a-z0-9-]+)"\s+(?:not\s+)?in\s+sys\.argv', src))
    return argparse_flags | argv_flags


def test_runners_exist():
    # Guards against a glob/path mistake silently passing the gate on zero files.
    assert RUNNERS, f"no exp<NNN>.py runners found under {EXPERIMENTS}"


def test_retired_modal_flag_is_rejected():
    with pytest.raises(SystemExit, match="unknown flag '--modal'"):
        parse_meta(["compute.py", "--modal"])


@pytest.mark.parametrize("relative, flags", REMOVED_RUNNER_FLAGS.items())
def test_removed_runner_flags_stay_removed(relative, flags):
    accepted = _accepted_flags((EXPERIMENTS / relative).read_text())
    assert accepted.isdisjoint(flags)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_runner_accepts_only_meta_flags(runner):
    allowed = ALLOWED_EXACT | (
        EXP022_BANK_META if runner.parent.name == "exp022" else set()
    )
    if runner.parent != EXPERIMENTS:
        allowed |= STAGE_META
    offenders = sorted(
        f for f in _accepted_flags(runner.read_text()) if f not in allowed
    )
    assert not offenders, (
        f"{runner.name} exposes non-meta CLI flag(s) {offenders} — the runner is the "
        f"recipe, so science parameters must be hardcoded, not accepted as flags. "
        f"If a flag is genuinely meta, add it to ALLOWED in {Path(__file__).name}."
    )
