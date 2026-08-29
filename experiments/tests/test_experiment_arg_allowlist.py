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

Scope note: helper and dispatch modules are excluded.
"""

import re
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]

from experiments.helpers.cli import ALL_META_FLAGS

# Synced with helpers/cli.py — the closed meta vocabulary (+ legacy wipe/replot).
ALLOWED_EXACT = set(ALL_META_FLAGS) | {"--no-wipe-dir", "--wipe-dir", "--replot"}

# exp022 is also the collection's scheduler-facing checkpoint registry. These
# flags select committed cells, lifecycle actions, or output formatting; none
# overrides a scientific parameter. Keep the exception local so ordinary
# experiment runners cannot acquire campaign controls accidentally.
EXP022_CAMPAIGN_META = {
    "--campaign",
    "--campaign-aggregate",
    "--campaign-id",
    "--execution-origin",
    "--campaign-import-compatible",
    "--campaign-list",
    "--campaign-manifest",
    "--campaign-status",
    "--campaign-train-cell",
    "--campaign-validate",
    "--json",
    "--from-campaign",
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


def _is_meta(flag: str) -> bool:
    return flag in ALLOWED_EXACT


def _accepted_flags(src: str) -> set[str]:
    argparse_flags = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    argv_flags = set(re.findall(r'"(--[a-z0-9-]+)"\s+(?:not\s+)?in\s+sys\.argv', src))
    return argparse_flags | argv_flags


def test_runners_exist():
    # Guards against a glob/path mistake silently passing the gate on zero files.
    assert RUNNERS, f"no exp<NNN>.py runners found under {EXPERIMENTS}"


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_runner_accepts_only_meta_flags(runner):
    allowed = ALLOWED_EXACT | (
        EXP022_CAMPAIGN_META if runner.parent.name == "exp022" else set()
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
