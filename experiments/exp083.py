"""Issue #72 scientific-contract audit for the gamma-gated campaign.

This runner performs no simulation or training. It converts a previously captured
production dry run into sanitized evidence, inspects the frozen source contract,
and emits the structured audit record consumed by writings/exp083.typ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
AUDITED_COMMIT = "339c164c3e553b5888e36be30a539ff566cadc0d"
SLUG = "exp083"
OUT = REPO / "artifacts" / "data" / SLUG

EXPERIMENTS = (
    "exp022", "exp023", "exp047", "exp080", "exp081", "exp024", "exp025",
    "exp037", "exp038", "exp041", "exp044", "exp049", "exp082", "exp033",
    "exp042", "exp046", "exp054",
)

FOLLOW_UPS = {
    "F01": 74,
    "F02": 75,
    "F03": 76,
    "F04": 77,
    "F05": 78,
    "F08": 79,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise ValueError(f"missing audit anchor {needle!r} in {path}")


def code_ref(relative: str, needle: str, symbol: str) -> dict[str, Any]:
    path = REPO / relative
    return {"path": relative, "line": line_of(path, needle), "symbol": symbol}


def sanitize(value: Any, capture_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, capture_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, capture_root) for item in value]
    if not isinstance(value, str):
        return value

    substitutions = (
        (str(capture_root / "campaign"), "/audit/campaigns/issue72-production"),
        (str(capture_root / "resources.json"), "/audit/private/resources.json"),
        (str(capture_root), "/audit/capture"),
        (str(REPO), "/audit/repository"),
        ("/home/eoin/.local/bin/uv", "/audit/bin/uv"),
        ("/absolute/path/to/persistent/torch-data", "/audit/data/mnist"),
        ("SL2_PROJECT_FROM_MYBALANCE", "<SLURM_ACCOUNT>"),
    )
    result = value
    for old, new in substitutions:
        result = result.replace(old, new)
    return result


def fail_closed(payload: Any) -> None:
    text = json.dumps(payload, sort_keys=True)
    forbidden = (
        "/home/", "/tmp/", "PRIVATE KEY", "BEGIN OPENSSH", "ghp_", "github_pat_",
        "RUNPOD_API", "MODAL_TOKEN", "CUDA_VISIBLE_DEVICES=",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise RuntimeError(f"sanitization failed closed; forbidden tokens: {present}")
    if re.search(r"(?:^|[^0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?:[^0-9]|$)", text):
        raise RuntimeError("sanitization failed closed; possible IP address")


def load_capture(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = json.loads((root / "campaign" / "collection-plan.json").read_text())
    manifest = json.loads((root / "campaign" / "exp022" / "campaign.json").read_text())
    submission = json.loads((root / "submission.json").read_text())
    if manifest["repository"]["commit"] != AUDITED_COMMIT:
        raise RuntimeError("captured manifest is not from the registered audited commit")
    if manifest["repository"]["dirty"] is not False:
        raise RuntimeError("captured manifest was not produced from a clean checkout")
    if submission["mode"] != "dry-run":
        raise RuntimeError("capture is not a dry-run submission")
    if len(manifest["cells"]) != 90 or len(submission["jobs"]) != 23:
        raise RuntimeError("captured campaign shape differs from the registered plan")
    return plan, manifest, submission


def argument_values(manifest: dict[str, Any]) -> dict[str, list[Any]]:
    values: dict[str, set[str]] = defaultdict(set)
    for cell in manifest["cells"]:
        for flag, value in cell["parameters"]["arguments"].items():
            values[flag].add(json.dumps(value, sort_keys=True))
    return {
        flag: [json.loads(value) for value in sorted(unique)]
        for flag, unique in sorted(values.items())
    }


def inventory() -> list[dict[str, Any]]:
    from experiments.collections.gamma_gated_sparsity.graph import EXPERIMENTS as graph
    from experiments.collections.gamma_gated_sparsity.plan import RUNNER_ARGUMENTS

    rows = []
    for item in graph:
        rows.append(
            {
                "experiment": item.slug,
                "dependencies": list(item.dependencies),
                "training_run": item.training_run,
                "runner_arguments": list(RUNNER_ARGUMENTS[item.slug]),
                "source": f"experiments/{item.slug}.py",
                "writing": f"writings/{item.slug}.typ",
                "required_output": f"artifacts/data/{item.slug}/numbers.json",
            }
        )
    if tuple(row["experiment"] for row in rows) != EXPERIMENTS:
        raise RuntimeError("collection inventory changed during audit")
    return rows


def findings() -> list[dict[str, Any]]:
    return [
        {
            "id": "F01", "severity": "blocker", "issue": FOLLOW_UPS["F01"],
            "title": "Best-accuracy checkpoints are reported as final-epoch dynamics",
            "affected": ["exp041", "exp044", "exp049"],
            "observed": "Training saves weights.pth at the best internal-holdout epoch and weights_final.pth at the last epoch. Headline downstream inference reloads weights.pth while methods or captions describe epoch-50 dynamics.",
            "expected": "Every reported quantity identifies and loads the checkpoint epoch its prose claims.",
            "risk": "Firing rates and recurrent weights can continue changing after accuracy plateaus, so the reported dynamical state can be from the wrong epoch.",
            "resolution": "Issue #74 defines and tests explicit best-versus-final checkpoint roles and downstream checkpoint provenance.",
            "evidence": [
                code_ref("tools/snn/train.py", "torch.save(best_state, out_dir / \"weights.pth\")", "train"),
                code_ref("tools/snn/train.py", "torch.save(net.state_dict(), out_dir / \"weights_final.pth\")", "train"),
                code_ref("experiments/exp041.py", '"--load-weights", str(train_dir / "weights.pth")', "_infer_cell"),
                code_ref("writings/exp041.typ", "so the fit uses the final-epoch rates", "Convergence"),
                code_ref("experiments/exp044.py", '"--load-weights", str((train_dir / "weights.pth").resolve())', "_infer_cell"),
            ],
        },
        {
            "id": "F02", "severity": "blocker", "issue": FOLLOW_UPS["F02"],
            "title": "Cell validation omits scientific manifest arguments",
            "affected": ["exp022", "TR-01", "TR-02", "TR-03", "TR-04", "TR-05", "TR-06"],
            "observed": "The manifest records --w-in and the recurrent trainability flags, but ARG_TO_CONFIG maps none of them, so validate_cell never compares them with saved config.",
            "expected": "Every scientific manifest argument is validated against its resolved saved value, or validation fails closed.",
            "risk": "A wrong input distribution or wrong TR-05 trainability intervention can be accepted into the production bank.",
            "resolution": "Issue #75 adds complete argument coverage and fail-closed tests.",
            "evidence": [
                code_ref("experiments/exp022_support/campaign.py", "ARG_TO_CONFIG = {", "ARG_TO_CONFIG"),
                code_ref("experiments/exp022_support/campaign.py", "for flag, raw in cell[\"parameters\"][\"arguments\"].items()", "_expected_config"),
                code_ref("experiments/exp022.py", '"--w-in": SHARED_W_IN_MEAN', "MODEL_RECIPES"),
                code_ref("experiments/exp022.py", 'extra.append("--trainable-w-ei")', "_init_cells"),
            ],
        },
        {
            "id": "F03", "severity": "important", "issue": FOLLOW_UPS["F03"],
            "title": "exp025 discards two registered theta-u seeds",
            "affected": ["exp025", "TR-02"],
            "observed": "TR-02 registers three seeds at every theta-u value, but exp025 returns only seed 42 for each regularized condition.",
            "expected": "The published frontier aggregates all three registered cells per condition, with representative single-seed diagnostics labeled separately.",
            "risk": "The primary frontier uses a different aggregation unit and less replication than exp022 promises.",
            "resolution": "Issue #76 updates consumers, metadata, and aggregation tests.",
            "evidence": [
                code_ref("experiments/exp022.py", "return list(SEEDS_BASELINE)", "seeds_for"),
                code_ref("experiments/exp025.py", "return list(SEEDS_BASELINE) if theta_u is None else [SEED_SWEEP]", "seeds_for"),
                code_ref("writings/exp022.typ", "Error bars at every frontier point", "TR-02 table"),
            ],
        },
        {
            "id": "F04", "severity": "important", "issue": FOLLOW_UPS["F04"],
            "title": "exp044 reports the wrong training batch size",
            "affected": ["exp044", "TR-04"],
            "observed": "TR-04 cells inherit --batch-size 256 from exp022, while exp044 records and publishes batch size 64.",
            "expected": "A consumer derives training metadata from the checkpoint manifest rather than a retired local training constant.",
            "risk": "All 15 timestep-sweep cells have a materially misstated optimization method.",
            "resolution": "Issue #77 derives and verifies exp044 training provenance from consumed cells.",
            "evidence": [
                code_ref("experiments/exp022.py", '"--batch-size": "256"', "MODEL_RECIPES"),
                code_ref("experiments/exp044.py", "BATCH_SIZE: int = 64", "BATCH_SIZE"),
                code_ref("writings/exp044.typ", "size 64 throughout", "Methods"),
            ],
        },
        {
            "id": "F05", "severity": "important", "issue": FOLLOW_UPS["F05"],
            "title": "exp023 machine-readable drive provenance contradicts execution",
            "affected": ["exp023"],
            "observed": "The runner executes synthetic-spikes at 5 Hz for COBA and 45 Hz for PING, but numbers.json is assembled with an MNIST sample label and one 50 Hz value.",
            "expected": "Saved provenance distinguishes each raster operating point and the f-I sweep grid.",
            "risk": "The displayed untrained operating conditions cannot be reconstructed from the published data contract.",
            "resolution": "Issue #78 makes saved drive provenance derive from executed arguments.",
            "evidence": [
                code_ref("experiments/exp023.py", "COBA_INPUT_RATE_HZ = 5", "COBA_INPUT_RATE_HZ"),
                code_ref("experiments/exp023.py", "PING_INPUT_RATE_HZ = 45", "PING_INPUT_RATE_HZ"),
                code_ref("experiments/exp023.py", '"input": "mnist d0 s0"', "main payload"),
                code_ref("experiments/exp023.py", '"input_rate_hz": 50', "main payload"),
            ],
        },
        {
            "id": "F06", "severity": "important", "issue": None,
            "title": "The pooled MNIST holdout is called test data without a checkpoint-selection contract",
            "affected": ["exp022", "exp024", "exp025", "exp037", "exp038", "exp041", "exp044", "exp049", "exp082"],
            "observed": "The loader concatenates the official 60,000-image training and 10,000-image test partitions, optionally subsamples, then creates a fixed stratified 80/20 split. That 20% split is evaluated every epoch and selects weights.pth.",
            "expected": "Publication text calls this an internal pooled-MNIST holdout used for checkpoint selection, not an untouched or official test set, and reports 56,000/14,000 or 5,600/1,400 train/holdout counts.",
            "risk": "Readers can interpret repeatedly consulted checkpoint-selection data as a final held-out test set and can read 70,000 or 7,000 as the number of optimizer-training images.",
            "resolution": "Documented resolution: treat the split as an internal holdout throughout this campaign, state that the official partitions were pooled, and reserve final-test language for a future untouched evaluation. No scientific rerun is implied by this naming correction.",
            "evidence": [
                code_ref("tools/snn/datasets.py", "X = np.concatenate(", "load_dataset"),
                code_ref("tools/snn/datasets.py", "return train_test_split(X, y, test_size=0.2", "load_dataset"),
                code_ref("tools/snn/train.py", "new_best = acc > best_acc", "train"),
                code_ref("writings/exp022.typ", "[Training pool], [70,000 samples]", "TR-01 table"),
            ],
        },
        {
            "id": "F07", "severity": "minor", "issue": None,
            "title": "Dry-run submission requires the configured uv executable",
            "affected": ["collection scheduler"],
            "observed": "A dry submission invokes the configured uv executable to ask exp022 for retry-only cell lists; a placeholder executable prevents plan generation even though no scheduler submission is requested.",
            "expected": "The runbook states that dry planning still requires a working local uv environment, or planning derives cell lists without launching the configured remote executable.",
            "risk": "Operational surprise only; no result or production job is changed.",
            "resolution": "Accepted limitation for this campaign and recorded in the reproducer.",
            "evidence": [code_ref("experiments/collections/gamma_gated_sparsity/slurm.py", "def _exp022_cells", "_exp022_cells")],
        },
        {
            "id": "F08", "severity": "important", "issue": FOLLOW_UPS["F08"],
            "title": "The production manifest leaves scientific defaults unresolved",
            "affected": ["exp022", "TR-01", "TR-02", "TR-03", "TR-04", "TR-05", "TR-06"],
            "observed": "Cell commands omit the fixed input rate, topology, Dale constraint, weight decay, gradient clip, and AMPA decay; those values are inherited from shared CLI or model defaults and are absent from the manifest's resolved parameter block.",
            "expected": "The pre-run manifest states every scientific setting with physical meaning and units, including inherited values.",
            "risk": "The frozen source preserves reproducibility, but the plan is not cold-readable and a shared default change can alter a regenerated campaign without a registry-level parameter diff.",
            "resolution": "Issue #79 defines a complete resolved scientific schema and cross-cell difference check.",
            "evidence": [
                code_ref("experiments/exp022.py", "def build_train_args", "build_train_args"),
                code_ref("tools/snn/train.py", "default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)", "train"),
                code_ref("tools/snn/train.py", "GRAD_CLIP = 1.0", "GRAD_CLIP"),
                code_ref("writings/exp022.typ", "[Input rate], [25 Hz maximum-pixel rate]", "TR-01 table"),
            ],
        },
    ]


def false_alarms() -> list[dict[str, Any]]:
    return [
        {"contract": "COBA/PING matched initialization", "disposition": "false alarm", "evidence": "Both recipes use the same --w-in, sparsity, readout mode, readout initializer, learning rate, batch size, duration, timestep, and seeds. Registered differences are --ei-strength and --v-grad-dampen."},
        {"contract": "Smoke/production separation", "disposition": "false alarm", "evidence": "Initialization records profile=smoke only when --smoke is supplied, uses --plumbing only for that profile, and requires a unique external campaign root."},
        {"contract": "Source and cache identity", "disposition": "false alarm", "evidence": "The collection plan binds commit and lockfile; exp022 cells bind the manifest hash; downstream numbers bind campaign, commit, lockfile, manifest, dependencies, and training-run ID."},
        {"contract": "Stochastic operations", "disposition": "false alarm", "evidence": "All reviewed runner-local NumPy and PyTorch stochastic operations use explicit constants, cell seeds, or stable derived seeds. Training seeds Python, NumPy, and PyTorch before model and loader construction."},
        {"contract": "Dataset split symmetry", "disposition": "false alarm", "evidence": "Every exp022 cell uses the same seed-42 subsample and stratified 80/20 split; COBA and PING therefore see identical image identities within a scale."},
        {"contract": "Collection output isolation", "disposition": "false alarm", "evidence": "The scheduler supplies distinct absolute state, derived, and log roots and PINGLAB_REQUIRE_ISOLATED=1; helpers reject partial or repository-artifact destinations."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    args = parser.parse_args()
    plan, manifest, submission = load_capture(args.capture_root.resolve())
    sanitized = {
        "collection_plan": sanitize(plan, args.capture_root.resolve()),
        "exp022_manifest": sanitize(manifest, args.capture_root.resolve()),
        "submission": sanitize(submission, args.capture_root.resolve()),
    }
    fail_closed(sanitized)

    rows = findings()
    tiers = Counter(cell["resource_tier"] for cell in manifest["cells"])
    training_runs = Counter(cell["training_run_id"] for cell in manifest["cells"])
    summary = {
        "status": "complete",
        "audit_issue": 72,
        "audited_commit": AUDITED_COMMIT,
        "audit_branch": "audit/issue-72-scientific-contract",
        "scope": {
            "experiments": list(EXPERIMENTS),
            "experiment_count": len(EXPERIMENTS),
            "exp022_cell_count": len(manifest["cells"]),
            "planned_job_count": len(submission["jobs"]),
            "paid_compute_jobs_created": 0,
        },
        "production_plan": {
            "profile": plan["profile"],
            "campaign_id": plan["campaign_id"],
            "tiers": dict(sorted(tiers.items())),
            "training_runs": dict(sorted(training_runs.items())),
            "job_names": [job["name"] for job in submission["jobs"]],
            "source_commit": plan["source"]["git_commit"],
            "lockfile_sha256": plan["source"]["lockfile"]["sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "resource_file_sha256": submission["resource_file_sha256"],
        },
        "resolved_argument_values": argument_values(sanitized["exp022_manifest"]),
        "inventory": inventory(),
        "findings": rows,
        "finding_counts": dict(Counter(row["severity"] for row in rows)),
        "false_alarms": false_alarms(),
        "accepted_limitations": [
            "No production checkpoints or outputs existed, so validation was assessed against manifest construction, validator behavior, focused synthetic tests, and current committed historical artifacts rather than a completed 90-cell bank.",
            "The pooled-MNIST internal holdout remains the campaign evaluation partition; the audit requires accurate naming and counts but does not redesign the registered dataset split.",
            "The dry planner requires a working uv executable even when it creates no scheduler jobs.",
        ],
        "production_readiness": {
            "decision": "not ready",
            "reason": "Two unresolved blockers can admit a scientifically wrong cell or report a checkpoint from a different epoch than claimed.",
            "blockers": [74, 75],
            "important_follow_ups": [76, 77, 78, 79],
        },
        "validation": {
            "focused_tests": "101 passed",
            "focused_test_command": "uv run pytest -q experiments/tests/test_exp022_campaign.py experiments/tests/test_gamma_gated_sparsity_collection.py experiments/tests/test_experiment_arg_allowlist.py experiments/tests/test_runner_isolation.py experiments/tests/test_downstream_smoke_caps.py",
            "production_or_paid_compute_run": False,
        },
    }
    fail_closed(summary)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "production-plan.json").write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    provenance = {
        "audited_commit": AUDITED_COMMIT,
        "runner_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "capture_hashes": {
            "collection_plan_sha256": sha256(args.capture_root / "campaign" / "collection-plan.json"),
            "exp022_manifest_sha256": sha256(args.capture_root / "campaign" / "exp022" / "campaign.json"),
            "submission_sha256": sha256(args.capture_root / "submission.json"),
        },
        "published_plan_sha256": sha256(OUT / "production-plan.json"),
        "sanitization": "fail-closed scan passed; private paths and resource identities replaced with publication placeholders",
    }
    fail_closed(provenance)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    reproducer = {
        "purpose": "Recreate the issue-72 audit evidence without training or scheduler submission.",
        "commands": [
            "git checkout --detach " + AUDITED_COMMIT,
            "uv run python -m experiments.collections.gamma_gated_sparsity init --campaign-root <EXTERNAL_CAPTURE>/campaign --campaign-id issue72-audit-production",
            "uv run python -m experiments.collections.gamma_gated_sparsity submit --campaign-root <EXTERNAL_CAPTURE>/campaign --resources <PRIVATE_RESOURCES_JSON>",
            "git switch audit/issue-72-scientific-contract",
            "uv run python experiments/exp083.py <EXTERNAL_CAPTURE>",
        ],
        "note": "The submit command intentionally omits --live and --test-only. It emits a dry plan and creates no scheduler jobs. Its resource file needs nonzero placeholder walltimes and a working uv executable because retry-only cell resolution runs locally.",
    }
    fail_closed(reproducer)
    (OUT / "reproducer.json").write_text(json.dumps(reproducer, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
