"""Shared execution boundary for the four snnlang integration experiments."""

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def lineage(repo, recipe, identity, reference=None):
    found, visiting = {}, set()

    def visit(name, pin):
        if name in visiting:
            raise PingstoreError("cyclic integration evidence")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise PingstoreError("conflicting integration evidence pins")
            return
        run = source_run(
            repo / ".pingstore", name, experiment=recipe.SLUG, reference=pin
        )
        if run.record["execution"].get("configuration") != recipe.configuration():
            raise PingstoreError("source does not match the integration recipe")
        expected = {
            "compute": {},
            "analyse": {"compute": "compute"},
            "present": {"analysis": "analyse"},
        }[run.record["stage"]]
        if set(run.record["inputs"]) != set(expected):
            raise PingstoreError("unexpected integration stage inputs")
        visiting.add(name)
        for role, upstream in run.record["inputs"].items():
            visit(upstream["run_id"], upstream)
            if found[upstream["run_id"]].record["stage"] != expected[role]:
                raise PingstoreError("incorrect integration input stage")
        visiting.remove(name)
        found[name] = run

    visit(identity, reference)
    return found


def source(repo, recipe, identity, stage, *, reference=None):
    run = lineage(repo, recipe, identity, reference)[identity]
    if run.record["stage"] != stage:
        raise PingstoreError(f"{identity} is not a {stage} run")
    return run


def analysis_sources(repo, recipe, identity):
    analysis = source(repo, recipe, identity, "analyse")
    pin = analysis.record["inputs"]["compute"]
    compute = source(repo, recipe, pin["run_id"], "compute", reference=pin)
    result = load_json(analysis.export / "results.json")
    if result.get("schema") != f"{recipe.SLUG}.analysis/v1":
        raise PingstoreError("unsupported integration analysis schema")
    return analysis, compute, result


@contextmanager
def execution(repo, recipe, stage, *, sources=None, run_id=None):
    sources = sources or {}
    ancestors = {}
    for selected in sources.values():
        for name, ancestor in lineage(
            repo, recipe, selected.record["run_id"], selected.reference
        ).items():
            if name in ancestors and ancestors[name].reference != ancestor.reference:
                raise PingstoreError("conflicting integration ancestry")
            ancestors[name] = ancestor
    with stage_run(
        repo,
        recipe.SLUG,
        stage,
        inputs=sources,
        run_id=run_id,
        configuration=recipe.configuration(),
    ) as run:
        yield run
        for ancestor in ancestors.values():
            ancestor.check_unchanged()


def command(repo, provenance, name, argv):
    """Keep the actual child command and complete logs, including failed work."""
    env = dict(os.environ)
    env.setdefault("PINGLAB_NO_COMPILE", "1")
    started = time.monotonic()
    with (
        (provenance / f"{name}.stdout").open("w") as out,
        (provenance / f"{name}.stderr").open("w") as err,
    ):
        result = subprocess.run(
            argv, cwd=repo, env=env, stdout=out, stderr=err, check=False
        )
    record = {
        "cmd": argv,
        "returncode": result.returncode,
        "elapsed_s": time.monotonic() - started,
        "environment": {
            key: env.get(key)
            for key in ("PINGLAB_NO_COMPILE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json_atomic(provenance / f"{name}.json", record)
    # The simulator owns these execution attachments, not scientific outputs.
    # Move them before atomic completion; completed sources are never rewritten.
    if "--out-dir" in argv:
        output = Path(argv[argv.index("--out-dir") + 1])
        for filename in ("run.sh", "run.jsonl", "output.log"):
            attachment = output / filename
            if attachment.is_file():
                retained = provenance / name
                retained.mkdir(exist_ok=True)
                attachment.rename(retained / filename)
    if result.returncode:
        tail = (provenance / f"{name}.stderr").read_text()[-4000:]
        raise RuntimeError(f"{name} failed ({result.returncode}): {tail}")
    return record


def test_evidence(repo, run, name, nodes):
    """Execute an explicitly scoped numerical gate and retain its test report."""
    import sys

    report = run.provenance / f"{name}.xml"
    record = command(
        repo,
        run.provenance,
        name,
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            *nodes,
        ],
    )
    suites = ElementTree.parse(report).getroot().findall(".//testsuite")
    counts = {
        key: sum(int(suite.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if not counts["tests"] or any(
        counts[key] for key in ("failures", "errors", "skipped")
    ):
        raise PingstoreError(
            f"{name} did not execute every numerical gate successfully"
        )
    return {"nodes": nodes, **counts, "elapsed_s": record["elapsed_s"], "passed": True}


def configuration(constants, graphs):
    """Canonical JSON representation of committed settings and compiled graphs."""
    return json.loads(
        json.dumps({"constants": constants, "graphs": graphs}, sort_keys=True)
    )
