"""Release-identity contracts for experiment-facing tools."""

import re

from tools import snnlang, snnsim, snnviz
from tools.pingstore.campaign_runtime import tool_versions

SEMVER = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def test_experiment_tool_versions_are_public_semver():
    expected = {
        "snnlang": "0.1.0",
        "snnsim": "0.1.0",
        "snnviz": "0.1.0",
    }

    assert tool_versions() == expected
    assert {
        "snnlang": snnlang.__version__,
        "snnsim": snnsim.__version__,
        "snnviz": snnviz.__version__,
    } == expected
    assert all(SEMVER.fullmatch(version) for version in expected.values())


def test_snnlang_bundle_records_its_compiler_version():
    network = snnlang.Network("version-contract")
    bundle = snnlang.compile(network)

    assert bundle.manifest["compiler"] == {
        "name": "snnlang",
        "version": snnlang.__version__,
    }
