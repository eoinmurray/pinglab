import numpy as np

from pinglab.io import collect_scans, linspace_from_scan, scan_variant


def test_collect_scans_picks_scan_and_scan_prefixed_keys() -> None:
    meta = {
        "scan": {"parameter": "a", "steps": 2},
        "scan_b": {"parameter": "b", "steps": 2},
        "other": {"ignored": True},
    }
    scans = collect_scans(meta)
    assert scans[0][0] == "scan"
    assert ("scan_b", meta["scan_b"]) in scans


def test_linspace_from_scan_uses_steps() -> None:
    values = linspace_from_scan({"start": 0.0, "stop": 1.0, "steps": 3})
    assert np.allclose(values, np.array([0.0, 0.5, 1.0]))


def test_scan_variant_uses_parameter_suffix() -> None:
    assert scan_variant("e_to_e.w.std", "scan_1") == "std"
    assert scan_variant("e_to_e.w.mean", "scan_1") == "mean"
    assert scan_variant("foo.bar", "scan_custom") == "scan_custom"
