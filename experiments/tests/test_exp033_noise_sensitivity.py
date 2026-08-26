from __future__ import annotations

import numpy as np
from experiments import exp033


def test_refined_hopf_lies_inside_coarse_bracket() -> None:
    grid = np.linspace(0.0, 1.2, 25)
    hopf = exp033.find_hopf(exp033.sweep(grid, sigma=4.0), sigma=4.0)
    assert hopf is not None
    lo, hi = hopf["coarse_bracket_nA"]
    assert lo <= hopf["I_ext_star"] <= hi
    assert abs(hopf["leading_eigenvalue"][0]) < 1e-8


def test_hysteresis_propagates_sigma(monkeypatch) -> None:
    observed: list[float] = []

    def fake_fixed_point(_drive, _tau=exp033.TAU_GABA_MS, x0=None, sigma=4.0):
        observed.append(sigma)
        return np.ones(4)

    def fake_settle(drive, state, _tau=exp033.TAU_GABA_MS, sigma=4.0, **_kwargs):
        observed.append(sigma)
        return max(0.0, drive - 0.5), state

    monkeypatch.setattr(exp033, "fixed_point", fake_fixed_point)
    monkeypatch.setattr(exp033, "settle", fake_settle)
    exp033.hysteresis_sweep(0.5, sigma=5.5, span=(-0.05, 0.05), n=3)
    assert observed and set(observed) == {5.5}


def test_tau_gaba_sweep_propagates_sigma(monkeypatch) -> None:
    observed: list[tuple[float, float]] = []

    def fake_sweep(_grid, tau_gaba=0.0, sigma=0.0):
        observed.append((tau_gaba, sigma))
        return []

    monkeypatch.setattr(exp033, "sweep", fake_sweep)
    monkeypatch.setattr(exp033, "find_hopf", lambda *_args, **_kwargs: None)
    exp033.frequency_vs_tau_gaba([4.5, 9.0], np.array([0.0, 1.0]), sigma=6.0)
    assert observed == [(4.5, 6.0), (9.0, 6.0)]


def test_exp054_explicitly_selects_reference_sigma() -> None:
    source = (exp033.REPO / "experiments" / "exp054.py").read_text()
    assert "sigma = exp033.SIGMA_V_MV" in source
    assert "hysteresis_sweep(hopf[\"I_ext_star\"], sigma=sigma)" in source


def test_publication_text_does_not_claim_fully_fitted_scale() -> None:
    corpus = "\n".join(
        (exp033.REPO / path).read_text()
        for path in (
            "writings/exp033.typ",
            "writings/exp054.typ",
            "writings/exp009.typ",
        )
    ).lower()
    assert "no fitted scale" not in corpus
    assert "absolute scale is fixed by the biophysics" not in corpus
