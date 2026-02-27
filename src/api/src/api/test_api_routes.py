from __future__ import annotations

from fastapi.testclient import TestClient

import api.api as api_module
import api.web_app as web_app_module
from pinglab.service.contracts import RunResponse, SpikesResponse, WeightsResponse


def _sample_run_response() -> RunResponse:
    return RunResponse(
        spikes=SpikesResponse(times=[1.0, 2.0], ids=[0, 1], types=[0, 1]),
        core_sim_ms=12.5,
        runtime_ms=12.5,
        num_steps=10,
        num_spikes=2,
        spikes_truncated=False,
        mean_rate_E=1.0,
        mean_rate_I=1.5,
        isi_cv_E=0.2,
        autocorr_peak=0.3,
        xcorr_peak=0.4,
        coherence_peak=0.5,
        lagged_coherence=0.6,
        population_rate_t_ms=[],
        population_rate_hz_E=[],
        population_rate_hz_I=[],
        population_rate_hz_layers=[],
        membrane_t_ms=[],
        membrane_V_E=[],
        membrane_V_I=[],
        membrane_V_layers=[],
        membrane_g_e_E=[],
        membrane_g_i_E=[],
        membrane_g_e_I=[],
        membrane_g_i_I=[],
        autocorr_lags_ms=[],
        autocorr_corr=[],
        autocorr_lags_layers_ms=[],
        autocorr_corr_layers=[],
        xcorr_lags_ms=[],
        xcorr_corr=[],
        xcorr_lags_layers_ms=[],
        xcorr_corr_layers=[],
        coherence_lags_ms=[],
        coherence_corr=[],
        weights_hist_bins=[],
        weights_hist_counts_ee=[],
        weights_hist_counts_ei=[],
        weights_hist_counts_ie=[],
        weights_hist_counts_ii=[],
        weights_hist_blocks_ee=[],
        weights_hist_blocks_ei=[],
        weights_hist_blocks_ie=[],
        weights_hist_blocks_ii=[],
        weights_heatmap=[],
        psd_freqs_hz=[],
        psd_power=[],
        psd_power_layers=[],
        input_t_ms=[],
        input_mean_E=[],
        input_mean_I=[],
        input_mean_layers=[],
        layer_labels=[],
        input_prep_ms=1.0,
        weights_build_ms=2.0,
        analysis_ms=3.0,
        response_build_ms=4.0,
        server_compute_ms=5.0,
    )


def _sample_weights_response() -> WeightsResponse:
    return WeightsResponse(
        weights_hist_bins=[0.1, 0.2],
        weights_hist_counts_ee=[1.0, 0.0],
        weights_hist_counts_ei=[0.0, 1.0],
        weights_hist_counts_ie=[0.0, 0.0],
        weights_hist_counts_ii=[0.0, 0.0],
        weights_hist_blocks_ee=[[1.0, 0.0]],
        weights_hist_blocks_ei=[[0.0, 1.0]],
        weights_hist_blocks_ie=[[0.0, 0.0]],
        weights_hist_blocks_ii=[[0.0, 0.0]],
        weights_heatmap=[[1.0]],
    )


def test_run_route_returns_timing_headers(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "run_simulation", lambda request: _sample_run_response())
    client = TestClient(api_module.app)

    response = client.post("/run", json={})

    assert response.status_code == 200
    assert response.json()["num_steps"] == 10
    for header in (
        "x-pinglab-core-sim-ms",
        "x-pinglab-input-prep-ms",
        "x-pinglab-weights-build-ms",
        "x-pinglab-analysis-ms",
        "x-pinglab-response-build-ms",
        "x-pinglab-server-compute-ms",
        "x-pinglab-serialize-ms",
        "x-pinglab-response-bytes",
    ):
        assert header in response.headers


def test_run_route_returns_arrow_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "run_simulation", lambda request: _sample_run_response())
    client = TestClient(api_module.app)

    response = client.post("/run", json={}, headers={"accept": "application/vnd.apache.arrow.stream"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
    assert len(response.content) > 0


def test_weights_route_accepts_empty_body(monkeypatch) -> None:
    captured = {"request": "unset"}

    def fake_build(request):
        captured["request"] = request
        return _sample_weights_response()

    monkeypatch.setattr(web_app_module, "build_weights_preview", fake_build)
    client = TestClient(api_module.app)

    response = client.post("/weights")

    assert response.status_code == 200
    assert isinstance(captured["request"], web_app_module.WeightsRequest)
    assert captured["request"].config is None
    assert captured["request"].weights is None
    assert response.json()["weights_heatmap"] == [[1.0]]


def test_weights_route_returns_arrow_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "build_weights_preview", lambda request: _sample_weights_response())
    client = TestClient(api_module.app)

    response = client.post("/weights", headers={"accept": "application/vnd.apache.arrow.stream"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
    assert len(response.content) > 0
