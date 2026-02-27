import numpy as np

from pinglab.analysis import first_significant_peak_lag_ms


def test_first_significant_peak_lag_ms_returns_first_peak_lag() -> None:
    lags = np.array([-20, -10, 0, 10, 20, 30, 40, 50], dtype=float)
    corr = np.array([0.0, 0.0, 1.0, 0.1, 0.6, 0.2, 0.7, 0.1], dtype=float)
    lag = first_significant_peak_lag_ms(
        lags,
        corr,
        corr_min_lag_ms=10.0,
        corr_max_lag_ms=50.0,
        corr_peak_min=0.2,
        corr_peak_prominence=0.1,
    )
    assert lag == 20.0
