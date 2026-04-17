"""encode_smnist: row-by-row sequential encoding. If the row→time mapping
is wrong, sMNIST accuracy silently collapses but nothing crashes.
"""
import pytest
import torch

from oscilloscope import encode_smnist


def _row_pattern(active_row: int, B: int = 1) -> torch.Tensor:
    """Return (B, 784) pixels where only `active_row` is all-ones."""
    img = torch.zeros(B, 28, 28)
    img[:, active_row, :] = 1.0
    return img.reshape(B, 784)


class TestEncodeSmnist:
    @pytest.mark.parametrize("active_row", [0, 7, 13, 27])
    def test_spikes_confined_to_active_row_window(self, active_row):
        """Only rows with nonzero pixels should produce spikes, and only
        during that row's [r*steps_per_row, (r+1)*steps_per_row) window."""
        dt = 1.0
        t_ms_per_row = 10.0
        steps_per_row = int(t_ms_per_row / dt)   # 10
        img = _row_pattern(active_row)
        # Very high firing rate so the active row reliably fires every step
        spikes = encode_smnist(img, dt=dt, max_rate_hz=999.0,
                               t_ms_per_row=t_ms_per_row,
                               generator=torch.Generator().manual_seed(0))
        # Shape: (T_steps, B, 28)
        assert spikes.shape == (28 * steps_per_row, 1, 28)

        # Active row's window should have ~all-ones
        s = active_row * steps_per_row
        e = s + steps_per_row
        assert spikes[s:e].mean() > 0.95, \
            f"row {active_row} window mean={spikes[s:e].mean():.2f}"

        # Every other window should be exactly zero
        mask = torch.ones(28 * steps_per_row, dtype=torch.bool)
        mask[s:e] = False
        assert spikes[mask].sum().item() == 0.0

    def test_output_channel_dimension_is_ncols(self):
        """Output's last axis is 28 (column count), not 784."""
        img = torch.rand(3, 784)
        out = encode_smnist(img, dt=1.0, max_rate_hz=10.0)
        assert out.shape[-1] == 28
        assert out.shape[1] == 3

    def test_total_timesteps_matches_dt(self):
        for dt in [0.25, 0.5, 1.0]:
            out = encode_smnist(torch.zeros(1, 784), dt=dt, max_rate_hz=10.0,
                                t_ms_per_row=10.0)
            expected = 28 * int(10.0 / dt)
            assert out.shape[0] == expected, f"dt={dt} shape={out.shape}"

    def test_empirical_rate_matches_target(self):
        """Across a fully-on image, empirical spike rate ≈ max_rate_hz."""
        B = 32
        img = torch.ones(B, 784)
        rate_hz = 100.0
        dt = 0.5
        t_ms_per_row = 10.0
        out = encode_smnist(img, dt=dt, max_rate_hz=rate_hz,
                            t_ms_per_row=t_ms_per_row,
                            generator=torch.Generator().manual_seed(0))
        # Mean over all (T, B, cols) entries is per-step spike probability.
        # Convert to Hz: p * 1000/dt.
        emp_hz = out.mean().item() * 1000.0 / dt
        assert emp_hz == pytest.approx(rate_hz, rel=0.1), \
            f"empirical={emp_hz:.1f} Hz target={rate_hz} Hz"
