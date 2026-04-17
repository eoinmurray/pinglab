import numpy as np
import pytest
import torch

from spectral import analyze_spectrum, build_recurrent_matrix


class _FakeNet:
    """Minimal stand-in for PINGNet exposing the three recurrent buffers."""
    def __init__(self, W_ee, W_ei, W_ie):
        self.W_ee = torch.tensor(W_ee, dtype=torch.float32)
        self.W_ei = torch.tensor(W_ei, dtype=torch.float32)
        self.W_ie = torch.tensor(W_ie, dtype=torch.float32)


class TestBuildRecurrentMatrix:
    def test_block_structure(self):
        N_E, N_I = 3, 2
        W_ee = np.arange(N_E * N_E, dtype=float).reshape(N_E, N_E)
        W_ei = np.arange(N_E * N_I, dtype=float).reshape(N_E, N_I) + 100
        W_ie = np.arange(N_I * N_E, dtype=float).reshape(N_I, N_E) + 200
        net = _FakeNet(W_ee, W_ei, W_ie)

        J, n_e, n_i = build_recurrent_matrix(net)

        assert (n_e, n_i) == (N_E, N_I)
        assert J.shape == (N_E + N_I, N_E + N_I)
        np.testing.assert_array_equal(J[:N_E, :N_E], W_ee)
        np.testing.assert_array_equal(J[:N_E, N_E:], -W_ie.T)
        np.testing.assert_array_equal(J[N_E:, :N_E], W_ei.T)
        np.testing.assert_array_equal(J[N_E:, N_E:], np.zeros((N_I, N_I)))

    def test_sign_conventions(self):
        """E columns positive, I columns negative (Dale's law encoding)."""
        W_ee = np.ones((2, 2))
        W_ei = np.ones((2, 2))
        W_ie = np.ones((2, 2))
        net = _FakeNet(W_ee, W_ei, W_ie)
        J, _, _ = build_recurrent_matrix(net)
        # E columns (0, 1) should be >= 0; I columns (2, 3) should be <= 0.
        assert np.all(J[:, :2] >= 0)
        assert np.all(J[:, 2:] <= 0)


class TestAnalyzeSpectrum:
    def test_known_2x2_block_eigenvalues(self):
        """Hand-solvable E-I toy: W_ee=W_ei=W_ie=a (scalars, N_E=N_I=1).

        J = [[a, -a], [a, 0]].  Characteristic poly: λ² - aλ + a² = 0.
        Roots: λ = a/2 ± a·i·√3/2  =>  |imag| = a·√3/2.
        Predicted f0 = |imag| / (2π) * 1000 Hz.
        """
        a = 0.8
        net = _FakeNet(np.array([[a]]), np.array([[a]]), np.array([[a]]))
        J, N_E, N_I = build_recurrent_matrix(net)
        result = analyze_spectrum(J, dt=0.25, N_E=N_E)

        expected_imag = a * np.sqrt(3) / 2
        expected_f0 = expected_imag / (2 * np.pi) * 1000.0
        assert result["predicted_f0_linear"] == pytest.approx(expected_f0, rel=1e-4)

    def test_no_complex_eigenvalues_gives_zero_f0(self):
        """Diagonal real matrix => all real eigenvalues => linear f0 = 0."""
        J = np.diag([0.5, -0.3, 0.1, -0.2])
        result = analyze_spectrum(J, dt=0.25, N_E=2)
        assert result["predicted_f0_linear"] == 0.0
        assert result["dominant_complex"] is None

    def test_spectral_radius_matches_max_abs_eig(self):
        rng = np.random.RandomState(0)
        J = rng.randn(6, 6) * 0.1
        result = analyze_spectrum(J, dt=0.25, N_E=3)
        expected = np.max(np.abs(np.linalg.eigvals(J)))
        assert result["spectral_radius"] == pytest.approx(expected)
