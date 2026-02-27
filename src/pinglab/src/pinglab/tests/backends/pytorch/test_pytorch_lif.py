import pytest


torch = pytest.importorskip("torch")

from pinglab.backends.pytorch import lif_step


def test_lif_step_spike_and_reset() -> None:
    V = torch.tensor([-49.0, -60.0], dtype=torch.float32)
    g_e = torch.tensor([0.0, 0.0], dtype=torch.float32)
    g_i = torch.tensor([0.0, 0.0], dtype=torch.float32)
    I_ext = torch.tensor([10.0, 0.0], dtype=torch.float32)
    can_spike = torch.tensor([True, True])

    V_new, spiked = lif_step(
        V,
        g_e,
        g_i,
        I_ext,
        0.1,
        E_L=-65.0,
        E_e=0.0,
        E_i=-80.0,
        C_m=1.0,
        g_L=0.05,
        V_th=-50.0,
        V_reset=-65.0,
        can_spike=can_spike,
    )

    assert bool(spiked[0]) is True
    assert V_new[0].item() == pytest.approx(-65.0)


def test_lif_step_honors_refractory_mask() -> None:
    V = torch.tensor([-45.0], dtype=torch.float32)
    g_e = torch.tensor([0.0], dtype=torch.float32)
    g_i = torch.tensor([0.0], dtype=torch.float32)
    I_ext = torch.tensor([100.0], dtype=torch.float32)
    can_spike = torch.tensor([False])

    V_new, spiked = lif_step(
        V,
        g_e,
        g_i,
        I_ext,
        0.1,
        E_L=-65.0,
        E_e=0.0,
        E_i=-80.0,
        C_m=1.0,
        g_L=0.05,
        V_th=-50.0,
        V_reset=-65.0,
        can_spike=can_spike,
    )

    assert bool(spiked[0]) is False
    assert V_new[0].item() == pytest.approx(-65.0)
