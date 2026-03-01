"""Surrogate gradient spike function for differentiable SNN training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class SpikeFunction(object):
    """Hard-threshold spike with fast-sigmoid surrogate gradient.

    Forward:  s = (u >= 0).float()
    Backward: ds/du ≈ 1 / (1 + |u|)^2
    """

    @staticmethod
    def apply(u: "torch.Tensor") -> "torch.Tensor":
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyTorch backend requires torch to be installed") from exc
        return _SpikeFunctionAutograd.apply(u)


class _SpikeFunctionAutograd:
    # Defined below after torch is available at class-definition time via lazy import.
    pass


def _make_autograd_fn() -> type:
    import torch

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, u: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            ctx.save_for_backward(u)
            return (u >= 0).float()

        @staticmethod
        def backward(ctx: Any, grad_output: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            (u,) = ctx.saved_tensors
            surrogate = 1.0 / (1.0 + u.abs()) ** 2
            return grad_output * surrogate

    return _Fn


# Lazily initialise the autograd function on first use.
_autograd_fn: Any = None


def _get_autograd_fn() -> Any:
    global _autograd_fn
    if _autograd_fn is None:
        _autograd_fn = _make_autograd_fn()
    return _autograd_fn


# Patch SpikeFunction.apply to use the real autograd Function.
class SpikeFunction:  # type: ignore[no-redef]  # noqa: F811
    """Hard-threshold spike with fast-sigmoid surrogate gradient.

    Forward:  s = (u >= 0).float()
    Backward: ds/du ≈ 1 / (1 + |u|)^2
    """

    @staticmethod
    def apply(u: "torch.Tensor") -> "torch.Tensor":
        return _get_autograd_fn().apply(u)


def surrogate_lif_step(
    V: "torch.Tensor",
    g_e: "torch.Tensor",
    g_i: "torch.Tensor",
    I_ext: "torch.Tensor",
    dt: float,
    *,
    E_L: float,
    E_e: float,
    E_i: float,
    C_m: float | "torch.Tensor",
    g_L: float | "torch.Tensor",
    V_th: float | "torch.Tensor",
    V_reset: float,
    can_spike: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """LIF step with surrogate gradient spike function.

    Identical signature to lif_step. The only difference is the spike
    decision uses SpikeFunction (differentiable) instead of a hard boolean
    comparison, allowing gradients to flow back through the threshold.

    Pass this as the spike_fn argument to simulate_network for training.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    from pinglab.backends.pytorch.simulate_network import _as_tensor

    if can_spike is None:
        can_spike = torch.ones_like(V, dtype=torch.bool)

    C_m_t = _as_tensor(C_m, V)
    g_L_t = _as_tensor(g_L, V)
    V_th_t = _as_tensor(V_th, V)

    dVdt = (g_L_t * (E_L - V) + g_e * (E_e - V) + g_i * (E_i - V) + I_ext) / C_m_t
    V_new = V + float(dt) * dVdt

    # Surrogate spike: gradient-aware threshold via fast-sigmoid approximation.
    spiked = SpikeFunction.apply(V_new - V_th_t) * can_spike.float()

    # Reset spiking neurons (out-of-place to stay in autograd graph).
    spiked_bool = spiked.detach().bool()
    not_can = ~can_spike
    if bool(spiked_bool.any()) or bool(not_can.any()):
        reset_mask = spiked_bool | not_can
        V_new = torch.where(reset_mask, torch.full_like(V_new, float(V_reset)), V_new)

    return V_new, spiked
