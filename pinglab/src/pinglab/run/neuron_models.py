from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pinglab.lib import (
    lif_step,
    hh_step,
    hh_init_gating,
    adex_step,
    cs_step,
    cs_init_gating,
    fhn_step,
    izh_step,
    izh_init_u,
    mqif_step,
    qif_step,
)
from pinglab.types import NetworkConfig


class BaseNeuronModel:
    requires_reset: bool = False

    def __init__(self, config: NetworkConfig):
        self.config = config

    def initialize(self, V: np.ndarray) -> None:
        return None

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {}


class LIFModel(BaseNeuronModel):
    requires_reset = True

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V, spiked = lif_step(
            V,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            E_L=self.config.E_L,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
            C_m=C_m,
            g_L=g_L,
            V_th=V_th,
            V_reset=self.config.V_reset,
            can_spike=can_spike,
        )
        spiked = spiked & can_spike
        return V, spiked


class HHModel(BaseNeuronModel):
    def initialize(self, V: np.ndarray) -> None:
        self.m, self.h, self.n = hh_init_gating(V)

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V_prev = V.copy()
        V, self.m, self.h, self.n = hh_step(
            V,
            self.m,
            self.h,
            self.n,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            C_m=C_m,
            g_L=g_L,
            g_Na=self.config.g_Na,
            g_K=self.config.g_K,
            E_L=self.config.E_L,
            E_Na=self.config.E_Na,
            E_K=self.config.E_K,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
        )
        spiked = (V_prev < V_th) & (V >= V_th) & can_spike
        return V, spiked

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {"m": self.m, "h": self.h, "n": self.n}


class AdExModel(BaseNeuronModel):
    requires_reset = True

    def initialize(self, V: np.ndarray) -> None:
        self.w = np.zeros_like(V)

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V, self.w, spiked = adex_step(
            V,
            self.w,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            C_m=C_m,
            g_L=g_L,
            E_L=self.config.E_L,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
            V_T=self.config.adex_V_T,
            Delta_T=self.config.adex_delta_T,
            tau_w=self.config.adex_tau_w,
            a=self.config.adex_a,
            b=self.config.adex_b,
            V_reset=self.config.V_reset,
            V_peak=self.config.adex_V_peak,
            can_spike=can_spike,
        )
        return V, spiked

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {"w": self.w}


class ConnorStevensModel(BaseNeuronModel):
    def initialize(self, V: np.ndarray) -> None:
        self.m, self.h, self.n, self.cs_a, self.cs_b = cs_init_gating(V)

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V_prev = V.copy()
        V, self.m, self.h, self.n, self.cs_a, self.cs_b = cs_step(
            V,
            self.m,
            self.h,
            self.n,
            self.cs_a,
            self.cs_b,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            C_m=C_m,
            g_L=g_L,
            g_Na=self.config.g_Na,
            g_K=self.config.g_K,
            g_A=self.config.g_A,
            E_L=self.config.E_L,
            E_Na=self.config.E_Na,
            E_K=self.config.E_K,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
        )
        spiked = (V_prev < V_th) & (V >= V_th) & can_spike
        return V, spiked

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {"m": self.m, "h": self.h, "n": self.n, "cs_a": self.cs_a, "cs_b": self.cs_b}


class FitzHughModel(BaseNeuronModel):
    def initialize(self, V: np.ndarray) -> None:
        self.w = np.zeros_like(V)

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V_prev = V.copy()
        V, self.w = fhn_step(
            V,
            self.w,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            a=self.config.fhn_a,
            b=self.config.fhn_b,
            tau_w=self.config.fhn_tau_w,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
        )
        spiked = (V_prev < V_th) & (V >= V_th) & can_spike
        return V, spiked

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {"w": self.w}


class MQIFModel(BaseNeuronModel):
    requires_reset = True

    def initialize(self, V: np.ndarray) -> None:
        self.a_terms = np.asarray(self.config.mqif_a, dtype=float)
        self.vr_terms = np.asarray(self.config.mqif_Vr, dtype=float)
        if self.a_terms.size != self.vr_terms.size:
            raise ValueError("mqif_a and mqif_Vr must have the same length")

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V, spiked = mqif_step(
            V,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            C_m=C_m,
            g_L=g_L,
            E_L=self.config.E_L,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
            a_terms=self.a_terms,
            V_r_terms=self.vr_terms,
            V_th=V_th,
            V_reset=self.config.V_reset,
            can_spike=can_spike,
        )
        return V, spiked

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {"a_terms": self.a_terms, "V_r_terms": self.vr_terms}


class QIFModel(BaseNeuronModel):
    requires_reset = True

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V, spiked = qif_step(
            V,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            C_m=C_m,
            g_L=g_L,
            E_L=self.config.E_L,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
            a=self.config.qif_a,
            V_r=self.config.qif_Vr,
            V_t=self.config.qif_Vt,
            V_th=V_th,
            V_reset=self.config.V_reset,
            can_spike=can_spike,
        )
        return V, spiked


class IzhikevichModel(BaseNeuronModel):
    def initialize(self, V: np.ndarray) -> None:
        self.u = izh_init_u(V, self.config.izh_b)

    def step(
        self,
        V: np.ndarray,
        g_e: np.ndarray,
        g_i: np.ndarray,
        I_ext: np.ndarray,
        C_m: np.ndarray,
        g_L: np.ndarray,
        V_th: np.ndarray,
        can_spike: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V, self.u, spiked = izh_step(
            V,
            self.u,
            g_e,
            g_i,
            I_ext,
            self.config.dt,
            a=self.config.izh_a,
            b=self.config.izh_b,
            c=self.config.izh_c,
            d=self.config.izh_d,
            V_th=V_th,
            E_e=self.config.E_e,
            E_i=self.config.E_i,
            can_spike=can_spike,
        )
        return V, spiked

    def get_state(self) -> dict[str, np.ndarray | float | int | None]:
        return {"u": self.u}


MODEL_BUILDERS = {
    "lif": LIFModel,
    "hh": HHModel,
    "adex": AdExModel,
    "connor_stevens": ConnorStevensModel,
    "fitzhugh": FitzHughModel,
    "mqif": MQIFModel,
    "qif": QIFModel,
    "izhikevich": IzhikevichModel,
}


def build_model_from_config(config: NetworkConfig) -> BaseNeuronModel:
    if config.neuron_model not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported neuron_model: {config.neuron_model}")
    return MODEL_BUILDERS[config.neuron_model](config)
