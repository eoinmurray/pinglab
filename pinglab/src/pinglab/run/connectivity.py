from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pinglab.lib import WeightMatrices, split_weight_matrix
from pinglab.types import NetworkConfig


@dataclass
class AdjacencyConnectivity:
    buffer_e_to_i: np.ndarray
    buffer_e_to_e: np.ndarray
    buffer_i_to_e: np.ndarray
    buffer_i_to_i: np.ndarray
    buf_idx: int
    buf_len: int
    delay_ei_steps: int
    delay_ie_steps: int
    delay_ee_steps: int
    delay_ii_steps: int
    weights: WeightMatrices

    def apply(self, config: NetworkConfig, g_e: np.ndarray, g_i: np.ndarray) -> None:
        spikes_ei = self.buffer_e_to_i[self.buf_idx]
        spikes_ee = self.buffer_e_to_e[self.buf_idx]
        spikes_ie = self.buffer_i_to_e[self.buf_idx]
        spikes_ii = self.buffer_i_to_i[self.buf_idx]

        if spikes_ee.any():
            g_e[: config.N_E] += self.weights.W_ee @ spikes_ee
        if spikes_ei.any():
            g_e[config.N_E :] += self.weights.W_ei @ spikes_ei
        if spikes_ie.any():
            g_i[: config.N_E] += self.weights.W_ie @ spikes_ie
        if spikes_ii.any():
            g_i[config.N_E :] += self.weights.W_ii @ spikes_ii

        self.buffer_e_to_i[self.buf_idx] = 0.0
        self.buffer_e_to_e[self.buf_idx] = 0.0
        self.buffer_i_to_e[self.buf_idx] = 0.0
        self.buffer_i_to_i[self.buf_idx] = 0.0

    def schedule(self, spiked: np.ndarray, N_E: int) -> None:
        if not spiked.any():
            return
        spiked_E = spiked[:N_E].astype(float)
        spiked_I = spiked[N_E:].astype(float)
        tgt_ei = (self.buf_idx + self.delay_ei_steps) % self.buf_len
        tgt_ie = (self.buf_idx + self.delay_ie_steps) % self.buf_len
        tgt_ee = (self.buf_idx + self.delay_ee_steps) % self.buf_len
        tgt_ii = (self.buf_idx + self.delay_ii_steps) % self.buf_len
        self.buffer_e_to_i[tgt_ei] = spiked_E
        self.buffer_e_to_e[tgt_ee] = spiked_E
        self.buffer_i_to_e[tgt_ie] = spiked_I
        self.buffer_i_to_i[tgt_ii] = spiked_I

    def advance(self) -> None:
        self.buf_idx = (self.buf_idx + 1) % self.buf_len


def _delay_steps(config: NetworkConfig) -> tuple[int, int, int, int]:
    delay_ei_steps = max(1, int(np.round(config.delay_ei / config.dt)))
    delay_ie_steps = max(1, int(np.round(config.delay_ie / config.dt)))
    delay_ee_steps = max(1, int(np.round(config.delay_ee / config.dt)))
    delay_ii_steps = max(1, int(np.round(config.delay_ii / config.dt)))
    return delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps


def build_connectivity(config: NetworkConfig, weights: WeightMatrices | np.ndarray):
    delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps = _delay_steps(config)
    buf_len = max(delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps) + 1

    if isinstance(weights, np.ndarray):
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("Weights matrix must be square (N x N).")
        if weights.shape[0] != config.N_E + config.N_I:
            raise ValueError("Weights matrix size must match N_E + N_I.")
        weights = split_weight_matrix(weights, config.N_E)

    return AdjacencyConnectivity(
        buffer_e_to_i=np.zeros((buf_len, config.N_E), dtype=float),
        buffer_e_to_e=np.zeros((buf_len, config.N_E), dtype=float),
        buffer_i_to_e=np.zeros((buf_len, config.N_I), dtype=float),
        buffer_i_to_i=np.zeros((buf_len, config.N_I), dtype=float),
        buf_idx=0,
        buf_len=buf_len,
        delay_ei_steps=delay_ei_steps,
        delay_ie_steps=delay_ie_steps,
        delay_ee_steps=delay_ee_steps,
        delay_ii_steps=delay_ii_steps,
        weights=weights,
    )
