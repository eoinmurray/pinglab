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

        def _matmul_add(target: np.ndarray, W: np.ndarray, spikes: np.ndarray, label: str) -> None:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                delta = W @ spikes
            if not np.isfinite(delta).all():
                raise RuntimeError(f"Non-finite conductance update in {label}.")
            target += delta

        if spikes_ee.any():
            _matmul_add(g_e[: config.N_E], self.weights.W_ee, spikes_ee, "W_ee @ spikes_ee")
        if spikes_ei.any():
            _matmul_add(g_e[config.N_E :], self.weights.W_ei, spikes_ei, "W_ei @ spikes_ei")
        if spikes_ie.any():
            _matmul_add(g_i[: config.N_E], self.weights.W_ie, spikes_ie, "W_ie @ spikes_ie")
        if spikes_ii.any():
            _matmul_add(g_i[config.N_E :], self.weights.W_ii, spikes_ii, "W_ii @ spikes_ii")

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


@dataclass
class EventConnectivity:
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
    ee_targets: list[np.ndarray]
    ee_weights: list[np.ndarray]
    ei_targets: list[np.ndarray]
    ei_weights: list[np.ndarray]
    ie_targets: list[np.ndarray]
    ie_weights: list[np.ndarray]
    ii_targets: list[np.ndarray]
    ii_weights: list[np.ndarray]

    def _apply_block(
        self,
        target: np.ndarray,
        spikes: np.ndarray,
        targets: list[np.ndarray],
        weights: list[np.ndarray],
        label: str,
    ) -> None:
        spiked_idx = np.nonzero(spikes)[0]
        if spiked_idx.size == 0:
            return
        for idx in spiked_idx:
            tgt = targets[idx]
            if tgt.size == 0:
                continue
            target[tgt] += weights[idx]
        if not np.isfinite(target).all():
            raise RuntimeError(f"Non-finite conductance update in {label}.")

    def apply(self, config: NetworkConfig, g_e: np.ndarray, g_i: np.ndarray) -> None:
        spikes_ei = self.buffer_e_to_i[self.buf_idx]
        spikes_ee = self.buffer_e_to_e[self.buf_idx]
        spikes_ie = self.buffer_i_to_e[self.buf_idx]
        spikes_ii = self.buffer_i_to_i[self.buf_idx]

        if spikes_ee.any():
            self._apply_block(
                g_e[: config.N_E],
                spikes_ee,
                self.ee_targets,
                self.ee_weights,
                "event W_ee",
            )
        if spikes_ei.any():
            self._apply_block(
                g_e[config.N_E :],
                spikes_ei,
                self.ei_targets,
                self.ei_weights,
                "event W_ei",
            )
        if spikes_ie.any():
            self._apply_block(
                g_i[: config.N_E],
                spikes_ie,
                self.ie_targets,
                self.ie_weights,
                "event W_ie",
            )
        if spikes_ii.any():
            self._apply_block(
                g_i[config.N_E :],
                spikes_ii,
                self.ii_targets,
                self.ii_weights,
                "event W_ii",
            )

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


def _outgoing_lists(W: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for src in range(W.shape[1]):
        nz = np.nonzero(W[:, src])[0]
        targets.append(nz)
        weights.append(W[nz, src])
    return targets, weights


def build_event_connectivity(
    config: NetworkConfig, weights: WeightMatrices | np.ndarray
) -> EventConnectivity:
    delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps = _delay_steps(config)
    buf_len = max(delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps) + 1

    if isinstance(weights, np.ndarray):
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("Weights matrix must be square (N x N).")
        if weights.shape[0] != config.N_E + config.N_I:
            raise ValueError("Weights matrix size must match N_E + N_I.")
        weights = split_weight_matrix(weights, config.N_E)

    ee_targets, ee_weights = _outgoing_lists(weights.W_ee)
    ei_targets, ei_weights = _outgoing_lists(weights.W_ei)
    ie_targets, ie_weights = _outgoing_lists(weights.W_ie)
    ii_targets, ii_weights = _outgoing_lists(weights.W_ii)

    return EventConnectivity(
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
        ee_targets=ee_targets,
        ee_weights=ee_weights,
        ei_targets=ei_targets,
        ei_weights=ei_weights,
        ie_targets=ie_targets,
        ie_weights=ie_weights,
        ii_targets=ii_targets,
        ii_weights=ii_weights,
    )
