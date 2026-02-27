from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field



class Spikes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    times: np.ndarray  # Spike times in ms
    ids: np.ndarray  # Neuron indices corresponding to spike times
    types: np.ndarray | None = None  # 0=E,1=I, neuron types if applicable
    populations: np.ndarray | None = None  # For two-population simulations


class InstrumentsConfig(BaseModel):
    variables: list[str]  # e.g., ['V', 'g_e', 'g_i']
    neuron_ids: list[int] | None = None  # Neuron indices to record from
    downsample: int = 1  # Downsampling factor for recorded data
    population_means: bool = False  # Record population-averaged traces for E and I
    all_neurons: bool = False  # Record all neurons (overrides neuron_ids)


class InstrumentsResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    times: np.ndarray
    neuron_ids: np.ndarray
    types: np.ndarray | None = None  # Neuron types if applicable
    V: np.ndarray | None = None
    g_e: np.ndarray | None = None
    g_i: np.ndarray | None = None
    # Population means (when population_means=True)
    V_mean_E: np.ndarray | None = None
    V_mean_I: np.ndarray | None = None
    g_e_mean_E: np.ndarray | None = None
    g_e_mean_I: np.ndarray | None = None
    g_i_mean_E: np.ndarray | None = None
    g_i_mean_I: np.ndarray | None = None


class NetworkConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    instruments: InstrumentsConfig = Field(
        default_factory=lambda: InstrumentsConfig(
            variables=["V", "g_e", "g_i"],
            all_neurons=True,
        )
    )
    external_input: np.ndarray | None = (
        None  # Required at runtime. Shape: (num_steps, N_E + N_I) or (num_steps,) for uniform
    )
    neuron_model: Literal["lif", "mqif"] = "lif"
    dt: float
    T: float
    N_E: int
    N_I: int
    pulse_onset_ms: float = 0.0
    pulse_duration_ms: float = 0.0
    pulse_interval_ms: float = 0.0
    pulse_amplitude_E: float = 0.0
    pulse_amplitude_I: float = 0.0
    delay_ei: float = 1.5
    delay_ie: float = 1.5
    delay_ee: float = 1.5
    delay_ii: float = 1.5
    seed: int | None = None
    # LIF neuron parameters
    V_init: float = -65.0
    E_L: float = -65.0
    E_e: float = 0.0
    E_i: float = -80.0
    C_m_E: float = 1.0
    g_L_E: float = 0.1
    C_m_I: float = 1.0
    g_L_I: float = 0.1
    V_th: float = -50.0
    V_reset: float = -65.0
    # MQIF parameters
    mqif_a: list[float] = []
    mqif_Vr: list[float] = []
    mqif_w_a: list[float] = []
    mqif_w_Vr: list[float] = []
    mqif_w_tau: list[float] = []
    # Synaptic time constants
    tau_ampa: float = 5.0
    tau_gaba: float = 10.0
    t_ref_E: float = 3.0
    t_ref_I: float = 1.5
    # Heterogeneity parameters
    V_th_heterogeneity_sd: float = 0.0
    g_L_heterogeneity_sd: float = 0.0
    C_m_heterogeneity_sd: float = 0.0
    t_ref_heterogeneity_sd: float = 0.0


class NetworkResult(BaseModel):
    spikes: Spikes
    instruments: InstrumentsResults


class DictModel(BaseModel):
    """
    Base model that:
    - allows arbitrary extra fields
    - recursively converts EXTRA dicts/lists-of-dicts into DictModel
      so they're dot-accessible, without touching declared fields.
    """
    model_config = ConfigDict(extra="allow")

    def __init__(self, **data):
        fields = self.__class__.model_fields
        converted: dict = {}

        for k, v in data.items():
            if k not in fields:
                # Only auto-wrap *extra* fields
                if isinstance(v, dict):
                    v = DictModel(**v)
                elif isinstance(v, list):
                    v = [DictModel(**i) if isinstance(i, dict) else i for i in v]
            converted[k] = v

        super().__init__(**converted)


class Inputs(BaseModel):
    I_E: float
    I_I: float
    noise: float


class WeightBlockSpec(BaseModel):
    mean: float = 0.0
    std: float = 0.0


class WeightSpec(BaseModel):
    ee: WeightBlockSpec
    ei: WeightBlockSpec
    ie: WeightBlockSpec
    ii: WeightBlockSpec
    clamp_min: float | None = 0.0

class PlottingConfig(BaseModel):
    raster: 'RasterConfig'

class RasterConfig(BaseModel):
    start_time: float
    stop_time: float

class ExperimentConfig(DictModel):
    base: NetworkConfig
    plotting: PlottingConfig | None = None
    default_inputs: Inputs
    weights: WeightSpec | None = None
