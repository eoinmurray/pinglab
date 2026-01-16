from pydantic import BaseModel


class PhasePortraitConfig(BaseModel):
    V_min: float
    V_max: float
    w_min: float
    w_max: float
    V_points: int
    w_points: int
    I_ext: float


class AdExConfig(BaseModel):
    C_m: float
    g_L: float
    E_L: float
    V_T: float
    Delta_T: float
    tau_w: float
    a: float
    b: float
    V_reset: float
    V_peak: float


class LocalConfig(BaseModel):
    phase_portrait: PhasePortraitConfig
    adex: AdExConfig
