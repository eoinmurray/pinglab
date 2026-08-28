"""Compatibility exports for exp084; execution requires an explicit stage."""

from experiments.helpers.gamma_frequency import (
    estimate_gamma_from_raster as estimate_gamma_from_raster,
)

from .measurements import (
    _phase_lag_ms as _phase_lag_ms,
)
from .measurements import (
    _rhythmicity_contrast as _rhythmicity_contrast,
)
from .measurements import (
    _trial_rows as _trial_rows,
)
from .measurements import (
    summarize_condition as summarize_condition,
)
from .recipe import (
    BURN_MS as BURN_MS,
)
from .recipe import (
    DISPLAY_TRIAL as DISPLAY_TRIAL,
)
from .recipe import (
    DT_MS as DT_MS,
)
from .recipe import (
    FREQUENCY_CONFIG as FREQUENCY_CONFIG,
)
from .recipe import (
    INPUT_RATES_HZ as INPUT_RATES_HZ,
)
from .recipe import (
    N_E as N_E,
)
from .recipe import (
    N_I as N_I,
)
from .recipe import (
    N_INPUT as N_INPUT,
)
from .recipe import (
    NETWORK_SEED as NETWORK_SEED,
)
from .recipe import (
    REPRESENTATIVE_RATES_HZ as REPRESENTATIVE_RATES_HZ,
)
from .recipe import (
    SCALE as SCALE,
)
from .recipe import (
    SLUG as SLUG,
)
from .recipe import (
    T_MS as T_MS,
)
from .recipe import (
    TRIAL_SEEDS as TRIAL_SEEDS,
)
from .recipe import (
    author_network as author_network,
)
from .recipe import (
    make_inputs as make_inputs,
)
