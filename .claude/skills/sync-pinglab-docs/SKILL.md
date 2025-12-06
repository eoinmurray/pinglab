---
name: sync-pinglab-docs
description: Sync the pinglab Python library documentation with the source code. Reads all pinglab source files and generates comprehensive API documentation in biblio/docs/README.mdx.
license: MIT
---

This skill generates and updates comprehensive documentation for the pinglab Python library by reading the source code and producing a well-structured API reference.

## When to use

- User asks to update or sync the pinglab docs
- User asks to document the pinglab library
- User mentions changes to pinglab and wants docs updated
- User asks to regenerate the API reference

## Documentation location

The documentation lives at:
```
biblio/docs/README.mdx
```

## Source code location

The pinglab library source is at:
```
pinglab/src/pinglab/
├── __init__.py              # Main exports
├── types.py                 # Pydantic models
├── run/
│   └── run_network.py       # Core simulation function
├── inputs/
│   ├── tonic.py             # Constant input with noise
│   ├── oscillating.py       # Sinusoidal input
│   └── pulse.py             # Pulse stimulus functions
├── analysis/
│   ├── mean_firing_rates.py # Population firing rates
│   ├── population_rate.py   # Time-binned rates
│   ├── rate_psd.py          # Power spectral density
│   ├── crosscorr.py         # E→I cross-correlation
│   └── population_isi_cv.py # ISI coefficient of variation
├── plots/
│   ├── raster.py            # Spike raster plots
│   ├── instrument.py        # Voltage/conductance traces
│   └── styles.py            # Light/dark theme styles
├── utils/
│   ├── slice_spikes.py      # Time window extraction
│   ├── expand_parameter_spec.py  # Parameter sweeps
│   └── load_config.py       # YAML config loading
└── multiprocessing/
    └── parallel.py          # Parallel execution
```

## Task workflow

1. **Read all source files** in `pinglab/src/pinglab/`
2. **Extract function signatures, parameters, and docstrings**
3. **Generate documentation** following the structure below
4. **Write to** `biblio/docs/README.mdx`

## Documentation structure

```mdx
---
title: Pinglab Docs
description: Documentation for the Pinglab python package.
date: {current-date}
---

Pinglab is a Python library for simulating conductance-based spiking neural networks...

## Contents
[Table of contents with anchor links]

---

## Quick Start
[Complete working example]

---

## Core Function
### `run_network`
[Signature, description, parameters, returns]

---

## Types
### `NetworkConfig`
[All fields with types and defaults]

### `Spikes`
### `NetworkResult`
### `InstrumentsConfig`
### `InstrumentsResults`
### `ExperimentConfig`

---

## Input Generation
### `inputs.tonic`
### `inputs.oscillating`
### `inputs.add_pulse_to_input`
### `inputs.compute_spike_delta`

---

## Analysis
### `analysis.mean_firing_rates`
### `analysis.population_rate`
### `analysis.rate_psd`
### `analysis.crosscorr`
### `analysis.population_isi_cv`

---

## Plotting
### `plots.save_raster`
### `plots.save_instrument_traces`
### `plots.styles`

---

## Utilities
### `utils.slice_spikes`
### `utils.expand_parameter_spec`
### `utils.load_config`

---

## Multiprocessing
### `multiprocessing.parallel`

---

## Example: Parameter Sweep
[Complete example]

---

## Example: Recording Instruments
[Complete example]

---

## Numerical Considerations
[Time step requirements, firing rate limits]
```

## Documentation format for each function

```mdx
### `module.function_name`

Brief description of what the function does.

```python
from pinglab.module import function_name

result = function_name(
    param1=value1,       # Description
    param2=value2,       # Description
)
# Returns: description
```

**Parameters:**
- `param1` (`type`): Description with default if applicable
- `param2` (`type`): Description

**Returns:** Description of return value
```

## Guidelines

- Use KaTeX math: `$\tau_{mem}$` for inline, `$$equation$$` for display
- Include type annotations in signatures
- Show default values in NetworkConfig
- Add inline comments explaining each parameter
- Include complete working examples
- Document return types and shapes for numpy arrays
- Keep the table of contents up to date
- Use consistent formatting throughout

## Key types to document thoroughly

### NetworkConfig fields
Document ALL fields with:
- Type
- Default value
- Units (ms, Hz, mV, nS, etc.)
- Brief description

### Spikes
- times: spike times in ms
- ids: neuron indices
- types: 0=E, 1=I
- populations: optional for multi-population

### InstrumentsConfig
- variables: list of ['V', 'g_e', 'g_i']
- neuron_ids: which neurons to record
- downsample: recording frequency
- population_means: whether to record population averages
