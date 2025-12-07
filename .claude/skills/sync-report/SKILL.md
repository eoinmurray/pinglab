---
name: sync-report
description: Sync a README.mdx report or SLIDES.mdx presentation with its experiment.py code. For each image/gallery in the report, add methodology sections explaining what the image shows, how it was created, and what to expect.
license: MIT
---

This skill guides the synchronization of experiment reports (README.mdx) and slide presentations (SLIDES.mdx) with their underlying experiment code (experiment.py and related files).

## Rules
- Don't mention the python files in the report.
- No equations in titles
- Make sure all Galleries have titles and captions.

## When to use

- User asks to sync a report or slides with its experiment code
- User wants to add methodology sections to a report or slides
- User asks to document how figures/images were generated

## Report structure

Reports live in `biblio/{experiment-name}/` with this structure:
```
biblio/{experiment-name}/
├── README.mdx           # The report (MDX format)
├── SLIDES.mdx           # Slide presentation (MDX format)
├── experiment.py        # Main experiment runner
├── config.yaml          # Experiment parameters
├── local/               # Local modules
│   ├── model.py         # Pydantic config model
│   └── experiment_*.py  # Individual experiment functions
└── data/                # Generated outputs (images, etc.)
```

## Task workflow

1. **Read the README.mdx or SLIDES.mdx** to identify all Gallery components and their glob patterns
2. **Read experiment.py** to understand the experiment structure
3. **Read config.yaml** to get actual parameter values
4. **Read local/experiment_*.py** files to understand how each figure is generated
5. **For each Gallery**, add methodology sections (for README) or speaker notes (for SLIDES):

## Methodology section format

For each `<Gallery>` component in the report, add three subsections immediately before it:

```mdx
### What this shows

[1-2 sentences describing what the visualization represents and why it matters]

### How we created it

**Experiment N** (`experiment_N.py`): [Description of the methodology]
- Key parameters from config.yaml (use actual values)
- Analysis steps performed
- Any processing applied (slicing, smoothing, etc.)

### What we expect to see

[2-3 sentences describing expected results and their interpretation]

<Gallery ... />
```

## Slides format (SLIDES.mdx)

Slides use `---` as separators between slides. Each slide typically contains a single `<Gallery>` component. For slides, instead of adding methodology sections, ensure each Gallery has:

- **title**: What the visualization shows (concise, one line)
- **subtitle**: Key context or regime information
- **caption**: Brief methodology summary
- **captionLabel**: Figure numbering (e.g., "Figure 1")

### Slide structure

```mdx
---

<Gallery
  title="Raster plot of A-OSC transition as $g_{ei}$ increases."
  path="experiment/data"
  globs={["raster_g_ei_*.png"]}
  subtitle="At $I_E = 1.0$ nA in the oscillatory regime."
  caption="Sweep $g_{ei}$ from 0.8 to 2.4 nS with fixed $I_E = 1.0$ nA"
  captionLabel="Figure 1"
/>

---
```

### Slides-specific guidelines

- Keep titles under 80 characters
- Subtitles should provide key parameter context
- Captions summarize the methodology in one sentence
- Use `single` prop for galleries that should show one image at a time
- Ensure all parameter values match config.yaml

## Guidelines

- Use KaTeX math notation: `$I_E$` for inline, `$$equation$$` for display
- Reference actual parameter values from config.yaml
- Link experiment numbers to their source files
- Explain the scientific significance, not just the mechanics
- Keep descriptions concise but complete
- Fix any typos in existing captions while editing
- Update incorrect captions to match actual experiment outputs

## Example transformation

Before:
```mdx
## F-I curves

<Gallery
  path="experiment/data"
  globs={["firing_rates*.png"]}
  caption="F-I curves for E and I populations."
/>
```

After:
```mdx
## F-I curves

### What this shows

Frequency-input (F-I) curves characterizing the mean firing rate of E and I populations as a function of external drive $I_E$.

### How we created it

**Experiment 3** (`experiment_3.py`): We sweep $I_E$ from 0 to 100 (40 values) with fixed $g_{ei} = 1.4$. For each simulation, we compute the mean firing rate over the full 1000 ms window.

### What we expect to see

Both populations should show monotonically increasing firing rates. The E population receives direct input so its curve should be steeper.

<Gallery
  path="experiment/data"
  globs={["firing_rates*.png"]}
  caption="F-I curves for E and I populations."
/>
```

## Common analyses to document

- **Raster plots**: Spike times, population separation, time window
- **F-I curves**: Parameter sweep, rate calculation method
- **PSD**: Bin size, smoothing, frequency range, normalization
- **Cross-correlogram**: Pair selection, bin width, lag range
- **ISI CV**: Per-neuron calculation, population averaging
- **Phase analysis**: LFP proxy, phase extraction method
