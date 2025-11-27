# Core rules

- Dont start servers yourself, ask the user to do it and paste the output
- Any planning or implementation documents should be markdown files in './plan'
- Delete the plan dir when its not in use.
- Dont make git commits unless asked.
- use katex $ and $$ for math in markdown files.

# Content Directory Structure

The `content/` directory contains computational neuroscience experiments. Each experiment follows this pattern:

```
content/
├── .cathedral.json              # Auto-generated index (do not edit manually)
├── {experiment-name}/           # Kebab-case experiment directory
│   ├── README.md                # Experiment documentation (MDX format)
│   ├── config.yaml              # Experiment parameters
│   ├── experiment.py            # Python script to run experiment
│   ├── local.py                 # Optional local execution script
│   └── data/                    # Generated outputs
│       ├── {plot-name}_light.png    # Light theme version
│       └── {plot-name}_dark.png     # Dark theme version
```

## Conventions

- **Dual-theme plots**: Every visualization must have both `_light.png` and `_dark.png` versions
- **README.md structure**:
  - Theory/background section
  - Experimental design section with parameters
  - Results section with `<Gallery>` components
- **Gallery component**: Lists plot pairs with relative paths and captions
- **Math notation**: Use KaTeX syntax ($ for inline, $$ for display math)

## Example Gallery Usage

```jsx
<Gallery
  paths={[
    'experiment-name/data/plot1_light.png',
    'experiment-name/data/plot1_dark.png',
    'experiment-name/data/plot2_light.png',
    'experiment-name/data/plot2_dark.png'
  ]}
  caption="Description of what the plots show"
/>
```
