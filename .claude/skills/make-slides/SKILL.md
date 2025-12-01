---
name: make-slides
description: Convert a README.mdx report into a SLIDES.mdx presentation. Creates markdown slides split by '---' with concise content suitable for presentations.
license: MIT
---

This skill converts experiment reports (README.mdx) into slide presentations (SLIDES.mdx).

## Rules

- Use '---' to separate slides
- No equations in slide titles
- Keep bullet points concise (1 line each)
- One main idea per slide
- Gallery components go on their own "Results" slides
- No frontmatter in slides

## When to use

- User asks to make slides from a report
- User wants to create a presentation from README.mdx
- User asks to convert a report to SLIDES.mdx

## Output structure

Slides are written to `biblio/{experiment-name}/SLIDES.mdx` with this format:

```mdx

# Title Slide

---

## Section Title

[Concise content]

---
```

## Task workflow

1. **Read the README.mdx** to understand the report structure
2. **Create SLIDES.mdx** in the same directory
3. **Convert each section** into multiple digestible slides
4. **Add summary and conclusion slides** at the end

## Slide structure per section

For each major section in the README, create 2-3 slides:

### Slide 1: Introduction
```mdx
## [Section Title]

[1-2 sentence description of what this analysis shows]

- Key point 1
- Key point 2

**Parameters:** [Key parameters in compact form]
```

### Slide 2: Results
```mdx
## [Section Title]: Results

<Gallery
  path="..."
  globs={["..."]}
  caption="..."
/>
```

### Slide 3: Interpretation (if needed)
```mdx
## [Section Title]: Interpretation

**[Condition 1]:** [Result]

**[Condition 2]:** [Result]

- Bullet point insight
- Another insight
```

## Guidelines

- Use KaTeX math notation: `$I_E$` for inline
- Keep slides scannable - avoid paragraphs
- Use bold for emphasis on key terms
- Tables work well for summary slides
- Aim for 15-25 slides total
- Each slide should be readable in ~30 seconds

## Example transformation

README section:
```mdx
## F-I curves for E and I populations.

### What this shows

Frequency-input (F-I) curves characterizing the mean firing rate...

### How we created it

We run two sweeps with fixed $g_{ei} = 1.4$:
- **Large range**: $I_E$ from 0 to 100 (40 values)
- **Small range**: $I_E$ from 0 to 2 (40 values)

### What we expect to see

Both populations should show monotonically increasing firing rates...

<Gallery ... />
```

Becomes slides:
```mdx
---

## F-I Curves

Frequency-input curves characterizing mean firing rate vs. external drive $I_E$.

**Two sweeps with $g_{ei} = 1.4$:**

- Large range: $I_E$ from 0 to 100 (40 values)
- Small range: $I_E$ from 0 to 2 (40 values)

---

## F-I Curves: Results

<Gallery ... />

---

## F-I Curves: Interpretation

- Both populations show monotonically increasing firing rates
- E population: steeper curve (direct external input)
- I population: follows E but with lower initial rates
- High $I_E$: saturation due to refractory periods
```

## Summary slide template

```mdx
---

## Summary

| Metric | Condition A | Condition B |
|--------|-------------|-------------|
| Metric 1 | Result | Result |
| Metric 2 | Result | Result |
```

## Conclusion slide template

```mdx
---

## Conclusions

1. Main finding 1
2. Main finding 2
3. Main finding 3
4. Main finding 4
```
