# Distill v2

The unmodified browser runtime is downloaded from
https://distill.pub/template.v2.js on 2026-10-10.

The seven unmodified page CSS files in `src/styles/distill/` come from
https://github.com/distillpub/template at commit
`bf862cc4fe8abd56a94759453801fd04529fad9b`. They are concatenated in the
upstream `src/styles/styles.js` order and emitted before content renders,
using Distill's `distill-prerendered-styles` marker.

The appendix CSS is extracted unchanged from the same revision’s
`src/components/d-appendix.js` and loaded alongside the page styles. Dataset
metadata uses the native `d-appendix` footer.

The runtime is served locally. `LICENSE` and `AUTHORS` are retained here.
Distill is licensed under Apache-2.0.

Astro renders MDX inside native `d-title`, `d-byline`, and `d-article` elements.
Figures use the native `l-body` grid span to match the prose column. The body carries the official
`distill-prerendered` attribute because Astro supplies the complete initial HTML.

The byline retains Distill's grid markup and CSS but shows the real author,
created/updated dates, and collection instead of invented journal affiliations
or DOI fields. Pinglab supplies its own navigation. KaTeX renders math at build
time; the runtime does not re-render those equations.
