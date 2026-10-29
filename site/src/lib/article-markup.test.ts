import test from 'node:test';
import assert from 'node:assert/strict';
import { validateArticleMarkup } from './article-markup.ts';

test('rejects the exp157 failure: a flow expression returning text between blocks', () => {
  assert.throws(() => validateArticleMarkup(`<d-article>
    <h3>Sequential receiver following</h3>
    C sustains 57.44 Hz gamma while A and B remain inactive.
    <p>The B-driven receiver retains visible phase modulation.</p>
  </d-article>`, '/exp157'), error => {
    assert.match(error.message, /Article layout error at \/exp157/);
    assert.match(error.message, /C sustains 57\.44 Hz gamma/);
    assert.match(error.message, /must render a <p>/);
    return true;
  });
});

test('validates appendix grids, including the display:contents report wrapper', () => {
  for (const html of [
    '<d-appendix><h3>Parameters</h3>Unwrapped result</d-appendix>',
    '<d-appendix><div class="other report-appendix" data-heading-depth="2"><h2>Methods</h2>Unwrapped result</div></d-appendix>',
  ]) assert.throws(() => validateArticleMarkup(html, '/notes/example'), /unwrapped text/);
});

test('accepts paragraphs, inline bindings, rich text, components and empty conditionals', () => {
  assert.doesNotThrow(() => validateArticleMarkup(`<d-article>
    <!-- empty conditional --> <span id="equation-1"></span>
    <p>C sustains 57.44 Hz gamma with <strong>A</strong> inactive.</p>
    <section><h2>Abstract</h2><p>Text.</p></section>
    <figure><img src="/figure.svg"><figcaption>Figure 1.</figcaption></figure>
    <div class="source-table"><table><tbody><tr><td>57.44</td></tr></tbody></table></div>
    <pre><code>const text = 'a &lt; b';</code></pre>
  </d-article><d-appendix>
    <div class="report-appendix"><h2>Methods</h2><p>Bound value: 2.</p></div>
    <h3>Dataset</h3><section>Section contents have normal block layout.</section>
  </d-appendix>`, '/exp157'));
});

test('ignores non-article text and inert markup', () => {
  assert.doesNotThrow(() => validateArticleMarkup(`<!doctype html><html><body>
    Text outside the article grid.
    <script>const example = '<d-article>Example</d-article>';</script>
    <template><d-article>Example</d-article></template>
    <noscript><d-article>Example</d-article></noscript>
    <d-article><p>Valid prose.</p></d-article>
  </body></html>`, '/'));
});

test('checks every article and appendix rather than only the first grid', () => {
  assert.throws(() => validateArticleMarkup('<d-article><p>Valid</p></d-article><d-appendix>42</d-appendix>', '/exp157'), /unwrapped text in <d-appendix>/);
});
