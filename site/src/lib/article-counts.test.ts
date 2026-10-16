import assert from 'node:assert/strict';
import test from 'node:test';
import { countArticle } from './article-counts.ts';

test('counts rendered prose and abstract, excluding generated math, tables, code and captions', () => {
  const html = `<h2>Results</h2><p>We measured <strong>three</strong> cells &amp; four trials.</p>
    <p>The <span class="katex"><math>duplicated math</math><span>duplicated math</span></span> response increased.</p>
    <figure><div><img><img></div><figcaption>Many caption words.</figcaption></figure>
    <figure class="report-table"><table><tr><td>Tabular data</td></tr></table><figcaption>Table caption.</figcaption></figure>
    <pre>Example code here.</pre><script>hidden script</script>`;
  assert.deepEqual(countArticle(html, 'An abstract'), { words: 11, figures: 1 });
});

test('omits inline metadata, references and appendices, resuming at peer main sections', () => {
  const html = `<h2>Meta</h2><p>Internal editorial notes.</p><h2>Results</h2><p>One finding.</p>
    <h2>References</h2><ol><li>Bibliographic words.</li></ol><h2>Methods</h2><p>Two steps.</p>
    <h2>Appendix A</h2><p>Supplementary words.</p><figure><img></figure>`;
  assert.deepEqual(countArticle(html), { words: 4, figures: 0 });
});

test('counts multipanel and unavailable figures once and separates adjacent prose blocks', () => {
  assert.deepEqual(countArticle('<p>First</p><p>Second</p><figure><figure><img></figure></figure><figure><p class="figure-unavailable">Unavailable</p></figure>'), { words: 2, figures: 2 });
});
