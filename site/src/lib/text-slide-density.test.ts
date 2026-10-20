import { test } from 'node:test';
import assert from 'node:assert/strict';
import { textSlideDensity } from './text-slide-density.ts';

test('short lists get roomier typography; text length matters as well as item count', () => {
  assert.equal(textSlideDensity('<ul><li>Dominance</li><li>Prestige</li></ul>'), 'sparse');
  assert.equal(textSlideDensity(`<ul>${'<li>A longer explanatory sentence about social behaviour and identity.</li>'.repeat(5)}</ul>`), 'medium');
  assert.equal(textSlideDensity(`<ul><li>${'Long explanation. '.repeat(40)}</li></ul>`), 'dense');
  assert.equal(textSlideDensity(`<ol>${'<li>Point</li>'.repeat(9)}</ol>`), 'dense');
});

test('tables and other embedded content retain their existing typography', () => {
  for (const html of ['<div><table><tr><td>Short</td></tr></table></div>', '<p>Intro</p><img src="figure.png">', '<pre>code</pre>', '<svg></svg>']) {
    assert.equal(textSlideDensity(html), 'dense');
  }
});

test('counts visible text, list items and paragraphs, not markup or comments', () => {
  assert.equal(textSlideDensity(`<p><strong>Brief</strong> &amp; clear.</p><!-- ${'ignored '.repeat(100)} -->`), 'sparse');
  assert.equal(textSlideDensity('<p></p>'), 'dense');
  assert.equal(textSlideDensity('<p>Point</p>'.repeat(9)), 'dense');
});
