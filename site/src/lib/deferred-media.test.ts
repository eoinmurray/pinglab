import test from 'node:test';
import assert from 'node:assert/strict';
import { parse, type DefaultTreeAdapterMap } from 'parse5';
import { deferMedia } from './deferred-media.ts';

type Node = DefaultTreeAdapterMap['node'];
const all = (node: Node): DefaultTreeAdapterMap['element'][] => [
  ...('tagName' in node ? [node] : []), ...('childNodes' in node ? node.childNodes.flatMap(all) : []),
];
const attr = (node: DefaultTreeAdapterMap['element'], name: string) => node.attrs.find(attribute => attribute.name === name)?.value;

test('defers every fetchable image URL, including responsive pictures and cropped SVG images', () => {
  const html = deferMedia(`<!doctype html><html><head><title>Test</title></head><body>
    <img src="/plain.svg" alt="A &amp; B" width="640" height="480">
    <picture><source srcset="/wide.webp 2x" media="(min-width: 800px)"><img src="/small.png" srcset="/small2.png 2x" sizes="80vw" alt="Responsive"></picture>
    <svg viewBox="0 10 400 200" role="img"><image href="/crop.png" width="400" height="300"/></svg>
    <video src="/clip.mp4" poster="/poster.jpg" controls preload="none"><source src="/clip.webm" type="video/webm"></video>
    <img class="lightbox" alt=""><svg><use href="#icon"/></svg></body></html>`);
  const nodes = all(parse(html));
  assert.equal(nodes.filter(node => attr(node, 'data-deferred-media') !== undefined).length, 4);
  for (const node of nodes.filter(node => ['img', 'source', 'image', 'video'].includes(node.tagName))) {
    for (const name of ['src', 'srcset', 'href', 'poster']) assert.equal(attr(node, name), undefined);
  }
  const first = nodes.find(node => attr(node, 'data-media-src') === '/plain.svg')!;
  assert.equal(attr(first, 'width'), '640');
  assert.equal(attr(first, 'height'), '480');
  assert.equal(attr(first, 'alt'), 'A & B');
  assert.equal(attr(first, 'data-media-loading'), '');
  assert.match(attr(first, 'style')!, /--media-aspect: 640 \/ 480/);
  assert.equal(attr(nodes.find(node => attr(node, 'data-media-src') === '/small.png')!, 'data-media-loading'), '');
  assert.equal(attr(nodes.find(node => node.tagName === 'svg' && attr(node, 'viewBox'))!, 'viewBox'), '0 10 400 200');
  assert.equal(attr(nodes.find(node => node.tagName === 'video')!, 'preload'), 'none');
  assert.equal(nodes.filter(node => node.tagName === 'noscript').length, 4);
  // A JavaScript-disabled parser sees the intact originals in the fallbacks.
  const withoutScript = all(parse(html, { scriptingEnabled: false }));
  assert.equal(withoutScript.filter(node => attr(node, 'src') === '/plain.svg').length, 1);
  assert.equal(withoutScript.filter(node => attr(node, 'srcset') === '/wide.webp 2x').length, 1);
  assert.equal(withoutScript.filter(node => attr(node, 'href') === '/crop.png').length, 1);
  assert.equal(attr(withoutScript.find(node => attr(node, 'src') === '/plain.svg')!, 'data-media-loading'), undefined);
});

test('skeletons preserve authored styles and ignore invalid dimensions', () => {
  const nodes = all(parse(deferMedia(`<img src="/styled.png" width="800" height="400" style="object-fit: contain">
    <img src="/unknown.png"><img src="/invalid.png" width="Infinity" height="0">`)));
  assert.equal(attr(nodes.find(node => attr(node, 'data-media-src') === '/styled.png')!, 'style'), 'object-fit: contain;--media-aspect: 800 / 400');
  for (const source of ['/unknown.png', '/invalid.png']) {
    const image = nodes.find(node => attr(node, 'data-media-src') === source)!;
    assert.equal(attr(image, 'data-media-loading'), '');
    assert.equal(attr(image, 'style'), undefined);
  }
});

test('leaves scripts, templates, existing noscript fallbacks and resource-free pages intact', () => {
  const plain = '<!doctype html><html><body><p>Text only.</p></body></html>';
  assert.equal(deferMedia(plain), plain);
  const input = '<!doctype html><html><body><script>const text = \'<img src="/not-an-image">\';</script><template><img src="/template.png"></template><noscript><img src="/fallback.png"></noscript><img src="/real.png"></body></html>';
  const html = deferMedia(input);
  assert.match(html, /const text = '<img src="\/not-an-image">';/);
  assert.match(html, /<template><img src="\/template.png"><\/template>/);
  assert.match(html, /<noscript><img src="\/fallback.png"><\/noscript>/);
  assert.equal(deferMedia(html), html);
});
