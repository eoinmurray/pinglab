import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { experimental_AstroContainer } from 'astro/container';
import { parseFragment } from 'parse5';

// Compile just this component: tests do not load articles or their run artifacts.
const require = createRequire(import.meta.resolve('astro'));
const { transform } = require('@astrojs/compiler-rs');
const source = await readFile(new URL('../components/Table.astro', import.meta.url), 'utf8');
const compiled = transform(source, {
  filename: 'Table.astro', internalURL: import.meta.resolve('astro/compiler-runtime'), resultScopedSlot: true,
  resolvePath: specifier => specifier,
});
assert.deepEqual(compiled.diagnostics.filter(diagnostic => diagnostic.severity === 'error'), []);
const directory = await mkdtemp(join(tmpdir(), 'pinglab-table-'));
after(() => rm(directory, { recursive: true, force: true }));
const modulePath = join(directory, 'Table.mjs');
await writeFile(modulePath, compiled.code
  .replace(/^import "[^"\n]+\?astro&type=style[^"\n]*";\n/gm, '')
  .replace(/from "([^"\n]+)"/g, (_, specifier) => `from "${import.meta.resolve(specifier)}"`));
const { default: Table } = await import(pathToFileURL(modulePath).href);
const render = async (props, slots = {}) => {
  const container = await experimental_AstroContainer.create();
  return container.renderToString(Table, { props, slots });
};
const find = (node, tag) => [
  ...(node.tagName === tag ? [node] : []),
  ...(node.childNodes ?? []).flatMap(child => find(child, tag)),
];
const attr = (node, name) => node.attrs.find(attribute => attribute.name === name)?.value;

test('flat tables retain escaping, rich cells, captions and numbering', async () => {
  const html = await render({
    headers: ['Parameter', 'Value'],
    rows: [['<script>unsafe</script>', 42], [{ markdown: '**Leak**' }, { math: 'g_L' }]],
    caption: 'Settings', number: 'A1', id: 'settings', label: 'Model settings',
  });
  const tree = parseFragment(html);
  assert.equal(find(tree, 'tbody').length, 1);
  assert.equal(find(tree, 'th').length, 2);
  assert.equal(find(tree, 'td').length, 4);
  assert.match(html, /&lt;script&gt;unsafe&lt;\/script&gt;/);
  assert.match(html, /<strong>Leak<\/strong>/);
  assert.match(html, /class="katex"/);
  assert.match(html, /Table A1\./);
  assert.match(html, /aria-label="Model settings"/);
  assert.equal(find(tree, 'figcaption').length, 1);
});

test('group headings span all comparison columns above their rows and support rich labels', async () => {
  const html = await render({
    headers: ['Parameter', 'COBA', 'PING'],
    groups: [
      { label: 'Input and simulation', rows: [
        ['Encoding', 'Poisson', 'Poisson'], ['Input encoding rate', '25 Hz', '25 Hz'],
      ] },
      { label: { markdown: '**Custom analysis**' }, rows: [['Window', '50–200 ms', '50–200 ms']] },
    ],
  }, { default: '<em>Paired settings.</em>' });
  const tree = parseFragment(html);
  const bodies = find(tree, 'tbody');
  assert.equal(bodies.length, 2);
  assert.equal(find(tree, 'thead')[0].childNodes.filter(node => node.tagName === 'tr').length, 1);
  assert.equal(find(find(tree, 'thead')[0], 'th').length, 3);
  for (const [body, count] of [[bodies[0], 2], [bodies[1], 1]]) {
    const heading = find(body, 'th')[0];
    assert.equal(attr(heading, 'scope'), 'rowgroup');
    assert.equal(attr(heading, 'colspan'), '3');
    assert.equal(attr(heading, 'rowspan'), undefined);
    assert.equal(find(body, 'tr')[0].childNodes.filter(node => node.tagName === 'th')[0], heading);
    assert.equal(find(find(body, 'tr')[0], 'td').length, 0);
    assert.equal(find(body, 'tr').length, count + 1);
    assert.equal(find(body, 'th').length, 1);
    assert.equal(find(body, 'td').length, count * 3);
  }
  assert.match(html, /<strong>Custom analysis<\/strong>/);
  assert.match(html, /<em>Paired settings\.<\/em>/);
});

test('two-column groups use full-width headings without adding a column, and escape plain labels', async () => {
  const html = await render({ headers: ['Parameter', 'Value'], groups: [
    { label: '<Network>', rows: [['Units', 10]] },
  ] });
  const tree = parseFragment(html);
  assert.equal(find(find(tree, 'thead')[0], 'th').length, 2);
  assert.equal(attr(find(find(tree, 'tbody')[0], 'th')[0], 'colspan'), '2');
  assert.match(html, /&lt;Network&gt;/);
});

test('malformed groups, ambiguous input, invalid cells and inconsistent widths are rejected', async () => {
  const headers = ['Parameter', 'Value'];
  for (const props of [
    { groups: [] }, { groups: [null] }, { groups: [{ rows: [['Units', 10]] }] },
    { groups: [{ label: 'Network', rows: [] }] },
    { groups: [{ label: 'Network', rows: [['Units']] }] },
    { groups: [{ label: 'Network', rows: [['Units', Infinity]] }] },
    { groups: [{ label: 'Network', rows: [['Units', 10]] }], rows: [['Units', 10]] },
    { rows: [['Units']] }, { rows: [] }, {},
  ]) {
    await assert.rejects(() => render({ headers, ...props }), /Table requires/);
  }
});
