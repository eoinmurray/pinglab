import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { getPapers, readPaperIndex, getPaperCollections, paperCollectionLabel, paperCollectionUrl } from './papers.ts';

test('joins metadata to nested PDFs while preserving unindexed files and ignoring other files', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paper-library-'));
  try {
    await mkdir(join(directory, 'nested'));
    await writeFile(join(directory, 'nested', 'a paper.pdf'), '%PDF-1.7');
    await writeFile(join(directory, 'other.PDF'), '%PDF-1.7');
    await writeFile(join(directory, '.hidden.pdf'), '%PDF-1.7');
    await writeFile(join(directory, 'notes.txt'), 'not a PDF');
    await writeFile(join(directory, 'index.yaml'), `nested/a paper.pdf:
  id: P01
  title: A title
  authors: [A. Author, B. Author]
  year: 2020
`);
    const papers = await getPapers(directory);
    assert.equal(papers.length, 2);
    assert.equal(papers[0].metadata?.title, 'A title');
    assert.deepEqual(papers[0].metadata?.authors, ['A. Author', 'B. Author']);
    assert.equal(papers[0].href, '/papers/nested/a%20paper.pdf');
    assert.equal(papers[1].metadata, undefined);
    assert.deepEqual(papers[1].collections, ['uncategorized']);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('supports multiple collections, derives navigation, and rejects invalid memberships', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paper-collections-'));
  const entry = (collection: string, id = 'P01') => `id: ${id}\ntitle: A title\nauthors: [A. Author]\nyear: 2026\ncollection: ${collection}\n`;
  try {
    for (const file of ['a.pdf', 'b.pdf', 'c.pdf']) await writeFile(join(directory, file), '%PDF-1.7');
    await writeFile(join(directory, 'index.yaml'), ['a.pdf:\n' + entry('[gamma, sampling]').replace(/^/gm, '  '),
      'b.pdf:\n' + entry('gamma', 'P02').replace(/^/gm, '  ')].join('\n'));
    const papers = await getPapers(directory);
    assert.deepEqual(papers[0].collections, ['gamma', 'sampling']);
    assert.deepEqual(getPaperCollections(papers), ['gamma', 'sampling', 'uncategorized']);
    assert.equal(papers.filter(paper => paper.collections.includes('gamma')).length, 2);
    assert.equal(paperCollectionLabel('probabilistic-coding'), 'Probabilistic coding');
    assert.equal(paperCollectionLabel('ping'), 'PING');
    assert.equal(paperCollectionUrl('gamma'), '/papers/collections/gamma');
    for (const invalid of ['[]', '[gamma, gamma]', '"Bad slug"', '42', 'null']) {
      await writeFile(join(directory, 'index.yaml'), 'a.pdf:\n' + entry(invalid).replace(/^/gm, '  '));
      await assert.rejects(readPaperIndex(directory), /collection must be/);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('an absent index is allowed; malformed metadata is rejected', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paper-index-'));
  try {
    assert.deepEqual(await readPaperIndex(directory), {});
    await writeFile(join(directory, 'index.yaml'), 'paper.pdf:\n  title: Only a title\n');
    await assert.rejects(readPaperIndex(directory), /Invalid.*paper.pdf/);
    await writeFile(join(directory, 'index.yaml'), '- not a mapping\n');
    await assert.rejects(readPaperIndex(directory), /filename-to-metadata mapping/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('warns about metadata whose PDF is missing', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'paper-missing-'));
  const warning = t.mock.method(console, 'warn', () => {});
  try {
    await writeFile(join(directory, 'index.yaml'), `missing.pdf:
  id: P01
  title: Missing paper
  authors: [A. Author]
  year: 2020
`);
    assert.deepEqual(await getPapers(directory), []);
    assert.match(warning.mock.calls[0].arguments[0], /Metadata has no matching PDF: missing.pdf/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('sorts newest years first, titles within years, and unindexed PDFs last', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paper-order-'));
  try {
    for (const file of ['a.pdf', 'b.pdf', 'c.pdf', 'd.pdf', '0.pdf']) {
      await writeFile(join(directory, file), '%PDF-1.7');
    }
    await writeFile(join(directory, 'index.yaml'), [
      ['a.pdf', 'Older', 2020], ['b.pdf', 'Zebra', 2025],
      ['c.pdf', 'Alpha', 2025], ['d.pdf', 'Newest', 2026],
    ].map(([file, title, year], i) => `${file}:\n  id: P0${i + 1}\n  title: ${title}\n  authors: [A. Author]\n  year: ${year}\n`).join(''));
    assert.deepEqual((await getPapers(directory)).map(paper => paper.file),
      ['d.pdf', 'c.pdf', 'b.pdf', 'a.pdf', '0.pdf']);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('requires unique permanent paper IDs and preserves them independently of year ordering', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paper-ids-'));
  const entry = (file: string, id: string, year: number) => `${file}:\n  id: ${id}\n  title: A paper\n  authors: [A. Author]\n  year: ${year}\n`;
  try {
    for (const file of ['old.pdf', 'new.pdf']) await writeFile(join(directory, file), '%PDF-1.7');
    await writeFile(join(directory, 'index.yaml'), entry('old.pdf', 'P17', 1990) + entry('new.pdf', 'P02', 2026));
    assert.deepEqual((await getPapers(directory)).map(paper => paper.metadata?.id), ['P02', 'P17']);
    await writeFile(join(directory, 'index.yaml'), entry('old.pdf', 'P01', 1990) + entry('new.pdf', 'P01', 2026));
    await assert.rejects(readPaperIndex(directory), /Duplicate paper ID P01/);
    for (const id of ['P00', 'P1', 'P001', 'p01', '17', '']) {
      await writeFile(join(directory, 'index.yaml'), entry('old.pdf', id, 1990));
      await assert.rejects(readPaperIndex(directory), /P01-style id/);
    }
    await writeFile(join(directory, 'index.yaml'), entry('old.pdf', 'P100', 1990));
    assert.equal((await readPaperIndex(directory))['old.pdf'].id, 'P100');
  } finally { await rm(directory, { recursive: true, force: true }); }
});
