import { test } from 'node:test';
import assert from 'node:assert/strict';
import { retryRead } from './remote-http.ts';

test('retries interrupted reads but never accepts repeated failures or missing data', async () => {
  let calls = 0;
  assert.equal(await retryRead('catalog', async () => {
    if (++calls === 1) throw new TypeError('connection terminated');
    return 'complete';
  }), 'complete');
  assert.equal(calls, 2);
  calls = 0;
  await assert.rejects(retryRead('asset', async () => { calls++; throw new Error('checksum mismatch'); }), /asset: checksum mismatch/);
  assert.equal(calls, 3);
  calls = 0;
  await assert.rejects(retryRead('missing', async () => {
    calls++; throw Object.assign(new Error('absent'), { code: 'ENOENT' });
  }), /absent/);
  assert.equal(calls, 1);
});
