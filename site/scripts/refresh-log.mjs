import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const path = fileURLToPath(new URL('../node_modules/.cache/pinglab/dev-refresh.jsonl', import.meta.url));
if (!existsSync(path)) {
  console.error('No refresh diagnostics yet. Start the development server and open a page first.');
  process.exitCode = 1;
} else {
  const lines = [`${path}.1`, path].filter(existsSync)
    .flatMap(file => readFileSync(file, 'utf8').trim().split('\n')).filter(Boolean);
  console.log(lines.slice(-200).join('\n'));
}
