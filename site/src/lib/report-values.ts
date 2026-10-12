import { getEntry } from 'astro:content';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { latestPresentation } from './pingstore';

const jsonFiles = new Map<string, Promise<any>>();
async function json(report: string, experiment = report) {
  const entry = await getEntry('reports', report);
  const run = entry?.data.runs?.[experiment] ?? (await latestPresentation(experiment)).id;
  const key = `${run}/numbers.json`;
  if (!jsonFiles.has(key)) jsonFiles.set(key, readFile(resolve('../.pingstore/runs', run, 'export/numbers.json'), 'utf8').then(JSON.parse));
  return jsonFiles.get(key)!;
}
const round = (value: number, digits = 0) => Number(value.toFixed(digits));
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
const percent = (value: number) => `${round(100 * value, 1)}%`;
const joinLast = (items: unknown[]) => items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`;

async function context(report: string): Promise<Record<string, any>> {
  const r = await json(report, report === 'exp110' ? 'exp082' : report);
  const ctx: Record<string, any> = { run: r, r, r049: r, cfg: r.config, rounded: (v: number, digits = 1) => round(v, digits), number: (v: number) => round(v, 2) };
  if (report === 'exp037') {
    const point = (model: string, mode: string, level: number) => r.perturbation_summary.find((row: any) => row.model === model && row.mode === mode && Math.abs(row.level - level) < .001);
    const baseline = r.test_baselines.find((row: any) => row.model === 'ping' && row.seed === 42);
    const row = r.perturbation.find((row: any) => row.model === 'ping' && row.seed === 42 && row.mode === 'add' && row.level === 100);
    const counts = row.perturbation.populations.e1;
    Object.assign(ctx, { baseline, row, natural: counts.raw_spikes / counts.slots * 1000 / row.perturbation.dt_ms,
      acc: (model: string, mode: string, level: number) => round(point(model, mode, level).acc, 1),
      'reference-rate': (model: string) => mean(r.baseline_results.filter((row: any) => row.model === model && row.rate_target_hz == null).map((row: any) => row.rate_e)),
      eval_n: r.config.evaluation_samples_per_seed[0], add_max: Math.max(...r.perturbation_summary.filter((row: any) => row.mode === 'add').map((row: any) => row.level)), labels: r.illustrative_labels,
      'trial-description': r.illustrative_labels?.length && r.illustrative_labels.every((n: number) => n === r.illustrative_labels[0]) ? `the same test image of digit ${r.illustrative_labels[0]}` : 'the same test image',
    });
  }
  if (report === 'exp038') {
    const off = r.ei_sweep_summary.find((row: any) => Math.abs(row.ei_strength) < .001);
    const on = r.ei_sweep_summary.find((row: any) => Math.abs(row.ei_strength - 1) < .001);
    Object.assign(ctx, { eval_n: r.config.evaluation_samples_per_seed[0], eval_pool: r.config.evaluation_pool_samples, rate_off: round(off.hid_rate_hz), rate_on: round(on.hid_rate_hz), inhibition_on: round(on.inh_rate_hz), acc_cost: round(off.acc - on.acc),
      'image-description': r.illustrative_labels ? `the same digit-${r.illustrative_labels.ei_rasters[0]} test image` : 'the same test image' });
  }
  if (report === 'exp041') Object.assign(ctx, {
    fa: round(r.fit.a_affine, 2), fp: round(r.fit.p_affine, 3), fr: round(r.fit.r2_affine, 3), po: round(r.fit.p_origin, 3), pr: round(r.fit.r2_origin, 3),
    fg_lo: round(Math.min(...r.results.map((x: any) => x.f_gamma_hz))), fg_hi: round(Math.max(...r.results.map((x: any) => x.f_gamma_hz))),
    er_lo: round(Math.min(...r.results.map((x: any) => x.e_rate_hz)), 1), er_hi: round(Math.max(...r.results.map((x: any) => x.e_rate_hz)), 1),
    acc_lo: round(Math.min(...r.results.map((x: any) => x.acc))), acc_hi: round(Math.max(...r.results.map((x: any) => x.acc))),
  });
  if (report === 'exp044') Object.assign(ctx, { c: r.config.training_contract.common, cfg: r.recipe, summary: r.summary, n: r.recipe.seeds.length });
  if (report === 'exp049') Object.assign(ctx, { contrast_low: round(r.rhythmicity.final_contrast_trainable_min, 3), contrast_high: round(r.rhythmicity.final_contrast_trainable_max, 3), contrast_first: round(r.rhythmicity.epoch1_contrast_trainable, 3), eval_n: r.config.evaluation_samples });
  if (report === 'exp082' || report === 'exp110') {
    const accuracyMean = (duration: number, rate: number) => mean(r.grid_per_seed.filter((row: any) => row.duration_ms === duration && row.rate_hz === rate).map((row: any) => row.accuracy));
    Object.assign(ctx, { accuracy: (duration: number, rate: number) => percent(accuracyMean(duration, rate)), 'stream-accuracy': (duration: number, rate: number) => percent(accuracyMean(duration, rate)), 'stream-pct': percent,
      'stream-upper-means': r.config.psychometric_rates_hz.filter((rate: number) => rate >= 5).map((rate: number) => accuracyMean(200, rate)),
    });
  }
  return ctx;
}

// Only the authored expression vocabulary is interpreted; no JavaScript eval.
function evaluate(expression: string, ctx: Record<string, any>): any {
  expression = expression.trim();
  if (/^-?\d+(\.\d+)?$/.test(expression)) return Number(expression);
  if (/^".*"$/.test(expression)) return JSON.parse(expression);
  if (expression.startsWith('r.hero_stream.labels.map')) return joinLast(ctx.r.hero_stream.labels);
  if (expression.startsWith('r.showcase_selection.configuration.conditions.map')) return joinLast(ctx.r.showcase_selection.configuration.conditions.map((pair: number[]) => `(${pair[0]} ms, ${pair[1]} Hz)`));
  const extrema = /^calc\.(min|max)\(\.\.(.+)\)$/.exec(expression);
  if (extrema) return extrema[1] === 'min' ? Math.min(...evaluate(extrema[2], ctx)) : Math.max(...evaluate(extrema[2], ctx));
  const call = /^([\w-]+)\((.*)\)$/.exec(expression);
  if (call && typeof ctx[call[1]] === 'function') {
    const args: string[] = []; let depth = 0; let quote = false; let start = 0;
    for (let i = 0; i < call[2].length; i++) { const c = call[2][i]; if (c === '"') quote = !quote; if (!quote) { if (c === '(') depth++; if (c === ')') depth--; if (c === ',' && depth === 0) { args.push(call[2].slice(start, i)); start = i + 1; } } }
    args.push(call[2].slice(start));
    return ctx[call[1]](...args.map(arg => evaluate(arg, ctx)));
  }
  const segments = expression.split('.'); let result = ctx[segments.shift()!];
  for (const segment of segments) {
    if (segment === 'first()') result = result[0];
    else if (segment === 'sum()') result = result.reduce((sum: number, n: number) => sum + n, 0);
    else result = result?.[segment];
  }
  if (result === undefined || result === null) throw new Error(`Missing authored value: ${expression}`);
  return result;
}
export async function reportValue(report: string, expression: string): Promise<string> {
  const value = evaluate(expression, await context(report));
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error(`Nonfinite authored value: ${report}/${expression}`);
  return String(value);
}
export async function bindMarkup(markup: string, report: string): Promise<string> {
  const matches = [...markup.matchAll(/<span data-live="([^"]+)">[\s\S]*?<\/span>/g)];
  for (const match of matches) {
    const expression = match[1].replaceAll('&quot;', '"').replaceAll('&amp;', '&').replaceAll('&gt;', '>').replaceAll('&lt;', '<');
    const value = await reportValue(report, expression);
    markup = markup.replace(match[0], value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'));
  }
  return markup;
}
