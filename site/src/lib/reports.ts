import { getCollection } from 'astro:content';
import { access } from 'node:fs/promises';
import { resolve } from 'node:path';

export async function getReports() {
  const reports = await getCollection('reports');
  // A running content loader can retain entries after their source is removed.
  const existing = await Promise.all(reports.map(async report => {
    try {
      await access(resolve('../experiments', report.id, 'report/report.mdx'));
      return report;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw error;
    }
  }));
  return existing.filter(report => report !== undefined);
}
