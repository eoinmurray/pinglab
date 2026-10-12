import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const reports = defineCollection({
  loader: glob({
    base: '../experiments',
    pattern: 'exp*/report.mdx',
    generateId: ({ data }) => String(data.id),
  }),
  schema: z.object({
    id: z.string().regex(/^exp\d{3}$/),
    title: z.string(),
    author: z.string(),
    description: z.string(),
    abstract: z.string().optional(),
    runs: z.record(z.string(), z.string()).optional(),
    created_at: z.string().date(),
    updated_at: z.string().date(),
    collection: z.string(),
    tags: z.array(z.string()),
    thumbnail: z.string().optional(),
    thumbnailRun: z.string().optional(),
  }),
});

export const collections = { reports };
