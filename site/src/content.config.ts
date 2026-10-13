import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const articleSchema = z.object({
  title: z.string(),
  author: z.string(),
  description: z.string(),
  abstract: z.string().optional(),
  created_at: z.string().date(),
  updated_at: z.string().date(),
  collection: z.string(),
});

const reports = defineCollection({
  loader: glob({
    base: '../md/experiments',
    pattern: 'exp*/report.mdx',
    generateId: ({ data }) => String(data.id),
  }),
  schema: articleSchema.extend({
    id: z.string().regex(/^exp\d{3}$/),
    inputs: z.array(z.string().regex(/^exp\d{3}$/)).default([]),
    runs: z.record(z.string(), z.string()).optional(),
    tags: z.array(z.string()),
    thumbnail: z.string().optional(),
    thumbnailRun: z.string().optional(),
  }),
});

const notes = defineCollection({
  loader: glob({ base: '../md/notes', pattern: '**/*.mdx' }),
  schema: articleSchema.extend({ collection: z.literal('notes') }),
});

const slides = defineCollection({
  loader: glob({
    base: '../md/slides',
    pattern: '**/*.{md,mdx}',
    generateId: ({ entry }) => entry.replace(/\.mdx?$/, '').replace(/\/slides$/, ''),
  }),
  schema: z.object({
    title: z.string(), author: z.string(), description: z.string(),
    created_at: z.string().date(), updated_at: z.string().date(),
  }),
});

export const collections = { reports, notes, slides };
