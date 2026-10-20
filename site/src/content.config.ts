import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const collectionId = z.string().regex(/^[a-z0-9][a-z0-9-]*$/);
const collectionSchema = z.union([collectionId, z.array(collectionId).min(1)])
  .refine(value => typeof value === 'string' || new Set(value).size === value.length, 'Collections must be unique');

const articleSchema = z.object({
  title: z.string(),
  author: z.string(),
  description: z.string(),
  abstract: z.string().optional(),
  created_at: z.union([z.string().datetime({ precision: 0 }), z.string().date()]),
  updated_at: z.string().datetime({ precision: 0 }),
  collection: collectionSchema,
});

const reports = defineCollection({
  loader: glob({
    base: '../experiments',
    pattern: 'exp*/report/report.mdx',
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
  loader: glob({ base: '../md/notes', pattern: '**/*.{md,mdx}' }),
  schema: articleSchema.extend({
    collection: collectionSchema.refine(value => typeof value === 'string' ? value === 'notes' : value.includes('notes'), 'Notes must include the notes collection'),
  }),
});

const slides = defineCollection({
  loader: glob({
    base: '../md/slides',
    pattern: '**/*.{md,mdx}',
    generateId: ({ entry }) => entry.replace(/\.mdx?$/, '').replace(/\/slides$/, ''),
  }),
  schema: z.object({
    title: z.string(), author: z.string(), description: z.string(),
    collection: collectionSchema.default(['slides']),
    created_at: z.string().date(), updated_at: z.string().date(),
  }),
});

export const collections = { reports, notes, slides };
