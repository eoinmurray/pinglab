import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import { NOTEBOOK_STATUSES } from "./config/status";

// Shared frontmatter schema for articles and notebooks. The entry's filename
// (ar009, nb058) is its id/slug; `entry` is the zero-padded display number.
const entrySchema = z.object({
  title: z.string(),
  date: z.coerce.date(),
  entry: z.number(),
  structure: z.string().optional(),
  collection: z.string().optional(),
  status: z.enum(NOTEBOOK_STATUSES).optional(),
  description: z.string().optional(),
  // opt out of the auto table-of-contents sidebar (content column then centers)
  hideSidebar: z.boolean().optional(),
});

const articles = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/articles" }),
  schema: entrySchema,
});

const notebooks = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/notebooks" }),
  schema: entrySchema,
});

export const collections = { articles, notebooks };
