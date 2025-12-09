import { useState, useEffect } from 'react'
import { compile, run } from '@mdx-js/mdx'
import * as runtime from 'react/jsx-runtime'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { mdxComponents } from '@/components/mdx-components'
import { cn } from '@/lib/utils'

export function RuntimeMDX({ 
  content, 
  size 
}: { 
  content: string; size?: "sm" | "md" | "lg" | "xl" | "2xl";
}) {
  const [MDXContent, setMDXContent] = useState<React.ComponentType<{ components: typeof mdxComponents }> | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false

    async function compileMDX() {
      try {
        const compiled = await compile(content, {
          outputFormat: 'function-body',
          remarkPlugins: [remarkGfm, remarkMath],
          rehypePlugins: [rehypeKatex as never],
        })

        const mod = await run(compiled, {
          ...runtime,
          baseUrl: import.meta.url,
        })

        if (!cancelled) {
          setMDXContent(() => mod.default)
          setError(null)
        }
      } catch (err) {
        console.error('MDX compilation error:', err)
        if (!cancelled) {
          setError(err as Error)
        }
      }
    }

    compileMDX()

    return () => {
      cancelled = true
    }
  }, [content])

  if (error) {
    return (
      <div className="text-destructive p-4 border border-destructive/30 rounded-lg bg-destructive/5">
        <h3 className="font-semibold">MDX Compilation Error</h3>
        <pre className="text-sm mt-2 overflow-x-auto font-mono">{error.message}</pre>
      </div>
    )
  }

  if (!MDXContent) {
    return <div className="text-muted-foreground">Loading content...</div>
  }

  return (
    <div className={cn(
      "prose prose-slate dark:prose-invert",
      size === "sm" ? "prose-sm" :
      size === "md" ? "prose-md" :
      size === "lg" ? "prose-lg" :
      size === "xl" ? "prose-xl" :
      size === "2xl" ? "prose-2xl" : "prose-md"

    )}>
      <MDXContent components={mdxComponents} />
    </div>
  )
}
