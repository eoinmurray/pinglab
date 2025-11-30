import { useState, useEffect } from 'react'
import { compile, run } from '@mdx-js/mdx'
import * as runtime from 'react/jsx-runtime'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { mdxComponents } from '@/lib/mdx-components'

export function RuntimeMDX({ content }: { content: string }) {
  const [MDXContent, setMDXContent] = useState<React.ComponentType<{ components: typeof mdxComponents }> | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false

    async function compileMDX() {
      try {
        const compiled = await compile(content, {
          outputFormat: 'function-body',
          remarkPlugins: [remarkMath],
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
    <div className="prose max-w-full prose-slate dark:prose-invert md:prose-base">
      <MDXContent components={mdxComponents} />
    </div>
  )
}
