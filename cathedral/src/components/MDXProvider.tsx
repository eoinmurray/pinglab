import { MDXProvider } from '@mdx-js/react'
import { ReactNode } from 'react'
import { mdxComponents } from '@/components/mdx-components'
import 'katex/dist/katex.min.css'

export function MDXProviderWrapper({ children }: { children: ReactNode }) {
  return (
    <MDXProvider components={mdxComponents}>
      <div className="prose max-w-full prose-slate dark:prose-invert md:prose-base">
        {children}
      </div>
    </MDXProvider>
  )
}
