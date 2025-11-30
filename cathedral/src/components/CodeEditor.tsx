import type { BundledLanguage } from '@/components/ui/shadcn-io/code-block';
import {
  CodeBlock,
  CodeBlockBody,
  CodeBlockContent,
  CodeBlockItem,
} from '@/components/ui/shadcn-io/code-block';

function expandFileType(fileType: string): BundledLanguage {
  switch (fileType) {
    case 'py':
      return 'python';
    case 'yaml':
      return 'yaml';
    case 'json':
      return 'json';
    default:
      return 'plaintext';
  }
}

export function CodeEditor ({ code, fileType }: { code: string; fileType: string }) {
  const data = [
    {
      language: expandFileType(fileType),
      filename: 'MyComponent.jsx',
      code: code,
    }
  ];

  return (
    <CodeBlock data={data} defaultValue={data[0].language}>
      <CodeBlockBody>
        {(item) => (
          <CodeBlockItem key={item.language} value={item.language}>
            <CodeBlockContent language={item.language as BundledLanguage}>
              {item.code}
            </CodeBlockContent>
          </CodeBlockItem>
        )}
      </CodeBlockBody>
    </CodeBlock>
  )
};