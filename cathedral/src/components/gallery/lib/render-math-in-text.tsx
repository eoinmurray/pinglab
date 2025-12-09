import { ReactNode } from "react";
import katex from "katex";

export function renderMathInText(text: string): ReactNode {
  // Match $...$ for inline math and $$...$$ for display math
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  // Match display math ($$...$$) first, then inline math ($...$)
  const regex = /\$\$([^$]+)\$\$|\$([^$]+)\$/g;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    // Add text before this match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const isDisplay = match[1] !== undefined;
    const mathContent = match[1] || match[2];

    try {
      const html = katex.renderToString(mathContent, {
        displayMode: isDisplay,
        throwOnError: false,
      });
      parts.push(
        <span
          key={key++}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    } catch {
      // If KaTeX fails, just show the original text
      parts.push(match[0]);
    }

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 && typeof parts[0] === 'string' ? parts[0] : <>{parts}</>;
}
