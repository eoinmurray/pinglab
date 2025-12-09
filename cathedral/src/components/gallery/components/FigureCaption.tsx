import { renderMathInText } from "../lib/render-math-in-text";

export function FigureCaption({ caption, label }: { caption?: string; label?: string }) {
  if (!caption && !label) return null;

  return (
    <div className="mx-auto max-w-md">
      <div className="text-sm text-muted-foreground leading-relaxed text-left">
        {label && <span className="font-medium text-foreground">{label}</span>}
        {label && caption && <span className="mx-1">—</span>}
        {caption && renderMathInText(caption)}
      </div>
    </div>
  );
}
