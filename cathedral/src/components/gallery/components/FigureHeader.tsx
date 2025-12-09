import { renderMathInText } from "../lib/render-math-in-text";

export function FigureHeader({ title, subtitle }: { title?: string; subtitle?: string }) {
  if (!title && !subtitle) return null;

  return (
    <div className="mx-auto max-w-md">
      {title && (
        <h3 className="text-sm md:text-base font-medium tracking-tight text-foreground text-left">
          {renderMathInText(title)}
        </h3>
      )}
      {subtitle && (
        <p className="text-sm text-muted-foreground leading-relaxed text-left mt-1">
          {renderMathInText(subtitle)}
        </p>
      )}
    </div>
  );
}
