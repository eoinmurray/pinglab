import { useState, ImgHTMLAttributes } from "react";
import { Image } from "lucide-react";
import { cn } from "@/lib/utils";

export function LoadingImage({
  className,
  wrapperClassName,
  ...props
}: ImgHTMLAttributes<HTMLImageElement> & { wrapperClassName?: string }) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  return (
    <div className={cn("relative", wrapperClassName)}>
      {isLoading && !hasError && (
        <div className="absolute inset-0 bg-muted/30 animate-pulse flex items-center justify-center">
          <div className="w-8 h-8 border border-border/50 rounded-sm" />
        </div>
      )}
      {hasError && (
        <div className="absolute inset-0 bg-muted/20 flex items-center justify-center">
          <div className="text-center">
            <Image className="h-5 w-5 text-muted-foreground/40 mx-auto" />
            <span className="text-xs text-muted-foreground/40 mt-1.5 block font-mono">failed</span>
          </div>
        </div>
      )}
      <img
        {...props}
        className={cn(
          className,
          "transition-opacity duration-500 ease-out-expo",
          isLoading && "opacity-0",
          hasError && "opacity-0"
        )}
        onLoad={(e) => {
          setIsLoading(false);
          props.onLoad?.(e);
        }}
        onError={(e) => {
          setIsLoading(false);
          setHasError(true);
          props.onError?.(e);
        }}
      />
    </div>
  );
}
