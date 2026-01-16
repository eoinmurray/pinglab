import {
  Tooltip,
  TooltipArrow,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

type InfoPopoverProps = {
  text: string;
};

export default function InfoPopover({ text }: InfoPopoverProps) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
        <button
          type="button"
          className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-black text-[10px] leading-none text-black dark:border-zinc-100 dark:text-zinc-100"
          aria-label="Parameter info"
        >
          i
        </button>
        </TooltipTrigger>
        <TooltipContent side="right" align="start">
          {text}
          <TooltipArrow className="fill-black" />
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
