import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";

const Select = SelectPrimitive.Root;
const SelectGroup = SelectPrimitive.Group;
const SelectValue = SelectPrimitive.Value;
const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={`flex h-8 w-full items-center justify-between rounded-md border border-black bg-white px-2 text-[11px] uppercase tracking-[0.08em] text-black shadow-sm outline-none transition-colors hover:bg-black hover:text-white dark:border-zinc-100 dark:bg-black dark:text-zinc-100 dark:hover:bg-white dark:hover:text-black [&_[data-placeholder]]:text-neutral-400 ${
      className ?? ""
    }`}
    {...props}
  >
    <span className="truncate text-current">{children}</span>
    <SelectPrimitive.Icon className="ml-2 text-current">
      <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" aria-hidden="true">
        <path d="M5 7l5 6 5-6H5z" fill="currentColor" />
      </svg>
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName;

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
  <SelectPrimitive.Content
      ref={ref}
      position={position}
      className={`z-50 min-w-[8rem] overflow-hidden rounded-md border border-black !bg-black p-1 text-[11px] uppercase tracking-[0.08em] !text-white shadow-md dark:border-zinc-100 ${
        className ?? ""
      }`}
      {...props}
    >
    <SelectPrimitive.Viewport
      className="p-1"
      style={{ backgroundColor: "#0b0b0b", color: "#f5f5f5" }}
    >
        {props.children}
      </SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
SelectContent.displayName = SelectPrimitive.Content.displayName;

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={`relative flex cursor-default select-none items-center rounded-sm px-2 py-1 text-[11px] uppercase tracking-[0.08em] outline-none data-[highlighted]:bg-black data-[highlighted]:text-white dark:data-[highlighted]:bg-white dark:data-[highlighted]:text-black ${
      className ?? ""
    }`}
    {...props}
  >
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
SelectItem.displayName = SelectPrimitive.Item.displayName;

export {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
};
