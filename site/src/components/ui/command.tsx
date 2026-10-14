import type { ComponentProps, ReactNode } from 'react';
import { Command as CommandPrimitive } from 'cmdk';
import * as Dialog from '@radix-ui/react-dialog';

// shadcn/ui's Command composition, styled with Pinglab CSS instead of Tailwind.
export function CommandDialog({ children, trigger, filter, ...props }: ComponentProps<typeof Dialog.Root> & {
  trigger: ReactNode;
  filter?: ComponentProps<typeof CommandPrimitive>['filter'];
}) {
  return (
    <Dialog.Root {...props}>
      <Dialog.Trigger asChild>{trigger}</Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="command-overlay" />
        <Dialog.Content className="command-dialog">
          <Dialog.Title className="command-sr-only">Navigate Pinglab</Dialog.Title>
          <Dialog.Description className="command-sr-only">
            Search articles and collections. Use the arrow keys to select and Enter to open.
          </Dialog.Description>
          <CommandPrimitive label="Articles and collections" filter={filter} loop vimBindings={false}>
            {children}
          </CommandPrimitive>
          <Dialog.Close className="command-close" aria-label="Close search">Esc</Dialog.Close>
          <div className="command-help" aria-hidden="true">↑ ↓ navigate · Enter open · Esc close</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export const CommandInput = CommandPrimitive.Input;
export const CommandList = CommandPrimitive.List;
export const CommandEmpty = CommandPrimitive.Empty;
export const CommandGroup = CommandPrimitive.Group;
export const CommandItem = CommandPrimitive.Item;
