import { useEffect } from "react";

type KeyModifiers = {
  meta?: boolean;
  ctrl?: boolean;
  alt?: boolean;
  shift?: boolean;
};

type KeyBinding = {
  key: string | string[];
  action: () => void;
  modifiers?: KeyModifiers;
  preventDefault?: boolean;
};

type UseKeyBindingsOptions = {
  enabled?: boolean | (() => boolean);
  ignoreModifiers?: boolean;
};

export function useKeyBindings(
  bindings: KeyBinding[],
  options: UseKeyBindingsOptions = {}
) {
  const { enabled = true, ignoreModifiers = true } = options;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isEnabled = typeof enabled === "function" ? enabled() : enabled;
      if (!isEnabled) return;

      // By default, ignore events with modifier keys unless explicitly specified
      if (ignoreModifiers && (e.metaKey || e.ctrlKey || e.altKey)) {
        // Check if any binding explicitly wants this modifier combination
        const hasModifierBinding = bindings.some((b) => {
          const mods = b.modifiers || {};
          return mods.meta || mods.ctrl || mods.alt;
        });
        if (!hasModifierBinding) return;
      }

      for (const binding of bindings) {
        const keys = Array.isArray(binding.key) ? binding.key : [binding.key];
        const mods = binding.modifiers || {};

        // Check if key matches
        if (!keys.includes(e.key)) continue;

        // Check modifiers if specified
        if (mods.meta !== undefined && mods.meta !== e.metaKey) continue;
        if (mods.ctrl !== undefined && mods.ctrl !== e.ctrlKey) continue;
        if (mods.alt !== undefined && mods.alt !== e.altKey) continue;
        if (mods.shift !== undefined && mods.shift !== e.shiftKey) continue;

        // If ignoreModifiers is true and no modifiers specified, skip if any modifier is pressed
        if (ignoreModifiers && !binding.modifiers && (e.metaKey || e.ctrlKey || e.altKey)) {
          continue;
        }

        if (binding.preventDefault !== false) {
          e.preventDefault();
        }
        binding.action();
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [bindings, enabled, ignoreModifiers]);
}
