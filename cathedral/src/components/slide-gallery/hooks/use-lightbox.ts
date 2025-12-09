import { useKeyBindings } from "@/hooks/useKeyBindings"; 
import { useCallback, useState } from "react";

export function useLightbox(totalImages: number) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const open = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const close = useCallback(() => {
    setSelectedIndex(null);
  }, []);

  const goToPrevious = useCallback(() => {
    setSelectedIndex(prev => prev !== null && prev > 0 ? prev - 1 : prev);
  }, []);

  const goToNext = useCallback(() => {
    setSelectedIndex(prev => prev !== null && prev < totalImages - 1 ? prev + 1 : prev);
  }, [totalImages]);

  useKeyBindings(
    [
      { key: "Escape", action: close },
      { key: "ArrowLeft", action: goToPrevious },
      { key: "ArrowRight", action: goToNext },
    ],
    { enabled: () => selectedIndex !== null }
  );

  return {
    selectedIndex,
    open,
    close,
    goToPrevious,
    goToNext,
    isOpen: selectedIndex !== null,
  };
}
