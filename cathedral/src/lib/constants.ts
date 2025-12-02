// Data attributes for fullscreen components
export const FULLSCREEN_DATA_ATTR = "data-fullscreen-active";

/**
 * Check if any fullscreen component (Gallery or Slides) is currently active.
 * Components using fullscreen should set the data-fullscreen-active attribute.
 */
export function isFullscreenActive(): boolean {
  return document.querySelector(`[${FULLSCREEN_DATA_ATTR}="true"]`) !== null;
}
