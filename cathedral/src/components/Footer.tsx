import { Kbd } from "./ui/kbd";


export default function Footer() {
  return (
    <footer className="w-full py-3 text-center text-sm text-gray-500 print:hidden">
      <div className="flex items-center justify-end gap-2 mt-1">
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground mr-2">
          <div className="flex items-center gap-1">
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd>
            <span>Navigate</span>
          </div>
          <div className="flex items-center gap-1">
            <Kbd>→</Kbd>
            <Kbd>↵</Kbd>
            <span>Open</span>
          </div>
          <div className="flex items-center gap-1">
            <Kbd>←</Kbd>
            <span>Back</span>
          </div>
          <div className="flex items-center gap-1">
            <Kbd>,</Kbd>
            <span>Home</span>
          </div>
          <div
            className="flex items-center gap-1"
          >
            <Kbd>.</Kbd>
            <span>Show/hide</span>
          </div>
        </div>
      </div>
    </footer>
  )
}