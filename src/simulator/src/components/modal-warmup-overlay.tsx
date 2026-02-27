export default function ModalWarmupOverlay() {
  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center bg-background/92 backdrop-blur-[1px]">
      <div className="w-[420px] space-y-4 rounded-xl border bg-card p-6 shadow-sm">
        <div className="text-base font-semibold">Modal warming up</div>
        <div className="text-sm text-muted-foreground">
          First request can take a few seconds while the container starts.
        </div>
      </div>
    </div>
  )
}
