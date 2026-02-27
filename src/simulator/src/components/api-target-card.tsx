import { Server } from 'lucide-react'
import { getApiTargetInfo } from '@/lib/api/target'

export default function ApiTargetCard() {
  const { target, hostname } = getApiTargetInfo()
  const label = target === 'local' ? 'Localhost API' : 'Modal API'

  return (
    <div className="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] text-muted-foreground">
      <Server className="size-3.5" />
      <span className="font-medium text-foreground">{label}</span>
      <span className="text-muted-foreground/80">({hostname})</span>
    </div>
  )
}
