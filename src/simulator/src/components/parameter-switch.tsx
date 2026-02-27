import { useState } from 'react'
import { Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'

type ParameterSwitchProps = {
  label: string
  info: string
  defaultChecked?: boolean
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
  onLabel?: string
  offLabel?: string
}

export default function ParameterSwitch({
  label,
  info,
  defaultChecked = false,
  checked,
  onCheckedChange,
}: ParameterSwitchProps) {
  const [internalChecked, setInternalChecked] = useState(defaultChecked)
  const isControlled = checked !== undefined && onCheckedChange !== undefined
  const currentChecked = isControlled ? checked : internalChecked

  return (
    <div className="space-y-1 px-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <div className="flex items-center gap-1">
          <span className="font-medium leading-none text-foreground">{label}</span>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                type="button"
                size="icon-xs"
                variant="ghost"
                className="size-5 rounded-sm text-muted-foreground [&_svg]:size-3"
                aria-label={`${label} info`}
              >
                <Info />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-64 text-[11px]" side="right" align="start">
              {info}
            </PopoverContent>
          </Popover>
        </div>
        <Switch
          size="sm"
          checked={currentChecked}
          onCheckedChange={(next) => {
            if (isControlled) {
              onCheckedChange(next)
              return
            }
            setInternalChecked(next)
          }}
          aria-label={label}
        />
      </div>
    </div>
  )
}
