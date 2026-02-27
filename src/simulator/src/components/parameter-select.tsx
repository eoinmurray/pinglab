import { useState } from 'react'
import { Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

type ParameterSelectOption<TValue extends string> = {
  value: TValue
  label: string
}

type ParameterSelectProps<TValue extends string> = {
  label: string
  info: string
  options: ParameterSelectOption<TValue>[]
  defaultValue?: TValue
  value?: TValue
  onValueChange?: (value: TValue) => void
  placeholder?: string
}

export default function ParameterSelect<TValue extends string>({
  label,
  info,
  options,
  defaultValue,
  value,
  onValueChange,
  placeholder = 'Select an option',
}: ParameterSelectProps<TValue>) {
  const initialValue =
    defaultValue ?? (options.length > 0 ? options[0].value : undefined)
  const [internalValue, setInternalValue] = useState<TValue | undefined>(
    initialValue
  )
  const isControlled = value !== undefined && onValueChange !== undefined
  const selectedValue = isControlled ? value : internalValue

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
      </div>
      <div>
        <Select
          value={selectedValue}
          onValueChange={(next) => {
            if (isControlled) {
              onValueChange(next as TValue)
              return
            }
            setInternalValue(next as TValue)
          }}
        >
          <SelectTrigger className="!h-6 w-full rounded-sm border-border/70 px-2 py-0 text-[11px] leading-none">
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent className="rounded-sm border-border/70 text-[11px]">
            {options.map((option) => (
              <SelectItem key={option.value} value={option.value} className="py-1 text-[11px]">
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
