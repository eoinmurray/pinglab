import * as React from 'react'

import { cn } from '@/lib/utils'

function ButtonGroup({
  className,
  orientation = 'horizontal',
  ...props
}: React.ComponentProps<'div'> & {
  orientation?: 'horizontal' | 'vertical'
}) {
  return (
    <div
      role="group"
      data-slot="button-group"
      data-orientation={orientation}
      className={cn(
        'inline-flex rounded-md',
        orientation === 'vertical' ? 'flex-col' : 'items-center',
        className
      )}
      {...props}
    />
  )
}

export { ButtonGroup }
