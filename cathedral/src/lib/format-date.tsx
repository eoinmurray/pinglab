
import { format } from 'date-fns'

export function formatDate(date: string | number | Date): string {
  return format(new Date(date), "dd MMM, yy")
}