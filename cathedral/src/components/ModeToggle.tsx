import { Moon, Sun, Monitor } from "lucide-react"
import { useTheme } from "next-themes"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function ModeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className="relative p-2 text-muted-foreground/70 hover:text-foreground transition-colors duration-300"
          aria-label="Toggle theme"
        >
          <Sun className={`h-4 w-4 transition-all duration-300 ${theme === "light" ? "scale-100 rotate-0" : "scale-0 -rotate-90 absolute top-2 left-2"}`} />
          <Moon className={`h-4 w-4 transition-all duration-300 ${theme === "dark" ? "scale-100 rotate-0" : "scale-0 rotate-90 absolute top-2 left-2"}`} />
          <Monitor className={`h-4 w-4 transition-all duration-300 ${theme === "system" || !theme ? "scale-100" : "scale-0 absolute top-2 left-2"}`} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuRadioGroup value={theme} onValueChange={setTheme}>
          <DropdownMenuRadioItem value="light">
            <Sun className="mr-2 h-4 w-4" />
            Light
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark">
            <Moon className="mr-2 h-4 w-4" />
            Dark
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system">
            <Monitor className="mr-2 h-4 w-4" />
            System
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
