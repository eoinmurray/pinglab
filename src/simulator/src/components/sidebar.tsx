import ParameterPanel from '@/components/parameter-panel'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
} from '@/components/ui/sidebar'

export default function PinglabSidebar() {
  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader>
        <div className="px-2 py-1 group-data-[collapsible=icon]:hidden">
          <p className="font-mono text-[13px] font-semibold uppercase tracking-[0.14em] text-primary">
            pinglab
          </p>
          <p className="text-[11px] leading-tight text-muted-foreground">
            Explore layered PING dynamics and signal propagation.
          </p>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Parameters</SidebarGroupLabel>
          <SidebarGroupContent>
            <ParameterPanel />
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}
