import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import PinglabSidebar from '@/components/sidebar'
import { SimulationProvider, useSimulationContext } from '@/context/simulation-context'
import { ModeToggle } from '@/components/mode-toggle'
import ApiTargetCard from '@/components/api-target-card'
import ModalWarmupOverlay from '@/components/modal-warmup-overlay'
import TabPlots from '@/components/tab-plots'
import TabGlobalWeights from '@/components/tab-global-weights'
import TabTopology from '@/components/tab-topology'
import { useModalWarmup } from '@/hooks/use-modal-warmup'
import {
  SidebarInset,
  SidebarProvider,
  // SidebarTrigger leaving in so its optional
} from '@/components/ui/sidebar'

function HomePageLayout() {
  const { activeTopTab, setActiveTopTab } = useSimulationContext()
  const { isModal, isWarming } = useModalWarmup()

  return (
    <SidebarProvider className="h-screen overflow-hidden">
      <PinglabSidebar />
      <SidebarInset className="h-screen overflow-hidden">
        <main className="relative flex h-full min-h-0 w-full flex-1 flex-col overflow-hidden p-3">
          {isModal && isWarming ? <ModalWarmupOverlay /> : null}
          <Tabs
            value={activeTopTab}
            onValueChange={(value) =>
              setActiveTopTab(value as 'topology' | 'plots' | 'weights')
            }
            className="flex min-h-0 w-full flex-1 flex-col"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                {/* <SidebarTrigger /> */}
                <TabsList>
                  <TabsTrigger value="plots">Plots</TabsTrigger>
                  <TabsTrigger value="weights">Weights</TabsTrigger>
                  <TabsTrigger value="topology">Topology</TabsTrigger>
                </TabsList>
              </div>
              <div className="flex items-center gap-2">
                <ApiTargetCard />
                <ModeToggle />
              </div>
            </div>
            <TabsContent value="plots" className="sim-main-panel min-h-0 flex-1 overflow-hidden">
              <TabPlots />
            </TabsContent>
            <TabsContent
              value="weights"
              className="sim-main-panel min-h-0 flex-1 overflow-hidden"
            >
              <TabGlobalWeights />
            </TabsContent>
            <TabsContent
              value="topology"
              className="sim-main-panel min-h-0 flex-1 overflow-hidden"
            >
              <TabTopology />
            </TabsContent>
          </Tabs>
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}

export default function HomePage() {
  return (
    <SimulationProvider>
      <HomePageLayout />
    </SimulationProvider>
  )
}
