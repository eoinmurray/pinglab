import {
  useCallback,
  createContext,
  type PropsWithChildren,
  useEffect,
  useContext,
  useMemo,
  useRef,
  useState,
} from 'react'
import { toast } from 'sonner'
import { useParameters } from '@/hooks/use-parameters'
import { useSimulation } from '@/hooks/use-simulation'
import { useWeightsPreview } from '@/hooks/use-weights-preview'
import { getRunValidationError } from '@/lib/api/validation'
import type {
  ConfigPreset,
  ConfigNumericKey,
  GraphEdgeState,
  GraphNodeKind,
  GraphNodeState,
  InputNumericKey,
  InputTargetPopulation,
  InputTargetStrategy,
  InputType,
  MqifNumericKey,
  NeuronModel,
  WeightBlockKey,
} from '@/hooks/use-parameters'

type TopTab = 'topology' | 'plots' | 'weights'
type PlotsSubTab =
  | 'inputs'
  | 'input-spikes'
  | 'decode-path'
  | 'voltage'
  | 'population-rate'
  | 'psd'
  | 'correlations'

const SIMULATION_RUNNING_WARNING_TOAST_ID = 'simulation-running-warning'

type SimulationContextValue = {
  parameters: ReturnType<typeof useParameters>['parameters']
  setPerformanceMode: ReturnType<typeof useParameters>['setPerformanceMode']
  setDownsampleEnabled: ReturnType<typeof useParameters>['setDownsampleEnabled']
  setUseArrowTransport: ReturnType<typeof useParameters>['setUseArrowTransport']
  setBurnInMs: ReturnType<typeof useParameters>['setBurnInMs']
  setConfigValue: (key: ConfigNumericKey, value: number) => void
  setNeuronModel: (value: NeuronModel) => void
  setMqifValue: (key: MqifNumericKey, value: number) => void
  setInputType: (value: InputType) => void
  setInputPopulation: (value: InputTargetPopulation) => void
  setInputValue: (key: InputNumericKey, value: number) => void
  setInputTargetedSubsetEnabled: (value: boolean) => void
  setInputTargetPopulation: (value: InputTargetPopulation) => void
  setInputTargetStrategy: (value: InputTargetStrategy) => void
  setInputTargetFraction: (value: number) => void
  setInputTargetSeed: (value: number) => void
  setInputProgramType: (inputId: string, value: InputType) => void
  setInputProgramPopulation: (inputId: string, value: InputTargetPopulation) => void
  setInputProgramValue: (inputId: string, key: InputNumericKey, value: number) => void
  setInputProgramTargetedSubsetEnabled: (inputId: string, value: boolean) => void
  setInputProgramTargetPopulation: (inputId: string, value: InputTargetPopulation) => void
  setInputProgramTargetStrategy: (inputId: string, value: InputTargetStrategy) => void
  setInputProgramTargetFraction: (inputId: string, value: number) => void
  setInputProgramTargetSeed: (inputId: string, value: number) => void
  setWeightValue: (
    block: WeightBlockKey,
    key: 'mean' | 'std',
    value: number
  ) => void
  setWeightsSeed: (value: number) => void
  addGraphNode: (kind: GraphNodeKind) => void
  updateGraphNode: (
    nodeId: string,
    patch: Partial<Pick<GraphNodeState, 'id' | 'type' | 'size'>>
  ) => void
  removeGraphNode: (nodeId: string) => void
  addGraphEdge: () => void
  updateGraphEdge: (
    edgeId: string,
    patch: Partial<Pick<GraphEdgeState, 'id' | 'from' | 'to' | 'kind' | 'delay_ms'>> & {
      w?: Partial<GraphEdgeState['w']>
    }
  ) => void
  removeGraphEdge: (edgeId: string) => void
  applyConfigPreset: (preset: ConfigPreset) => void
  collapsibleOpen: Record<string, boolean>
  setCollapsibleOpen: (section: string, open: boolean) => void
  runData: ReturnType<typeof useSimulation>['data']
  runLoading: ReturnType<typeof useSimulation>['loading']
  runError: ReturnType<typeof useSimulation>['error']
  runTimings: ReturnType<typeof useSimulation>['timings']
  runValidationError: string | null
  runSimulationNow: () => void
  weightsPreview: ReturnType<typeof useWeightsPreview>['data']
  weightsLoading: ReturnType<typeof useWeightsPreview>['loading']
  weightsError: ReturnType<typeof useWeightsPreview>['error']
  activeTopTab: TopTab
  setActiveTopTab: (tab: TopTab) => void
  plotsSubTab: PlotsSubTab
  setPlotsSubTab: (tab: PlotsSubTab) => void
  plotsLayer: string
  setPlotsLayer: (layer: string) => void
}

const SimulationContext = createContext<SimulationContextValue | null>(null)

export function SimulationProvider({ children }: PropsWithChildren) {
  const {
    parameters,
    setPerformanceMode,
    setDownsampleEnabled,
    setUseArrowTransport,
    setBurnInMs,
    setConfigValue,
    setNeuronModel,
    setMqifValue,
    setInputType,
    setInputPopulation,
    setInputValue,
    setInputTargetedSubsetEnabled,
    setInputTargetPopulation,
    setInputTargetStrategy,
    setInputTargetFraction,
    setInputTargetSeed,
    setInputProgramType,
    setInputProgramPopulation,
    setInputProgramValue,
    setInputProgramTargetedSubsetEnabled,
    setInputProgramTargetPopulation,
    setInputProgramTargetStrategy,
    setInputProgramTargetFraction,
    setInputProgramTargetSeed,
    setWeightValue,
    setWeightsSeed,
    addGraphNode,
    updateGraphNode,
    removeGraphNode,
    addGraphEdge,
    updateGraphEdge,
    removeGraphEdge,
    applyConfigPreset,
  } = useParameters()
  const [activeTopTab, setActiveTopTab] = useState<TopTab>('plots')
  const [plotsSubTab, setPlotsSubTab] = useState<PlotsSubTab>('inputs')
  const [plotsLayer, setPlotsLayer] = useState('L1')
  const [collapsibleOpen, setCollapsibleOpenState] = useState<Record<string, boolean>>(
    {}
  )
  const [runTrigger, setRunTrigger] = useState(0)
  const runInFlightRef = useRef(false)
  const previousRunLoadingRef = useRef(false)
  const runValidationError = useMemo(() => getRunValidationError(parameters), [parameters])

  const { data: runData, loading: runLoading, error: runError, timings: runTimings } =
    useSimulation(parameters, runTrigger, parameters.useArrowTransport.value)
  const {
    data: weightsPreview,
    loading: weightsLoading,
    error: weightsError,
  } = useWeightsPreview(
    parameters,
    activeTopTab === 'weights',
    parameters.useArrowTransport.value
  )

  const runSimulationNow = useCallback(() => {
    if (runValidationError) {
      toast.error(runValidationError)
      return
    }
    if (runLoading || runInFlightRef.current) {
      toast.warning('Simulation already running', {
        id: SIMULATION_RUNNING_WARNING_TOAST_ID,
      })
      return
    }
    runInFlightRef.current = true
    setRunTrigger((current) => current + 1)
  }, [runLoading, runValidationError])

  useEffect(() => {
    if (previousRunLoadingRef.current && !runLoading) {
      runInFlightRef.current = false
    }
    previousRunLoadingRef.current = runLoading
  }, [runLoading])
  const setCollapsibleOpen = useCallback((section: string, open: boolean) => {
    setCollapsibleOpenState((prev) => {
      if ((prev[section] ?? false) === open) {
        return prev
      }
      return { ...prev, [section]: open }
    })
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.isComposing || event.repeat) {
        return
      }
      if (event.key !== ' ') {
        return
      }

      const target = event.target as HTMLElement | null
      if (target) {
        const tagName = target.tagName
        const isEditable =
          target.isContentEditable ||
          tagName === 'INPUT' ||
          tagName === 'TEXTAREA' ||
          tagName === 'SELECT'
        if (isEditable) {
          return
        }
      }

      if (event.metaKey || event.ctrlKey || event.altKey) {
        return
      }
      event.preventDefault()
      runSimulationNow()
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [runSimulationNow])

  const value = useMemo(
    () => ({
      parameters,
      setPerformanceMode,
      setDownsampleEnabled,
      setUseArrowTransport,
      setBurnInMs,
      setConfigValue,
      setNeuronModel,
      setMqifValue,
      setInputType,
      setInputPopulation,
      setInputValue,
      setInputTargetedSubsetEnabled,
      setInputTargetPopulation,
      setInputTargetStrategy,
      setInputTargetFraction,
      setInputTargetSeed,
      setInputProgramType,
      setInputProgramPopulation,
      setInputProgramValue,
      setInputProgramTargetedSubsetEnabled,
      setInputProgramTargetPopulation,
      setInputProgramTargetStrategy,
      setInputProgramTargetFraction,
      setInputProgramTargetSeed,
      setWeightValue,
      setWeightsSeed,
      addGraphNode,
      updateGraphNode,
      removeGraphNode,
      addGraphEdge,
      updateGraphEdge,
      removeGraphEdge,
      applyConfigPreset,
      collapsibleOpen,
      setCollapsibleOpen,
      runData,
      runLoading,
      runError,
      runTimings,
      runValidationError,
      runSimulationNow,
      weightsPreview,
      weightsLoading,
      weightsError,
      activeTopTab,
      setActiveTopTab,
      plotsSubTab,
      setPlotsSubTab,
      plotsLayer,
      setPlotsLayer,
    }),
    [
      parameters,
      setPerformanceMode,
      setDownsampleEnabled,
      setUseArrowTransport,
      setBurnInMs,
      setConfigValue,
      setNeuronModel,
      setMqifValue,
      setInputType,
      setInputPopulation,
      setInputValue,
      setInputTargetedSubsetEnabled,
      setInputTargetPopulation,
      setInputTargetStrategy,
      setInputTargetFraction,
      setInputTargetSeed,
      setInputProgramType,
      setInputProgramPopulation,
      setInputProgramValue,
      setInputProgramTargetedSubsetEnabled,
      setInputProgramTargetPopulation,
      setInputProgramTargetStrategy,
      setInputProgramTargetFraction,
      setInputProgramTargetSeed,
      setWeightValue,
      setWeightsSeed,
      addGraphNode,
      updateGraphNode,
      removeGraphNode,
      addGraphEdge,
      updateGraphEdge,
      removeGraphEdge,
      applyConfigPreset,
      collapsibleOpen,
      setCollapsibleOpen,
      runData,
      runLoading,
      runError,
      runTimings,
      runValidationError,
      runSimulationNow,
      weightsPreview,
      weightsLoading,
      weightsError,
      activeTopTab,
      setActiveTopTab,
      plotsSubTab,
      setPlotsSubTab,
      plotsLayer,
      setPlotsLayer,
    ]
  )

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  )
}

export function useSimulationContext() {
  const context = useContext(SimulationContext)
  if (!context) {
    throw new Error('useSimulationContext must be used within SimulationProvider')
  }
  return context
}
