import Slider from "./Slider";
import Select from "./Select";

export type NeuronModel =
  | "lif"
  | "hh"
  | "adex"
  | "connor_stevens"
  | "fitzhugh"
  | "mqif"
  | "qif"
  | "izhikevich";

const NEURON_MODELS: NeuronModel[] = [
  "lif",
  "hh",
  "adex",
  "connor_stevens",
  "fitzhugh",
  "mqif",
  "qif",
  "izhikevich",
];


type ParameterPanelProps = {
  neuronModel: NeuronModel;
  setNeuronModel: (value: NeuronModel) => void;
  dt: number;
  setDt: (value: number) => void;
  T: number;
  setT: (value: number) => void;
  nE: number;
  setNE: (value: number) => void;
  nI: number;
  setNI: (value: number) => void;
  seed: number;
  setSeed: (value: number) => void;
  iE: number;
  setIE: (value: number) => void;
  iI: number;
  setII: (value: number) => void;
  noiseStd: number;
  setNoiseStd: (value: number) => void;
  inputSeed: number;
  setInputSeed: (value: number) => void;
  gEeMean: number;
  setGEeMean: (value: number) => void;
  gEiMean: number;
  setGEiMean: (value: number) => void;
  gIeMean: number;
  setGIeMean: (value: number) => void;
  gIiMean: number;
  setGIiMean: (value: number) => void;
  gEeStd: number;
  setGEeStd: (value: number) => void;
  gEiStd: number;
  setGEiStd: (value: number) => void;
  gIeStd: number;
  setGIeStd: (value: number) => void;
  gIiStd: number;
  setGIiStd: (value: number) => void;
  pEe: number;
  setPEe: (value: number) => void;
  pEi: number;
  setPEi: (value: number) => void;
  pIe: number;
  setPIe: (value: number) => void;
  pIi: number;
  setPIi: (value: number) => void;
  clampMin: number;
  setClampMin: (value: number) => void;
  weightsSeed: number;
  setWeightsSeed: (value: number) => void;
  delayEe: number;
  setDelayEe: (value: number) => void;
  delayEi: number;
  setDelayEi: (value: number) => void;
  delayIe: number;
  setDelayIe: (value: number) => void;
  delayIi: number;
  setDelayIi: (value: number) => void;
  VInit: number;
  setVInit: (value: number) => void;
  VTh: number;
  setVTh: (value: number) => void;
  VReset: number;
  setVReset: (value: number) => void;
  EL: number;
  setEL: (value: number) => void;
  Ee: number;
  setEe: (value: number) => void;
  Ei: number;
  setEi: (value: number) => void;
  CmE: number;
  setCmE: (value: number) => void;
  gLE: number;
  setGLE: (value: number) => void;
  CmI: number;
  setCmI: (value: number) => void;
  gLI: number;
  setGLI: (value: number) => void;
  tRefE: number;
  setTRefE: (value: number) => void;
  tRefI: number;
  setTRefI: (value: number) => void;
  tauAmpa: number;
  setTauAmpa: (value: number) => void;
  tauGaba: number;
  setTauGaba: (value: number) => void;
  gNa: number;
  setGNa: (value: number) => void;
  gK: number;
  setGK: (value: number) => void;
  ENa: number;
  setENa: (value: number) => void;
  EK: number;
  setEK: (value: number) => void;
  adexVT: number;
  setAdexVT: (value: number) => void;
  adexDeltaT: number;
  setAdexDeltaT: (value: number) => void;
  adexTauW: number;
  setAdexTauW: (value: number) => void;
  adexA: number;
  setAdexA: (value: number) => void;
  adexB: number;
  setAdexB: (value: number) => void;
  adexVPeak: number;
  setAdexVPeak: (value: number) => void;
  gA: number;
  setGA: (value: number) => void;
  fhnA: number;
  setFhnA: (value: number) => void;
  fhnB: number;
  setFhnB: (value: number) => void;
  fhnTauW: number;
  setFhnTauW: (value: number) => void;
  qifA: number;
  setQifA: (value: number) => void;
  qifVr: number;
  setQifVr: (value: number) => void;
  qifVt: number;
  setQifVt: (value: number) => void;
  izhA: number;
  setIzhA: (value: number) => void;
  izhB: number;
  setIzhB: (value: number) => void;
  izhC: number;
  setIzhC: (value: number) => void;
  izhD: number;
  setIzhD: (value: number) => void;
  mqifA: number;
  setMqifA: (value: number) => void;
  mqifVr: number;
  setMqifVr: (value: number) => void;
  pulseOnset: number;
  setPulseOnset: (value: number) => void;
  pulseDuration: number;
  setPulseDuration: (value: number) => void;
  pulseInterval: number;
  setPulseInterval: (value: number) => void;
  pulseAmpE: number;
  setPulseAmpE: (value: number) => void;
  pulseAmpI: number;
  setPulseAmpI: (value: number) => void;
  gLHet: number;
  setGLHet: (value: number) => void;
  cMHet: number;
  setCMHet: (value: number) => void;
  vThHet: number;
  setVThHet: (value: number) => void;
  tRefHet: number;
  setTRefHet: (value: number) => void;
};

export default function ParameterPanel({
  neuronModel,
  setNeuronModel,
  dt,
  setDt,
  T,
  setT,
  nE,
  setNE,
  nI,
  setNI,
  seed,
  setSeed,
  iE,
  setIE,
  iI,
  setII,
  noiseStd,
  setNoiseStd,
  inputSeed,
  setInputSeed,
  gEeMean,
  setGEeMean,
  gEiMean,
  setGEiMean,
  gIeMean,
  setGIeMean,
  gIiMean,
  setGIiMean,
  gEeStd,
  setGEeStd,
  gEiStd,
  setGEiStd,
  gIeStd,
  setGIeStd,
  gIiStd,
  setGIiStd,
  pEe,
  setPEe,
  pEi,
  setPEi,
  pIe,
  setPIe,
  pIi,
  setPIi,
  clampMin,
  setClampMin,
  weightsSeed,
  setWeightsSeed,
  delayEe,
  setDelayEe,
  delayEi,
  setDelayEi,
  delayIe,
  setDelayIe,
  delayIi,
  setDelayIi,
  VInit,
  setVInit,
  VTh,
  setVTh,
  VReset,
  setVReset,
  EL,
  setEL,
  Ee,
  setEe,
  Ei,
  setEi,
  CmE,
  setCmE,
  gLE,
  setGLE,
  CmI,
  setCmI,
  gLI,
  setGLI,
  tRefE,
  setTRefE,
  tRefI,
  setTRefI,
  tauAmpa,
  setTauAmpa,
  tauGaba,
  setTauGaba,
  gNa,
  setGNa,
  gK,
  setGK,
  ENa,
  setENa,
  EK,
  setEK,
  adexVT,
  setAdexVT,
  adexDeltaT,
  setAdexDeltaT,
  adexTauW,
  setAdexTauW,
  adexA,
  setAdexA,
  adexB,
  setAdexB,
  adexVPeak,
  setAdexVPeak,
  gA,
  setGA,
  fhnA,
  setFhnA,
  fhnB,
  setFhnB,
  fhnTauW,
  setFhnTauW,
  qifA,
  setQifA,
  qifVr,
  setQifVr,
  qifVt,
  setQifVt,
  izhA,
  setIzhA,
  izhB,
  setIzhB,
  izhC,
  setIzhC,
  izhD,
  setIzhD,
  mqifA,
  setMqifA,
  mqifVr,
  setMqifVr,
  pulseOnset,
  setPulseOnset,
  pulseDuration,
  setPulseDuration,
  pulseInterval,
  setPulseInterval,
  pulseAmpE,
  setPulseAmpE,
  pulseAmpI,
  setPulseAmpI,
  gLHet,
  setGLHet,
  cMHet,
  setCMHet,
  vThHet,
  setVThHet,
  tRefHet,
  setTRefHet,
}: ParameterPanelProps) {
  return (
    <div className="h-full w-72 rounded-lg p-3 text-xs">
      <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
        Parameters
      </div>
      <div className="mt-4 h-[calc(100%-24px)] space-y-4 overflow-y-auto pr-1">
        <div className="space-y-3">
          <Select
            label="neuron model"
            value={neuronModel}
            options={NEURON_MODELS}
            onChange={(value) => setNeuronModel(value as NeuronModel)}
          />
          <Slider label="dt" value={dt} min={0.05} max={1.0} step={0.05} precision={2} onChange={setDt} />
          <Slider label="T" value={T} min={200} max={5000} step={50} precision={0} onChange={setT} />
          <Slider label="N_E" value={nE} min={100} max={2000} step={50} precision={0} onChange={setNE} />
          <Slider label="N_I" value={nI} min={50} max={1000} step={50} precision={0} onChange={setNI} />
          <Slider label="seed" value={seed} min={0} max={20} step={1} precision={0} onChange={setSeed} />
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Inputs
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="I_E" value={iE} min={0.0} max={2.0} step={0.01} precision={2} onChange={setIE} />
            <Slider label="I_I" value={iI} min={0.0} max={2.0} step={0.01} precision={2} onChange={setII} />
            <Slider
              label="noise std"
              value={noiseStd}
              min={0.0}
              max={5.0}
              step={0.05}
              precision={2}
              onChange={setNoiseStd}
            />
            <Slider label="input seed" value={inputSeed} min={0} max={20} step={1} precision={0} onChange={setInputSeed} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Weights
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_ee mean" value={gEeMean} min={0.0} max={0.2} step={0.001} onChange={setGEeMean} />
            <Slider label="g_ei mean" value={gEiMean} min={0.005} max={0.2} step={0.001} onChange={setGEiMean} />
            <Slider label="g_ie mean" value={gIeMean} min={0.005} max={0.2} step={0.001} onChange={setGIeMean} />
            <Slider label="g_ii mean" value={gIiMean} min={0.0} max={0.2} step={0.001} onChange={setGIiMean} />
            <Slider label="g_ee stddev" value={gEeStd} min={0.0} max={0.2} step={0.001} onChange={setGEeStd} />
            <Slider label="g_ei stddev" value={gEiStd} min={0.0} max={0.2} step={0.001} onChange={setGEiStd} />
            <Slider label="g_ie stddev" value={gIeStd} min={0.0} max={0.2} step={0.001} onChange={setGIeStd} />
            <Slider label="g_ii stddev" value={gIiStd} min={0.0} max={0.2} step={0.001} onChange={setGIiStd} />
            <Slider label="p_ee" value={pEe} min={0.0} max={1.0} step={0.01} precision={2} onChange={setPEe} />
            <Slider label="p_ei" value={pEi} min={0.0} max={1.0} step={0.01} precision={2} onChange={setPEi} />
            <Slider label="p_ie" value={pIe} min={0.0} max={1.0} step={0.01} precision={2} onChange={setPIe} />
            <Slider label="p_ii" value={pIi} min={0.0} max={1.0} step={0.01} precision={2} onChange={setPIi} />
            <Slider label="clamp min" value={clampMin} min={0.0} max={0.1} step={0.005} precision={3} onChange={setClampMin} />
            <Slider label="weights seed" value={weightsSeed} min={0} max={20} step={1} precision={0} onChange={setWeightsSeed} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Synapse and delays
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="tau AMPA" value={tauAmpa} min={1} max={10} step={0.5} precision={1} onChange={setTauAmpa} />
            <Slider label="tau GABA" value={tauGaba} min={1} max={15} step={0.5} precision={1} onChange={setTauGaba} />
            <Slider label="delay EE" value={delayEe} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayEe} />
            <Slider label="delay EI" value={delayEi} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayEi} />
            <Slider label="delay IE" value={delayIe} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayIe} />
            <Slider label="delay II" value={delayIi} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayIi} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Membrane
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="V_init" value={VInit} min={-80} max={-50} step={1} precision={0} onChange={setVInit} />
            <Slider label="V_th" value={VTh} min={-70} max={-40} step={1} precision={0} onChange={setVTh} />
            <Slider label="V_reset" value={VReset} min={-80} max={-50} step={1} precision={0} onChange={setVReset} />
            <Slider label="E_L" value={EL} min={-80} max={-50} step={1} precision={0} onChange={setEL} />
            <Slider label="E_e" value={Ee} min={-10} max={10} step={1} precision={0} onChange={setEe} />
            <Slider label="E_i" value={Ei} min={-100} max={-60} step={1} precision={0} onChange={setEi} />
            <Slider label="C_m E" value={CmE} min={0.2} max={2.0} step={0.1} precision={2} onChange={setCmE} />
            <Slider label="g_L E" value={gLE} min={0.01} max={0.2} step={0.005} precision={3} onChange={setGLE} />
            <Slider label="C_m I" value={CmI} min={0.2} max={2.0} step={0.1} precision={2} onChange={setCmI} />
            <Slider label="g_L I" value={gLI} min={0.01} max={0.2} step={0.005} precision={3} onChange={setGLI} />
            <Slider label="t_ref E" value={tRefE} min={0.5} max={5.0} step={0.1} precision={1} onChange={setTRefE} />
            <Slider label="t_ref I" value={tRefI} min={0.5} max={5.0} step={0.1} precision={1} onChange={setTRefI} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            HH and AdEx
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_Na" value={gNa} min={50} max={200} step={5} precision={0} onChange={setGNa} />
            <Slider label="g_K" value={gK} min={10} max={80} step={2} precision={0} onChange={setGK} />
            <Slider label="E_Na" value={ENa} min={30} max={70} step={1} precision={0} onChange={setENa} />
            <Slider label="E_K" value={EK} min={-100} max={-60} step={1} precision={0} onChange={setEK} />
            <Slider label="AdEx V_T" value={adexVT} min={-70} max={-30} step={1} precision={0} onChange={setAdexVT} />
            <Slider
              label="AdEx Delta_T"
              value={adexDeltaT}
              min={0.5}
              max={5.0}
              step={0.1}
              precision={1}
              onChange={setAdexDeltaT}
            />
            <Slider label="AdEx tau_w" value={adexTauW} min={10} max={300} step={5} precision={0} onChange={setAdexTauW} />
            <Slider label="AdEx a" value={adexA} min={0} max={10} step={0.5} precision={1} onChange={setAdexA} />
            <Slider label="AdEx b" value={adexB} min={0} max={100} step={2} precision={0} onChange={setAdexB} />
            <Slider label="AdEx V_peak" value={adexVPeak} min={0} max={40} step={1} precision={0} onChange={setAdexVPeak} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Connor-Stevens and FitzHugh
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_A" value={gA} min={0} max={80} step={2} precision={0} onChange={setGA} />
            <Slider label="FHN a" value={fhnA} min={0.0} max={1.5} step={0.05} precision={2} onChange={setFhnA} />
            <Slider label="FHN b" value={fhnB} min={0.0} max={2.0} step={0.05} precision={2} onChange={setFhnB} />
            <Slider label="FHN tau_w" value={fhnTauW} min={1} max={30} step={1} precision={0} onChange={setFhnTauW} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            QIF and Izhikevich
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="QIF a" value={qifA} min={0.2} max={3.0} step={0.1} precision={1} onChange={setQifA} />
            <Slider label="QIF V_r" value={qifVr} min={-80} max={-40} step={1} precision={0} onChange={setQifVr} />
            <Slider label="QIF V_t" value={qifVt} min={-60} max={-30} step={1} precision={0} onChange={setQifVt} />
            <Slider label="Izh a" value={izhA} min={0.0} max={0.2} step={0.01} precision={2} onChange={setIzhA} />
            <Slider label="Izh b" value={izhB} min={0.0} max={0.4} step={0.01} precision={2} onChange={setIzhB} />
            <Slider label="Izh c" value={izhC} min={-80} max={-40} step={1} precision={0} onChange={setIzhC} />
            <Slider label="Izh d" value={izhD} min={0} max={20} step={1} precision={0} onChange={setIzhD} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            MQIF and pulse
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="MQIF a" value={mqifA} min={0.0} max={0.2} step={0.005} precision={3} onChange={setMqifA} />
            <Slider label="MQIF V_r" value={mqifVr} min={-70} max={-40} step={1} precision={0} onChange={setMqifVr} />
            <Slider label="pulse onset" value={pulseOnset} min={0} max={1000} step={10} precision={0} onChange={setPulseOnset} />
            <Slider label="pulse duration" value={pulseDuration} min={0} max={1000} step={10} precision={0} onChange={setPulseDuration} />
            <Slider label="pulse interval" value={pulseInterval} min={0} max={1000} step={10} precision={0} onChange={setPulseInterval} />
            <Slider label="pulse amp E" value={pulseAmpE} min={0} max={5} step={0.1} precision={1} onChange={setPulseAmpE} />
            <Slider label="pulse amp I" value={pulseAmpI} min={0} max={5} step={0.1} precision={1} onChange={setPulseAmpI} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Heterogeneity
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_L sd" value={gLHet} min={0.0} max={1.0} step={0.05} precision={2} onChange={setGLHet} />
            <Slider label="C_m sd" value={cMHet} min={0.0} max={1.0} step={0.05} precision={2} onChange={setCMHet} />
            <Slider label="V_th sd" value={vThHet} min={0.0} max={3.0} step={0.1} precision={1} onChange={setVThHet} />
            <Slider label="t_ref sd" value={tRefHet} min={0.0} max={2.0} step={0.1} precision={1} onChange={setTRefHet} />
          </div>
        </div>
      </div>
    </div>
  );
}
