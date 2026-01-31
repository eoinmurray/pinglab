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

type WeightDistName = "normal" | "lognormal" | "gamma" | "exponential";
type WeightTemplateName = "none" | "feedforward_blocks";

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

const WEIGHT_DISTS: WeightDistName[] = ["normal", "lognormal", "gamma", "exponential"];
const EE_TEMPLATES: WeightTemplateName[] = ["none", "feedforward_blocks"];

type ParameterPanelProps = {
  configList: string[];
  selectedConfig: string;
  onSelectConfig: (value: string) => void;
  saveName: string;
  setSaveName: (value: string) => void;
  onSaveConfig: () => void;
  configStatus: string | null;
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
  burnInMs: number;
  setBurnInMs: (value: number) => void;
  downsampleEnabled: boolean;
  setDownsampleEnabled: (value: boolean) => void;
  noiseStdE: number;
  setNoiseStdE: (value: number) => void;
  noiseStdI: number;
  setNoiseStdI: (value: number) => void;
  inputSeed: number;
  setInputSeed: (value: number) => void;
  inputType: "ramp" | "pulse" | "pulses";
  setInputType: (value: "ramp" | "pulse" | "pulses") => void;
  iEStart: number;
  setIEStart: (value: number) => void;
  iEEnd: number;
  setIEEnd: (value: number) => void;
  iIStart: number;
  setIIStart: (value: number) => void;
  iIEnd: number;
  setIIEnd: (value: number) => void;
  iEBase: number;
  setIEBase: (value: number) => void;
  iIBase: number;
  setIIBase: (value: number) => void;
  inputPulseT: number;
  setInputPulseT: (value: number) => void;
  inputPulseWidth: number;
  setInputPulseWidth: (value: number) => void;
  inputPulseInterval: number;
  setInputPulseInterval: (value: number) => void;
  inputPulseAmpE: number;
  setInputPulseAmpE: (value: number) => void;
  inputPulseAmpI: number;
  setInputPulseAmpI: (value: number) => void;
  eeDist: WeightDistName;
  setEeDist: (value: WeightDistName) => void;
  eeTemplate: WeightTemplateName;
  setEeTemplate: (value: WeightTemplateName) => void;
  eeTemplateBlocks: number;
  setEeTemplateBlocks: (value: number) => void;
  eiDist: WeightDistName;
  setEiDist: (value: WeightDistName) => void;
  ieDist: WeightDistName;
  setIeDist: (value: WeightDistName) => void;
  iiDist: WeightDistName;
  setIiDist: (value: WeightDistName) => void;
  eeMean: number;
  setEeMean: (value: number) => void;
  eiMean: number;
  setEiMean: (value: number) => void;
  ieMean: number;
  setIeMean: (value: number) => void;
  iiMean: number;
  setIiMean: (value: number) => void;
  eeStd: number;
  setEeStd: (value: number) => void;
  eiStd: number;
  setEiStd: (value: number) => void;
  ieStd: number;
  setIeStd: (value: number) => void;
  iiStd: number;
  setIiStd: (value: number) => void;
  eeSigma: number;
  setEeSigma: (value: number) => void;
  eiSigma: number;
  setEiSigma: (value: number) => void;
  ieSigma: number;
  setIeSigma: (value: number) => void;
  iiSigma: number;
  setIiSigma: (value: number) => void;
  eeShape: number;
  setEeShape: (value: number) => void;
  eiShape: number;
  setEiShape: (value: number) => void;
  ieShape: number;
  setIeShape: (value: number) => void;
  iiShape: number;
  setIiShape: (value: number) => void;
  eeScale: number;
  setEeScale: (value: number) => void;
  eiScale: number;
  setEiScale: (value: number) => void;
  ieScale: number;
  setIeScale: (value: number) => void;
  iiScale: number;
  setIiScale: (value: number) => void;
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
  mqifWA: number;
  setMqifWA: (value: number) => void;
  mqifWVr: number;
  setMqifWVr: (value: number) => void;
  mqifWTau: number;
  setMqifWTau: (value: number) => void;
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
  configList,
  selectedConfig,
  onSelectConfig,
  saveName,
  setSaveName,
  onSaveConfig,
  configStatus,
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
  burnInMs,
  setBurnInMs,
  downsampleEnabled,
  setDownsampleEnabled,
  noiseStdE,
  setNoiseStdE,
  noiseStdI,
  setNoiseStdI,
  inputSeed,
  setInputSeed,
  inputType,
  setInputType,
  iEStart,
  setIEStart,
  iEEnd,
  setIEEnd,
  iIStart,
  setIIStart,
  iIEnd,
  setIIEnd,
  iEBase,
  setIEBase,
  iIBase,
  setIIBase,
  inputPulseT,
  setInputPulseT,
  inputPulseWidth,
  setInputPulseWidth,
  inputPulseInterval,
  setInputPulseInterval,
  inputPulseAmpE,
  setInputPulseAmpE,
  inputPulseAmpI,
  setInputPulseAmpI,
  eeDist,
  setEeDist,
  eeTemplate,
  setEeTemplate,
  eeTemplateBlocks,
  setEeTemplateBlocks,
  eiDist,
  setEiDist,
  ieDist,
  setIeDist,
  iiDist,
  setIiDist,
  eeMean,
  setEeMean,
  eiMean,
  setEiMean,
  ieMean,
  setIeMean,
  iiMean,
  setIiMean,
  eeStd,
  setEeStd,
  eiStd,
  setEiStd,
  ieStd,
  setIeStd,
  iiStd,
  setIiStd,
  eeSigma,
  setEeSigma,
  eiSigma,
  setEiSigma,
  ieSigma,
  setIeSigma,
  iiSigma,
  setIiSigma,
  eeShape,
  setEeShape,
  eiShape,
  setEiShape,
  ieShape,
  setIeShape,
  iiShape,
  setIiShape,
  eeScale,
  setEeScale,
  eiScale,
  setEiScale,
  ieScale,
  setIeScale,
  iiScale,
  setIiScale,
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
  mqifWA,
  setMqifWA,
  mqifWVr,
  setMqifWVr,
  mqifWTau,
  setMqifWTau,
  gLHet,
  setGLHet,
  cMHet,
  setCMHet,
  vThHet,
  setVThHet,
  tRefHet,
  setTRefHet,
}: ParameterPanelProps) {
  const renderWeightBlock = (
    label: string,
    dist: WeightDistName,
    setDist: (value: WeightDistName) => void,
    mean: number,
    setMean: (value: number) => void,
    std: number,
    setStd: (value: number) => void,
    sigma: number,
    setSigma: (value: number) => void,
    shape: number,
    setShape: (value: number) => void,
    scale: number,
    setScale: (value: number) => void,
    p: number,
    setP: (value: number) => void,
    extra?: React.ReactNode
  ) => (
    <div className="space-y-2 rounded-md border border-black/10 p-2 dark:border-zinc-800">
      <div className="text-[10px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
        {label}
      </div>
      <Select
        label="dist"
        value={dist}
        options={WEIGHT_DISTS}
        onChange={(value) => setDist(value as WeightDistName)}
      />
      {(dist === "normal" || dist === "lognormal") && (
        <Slider label="mean [nS]" value={mean} min={0.0} max={1.0} step={0.0002} precision={4} onChange={setMean} showNudge />
      )}
      {dist === "normal" && (
        <Slider label="std [nS]" value={std} min={0.0} max={0.5} step={0.0002} precision={4} onChange={setStd} showNudge />
      )}
      {dist === "lognormal" && (
        <Slider label="sigma [unitless]" value={sigma} min={0.1} max={4.0} step={0.005} precision={2} onChange={setSigma} showNudge />
      )}
      {dist === "gamma" && (
        <>
          <Slider label="shape [unitless]" value={shape} min={0.1} max={10.0} step={0.01} precision={2} onChange={setShape} showNudge />
          <Slider label="scale [nS]" value={scale} min={0.0001} max={0.4} step={0.0001} precision={4} onChange={setScale} showNudge />
        </>
      )}
      {dist === "exponential" && (
        <Slider label="scale [nS]" value={scale} min={0.0001} max={0.4} step={0.0001} precision={4} onChange={setScale} showNudge />
      )}
      <Slider label="p [0-1]" value={p} min={0.0} max={1.0} step={0.001} precision={3} onChange={setP} showNudge />
      {extra}
    </div>
  );

  const configOptions = configList.length ? configList : ["(no configs)"];
  const handleConfigSelect = (value: string) => {
    if (value === "(no configs)") {
      onSelectConfig("");
    } else {
      onSelectConfig(value);
    }
  };

  return (
    <div className="h-full w-72 rounded-lg p-3 text-xs">
      <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
        Parameters
      </div>
      <div className="mt-4 h-[calc(100%-24px)] space-y-4 overflow-y-auto pr-1">
        <div className="rounded-lg border border-black/10 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-950">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Configs
          </div>
          <div className="mt-3 space-y-2">
            <Select
              label="load config"
              value={selectedConfig || configOptions[0]}
              options={configOptions}
              onChange={handleConfigSelect}
            />
            <label className="flex flex-col gap-2">
              <span className="text-[11px] text-zinc-600 dark:text-zinc-400">save as</span>
              <input
                value={saveName}
                onChange={(event) => setSaveName(event.target.value)}
                placeholder="config-name"
                className="rounded-md border border-black/10 bg-white px-2 py-1 text-xs text-black placeholder:text-zinc-400 dark:border-zinc-800 dark:bg-black dark:text-zinc-100 dark:placeholder:text-zinc-500"
              />
            </label>
            <button
              type="button"
              className="w-full rounded-md border border-black/10 bg-black px-2 py-1 text-xs text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
              onClick={onSaveConfig}
              disabled={!saveName.trim()}
            >
              Save
            </button>
            {configStatus && (
              <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                {configStatus}
              </div>
            )}
          </div>
        </div>
        <div className="space-y-3">
          <Select
            label="neuron model"
            value={neuronModel}
            options={NEURON_MODELS}
            onChange={(value) => setNeuronModel(value as NeuronModel)}
          />
          <Slider label="dt [ms]" value={dt} min={0.05} max={1.0} step={0.05} precision={2} onChange={setDt} />
          <Slider label="T [ms]" value={T} min={200} max={5000} step={50} precision={0} onChange={setT} />
          <Slider label="N_E [count]" value={nE} min={0} max={800} step={50} precision={0} onChange={setNE} />
          <Slider label="N_I [count]" value={nI} min={50} max={200} step={50} precision={0} onChange={setNI} />
          <Slider label="seed" value={seed} min={0} max={20} step={1} precision={0} onChange={setSeed} />
          <Slider label="burn-in [ms]" value={burnInMs} min={0} max={1000} step={25} precision={0} onChange={setBurnInMs} />
          <label className="flex items-center justify-between rounded-md border border-black/10 bg-white px-2 py-1 text-[11px] text-zinc-700 dark:border-zinc-800 dark:bg-black dark:text-zinc-200">
            <span>downsample spikes</span>
            <input
              type="checkbox"
              checked={downsampleEnabled}
              onChange={(event) => setDownsampleEnabled(event.target.checked)}
              className="h-4 w-4 accent-black dark:accent-white"
            />
          </label>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Inputs
          </div>
          <div className="mt-3 space-y-3">
            <Select
              label="input type"
              value={inputType}
              options={["ramp", "pulse", "pulses"]}
              onChange={(value) => setInputType(value as "ramp" | "pulse" | "pulses")}
            />
            <Slider label="I_E start [nA]" value={iEStart} min={0.0} max={4.0} step={0.01} precision={2} onChange={setIEStart} />
            <Slider label="I_E end [nA]" value={iEEnd} min={0.0} max={4.0} step={0.01} precision={2} onChange={setIEEnd} />
            <Slider label="I_I start [nA]" value={iIStart} min={0.0} max={4.0} step={0.01} precision={2} onChange={setIIStart} />
            <Slider label="I_I end [nA]" value={iIEnd} min={0.0} max={4.0} step={0.01} precision={2} onChange={setIIEnd} />
            <Slider label="I_E base [nA]" value={iEBase} min={0.0} max={4.0} step={0.01} precision={2} onChange={setIEBase} />
            <Slider label="I_I base [nA]" value={iIBase} min={0.0} max={4.0} step={0.01} precision={2} onChange={setIIBase} />
            <Slider
              label="noise std E [nA]"
              value={noiseStdE}
              min={0.0}
              max={5.0}
              step={0.05}
              precision={2}
              onChange={setNoiseStdE}
            />
            <Slider
              label="noise std I [nA]"
              value={noiseStdI}
              min={0.0}
              max={5.0}
              step={0.05}
              precision={2}
              onChange={setNoiseStdI}
            />
            <Slider label="input seed" value={inputSeed} min={0} max={20} step={1} precision={0} onChange={setInputSeed} />
            <Slider label="pulse t [ms]" value={inputPulseT} min={0} max={1000} step={10} precision={0} onChange={setInputPulseT} />
            <Slider label="pulse width [ms]" value={inputPulseWidth} min={1} max={200} step={5} precision={0} onChange={setInputPulseWidth} />
            <Slider label="pulse interval [ms]" value={inputPulseInterval} min={10} max={1000} step={10} precision={0} onChange={setInputPulseInterval} />
            <Slider label="pulse amp E [nA]" value={inputPulseAmpE} min={0} max={5} step={0.1} precision={1} onChange={setInputPulseAmpE} />
            <Slider label="pulse amp I [nA]" value={inputPulseAmpI} min={0} max={5} step={0.1} precision={1} onChange={setInputPulseAmpI} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Weights
          </div>
          <div className="mt-3 space-y-3">
            {renderWeightBlock(
              "E→E",
              eeDist,
              setEeDist,
              eeMean,
              setEeMean,
              eeStd,
              setEeStd,
              eeSigma,
              setEeSigma,
              eeShape,
              setEeShape,
              eeScale,
              setEeScale,
              pEe,
              setPEe,
              <>
                <Select label="EE template" value={eeTemplate} options={EE_TEMPLATES} onChange={setEeTemplate} />
                {eeTemplate === "feedforward_blocks" ? (
                  <Slider
                    label="EE template blocks [count]"
                    value={eeTemplateBlocks}
                    min={1}
                    max={8}
                    step={1}
                    precision={0}
                    onChange={setEeTemplateBlocks}
                  />
                ) : null}
              </>
            )}
            {renderWeightBlock(
              "E→I",
              eiDist,
              setEiDist,
              eiMean,
              setEiMean,
              eiStd,
              setEiStd,
              eiSigma,
              setEiSigma,
              eiShape,
              setEiShape,
              eiScale,
              setEiScale,
              pEi,
              setPEi
            )}
            {renderWeightBlock(
              "I→E",
              ieDist,
              setIeDist,
              ieMean,
              setIeMean,
              ieStd,
              setIeStd,
              ieSigma,
              setIeSigma,
              ieShape,
              setIeShape,
              ieScale,
              setIeScale,
              pIe,
              setPIe
            )}
            {renderWeightBlock(
              "I→I",
              iiDist,
              setIiDist,
              iiMean,
              setIiMean,
              iiStd,
              setIiStd,
              iiSigma,
              setIiSigma,
              iiShape,
              setIiShape,
              iiScale,
              setIiScale,
              pIi,
              setPIi
            )}
            <Slider label="clamp min [nS]" value={clampMin} min={0.0} max={0.2} step={0.005} precision={3} onChange={setClampMin} showNudge />
            <Slider label="weights seed" value={weightsSeed} min={0} max={20} step={1} precision={0} onChange={setWeightsSeed} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Synapse and delays
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="tau AMPA [ms]" value={tauAmpa} min={0} max={10} step={0.5} precision={1} onChange={setTauAmpa} />
            <Slider label="tau GABA [ms]" value={tauGaba} min={0} max={15} step={0.5} precision={1} onChange={setTauGaba} />
            <Slider label="delay EE [ms]" value={delayEe} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayEe} />
            <Slider label="delay EI [ms]" value={delayEi} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayEi} />
            <Slider label="delay IE [ms]" value={delayIe} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayIe} />
            <Slider label="delay II [ms]" value={delayIi} min={0.0} max={5.0} step={0.1} precision={1} onChange={setDelayIi} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Membrane
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="V_init [mV]" value={VInit} min={-80} max={-50} step={1} precision={0} onChange={setVInit} />
            <Slider label="V_th [mV]" value={VTh} min={-70} max={-40} step={1} precision={0} onChange={setVTh} />
            <Slider label="V_reset [mV]" value={VReset} min={-80} max={-50} step={1} precision={0} onChange={setVReset} />
            <Slider label="E_L [mV]" value={EL} min={-80} max={-50} step={1} precision={0} onChange={setEL} />
            <Slider label="E_e [mV]" value={Ee} min={-10} max={10} step={1} precision={0} onChange={setEe} />
            <Slider label="E_i [mV]" value={Ei} min={-100} max={-60} step={1} precision={0} onChange={setEi} />
            <Slider label="C_m E [nF]" value={CmE} min={0.2} max={2.0} step={0.1} precision={2} onChange={setCmE} />
            <Slider label="g_L E [nS]" value={gLE} min={0.01} max={0.2} step={0.005} precision={3} onChange={setGLE} />
            <Slider label="C_m I [nF]" value={CmI} min={0.2} max={2.0} step={0.1} precision={2} onChange={setCmI} />
            <Slider label="g_L I [nS]" value={gLI} min={0.01} max={0.2} step={0.005} precision={3} onChange={setGLI} />
            <Slider label="t_ref E [ms]" value={tRefE} min={0.0} max={5.0} step={0.1} precision={1} onChange={setTRefE} />
            <Slider label="t_ref I [ms]" value={tRefI} min={0.0} max={5.0} step={0.1} precision={1} onChange={setTRefI} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            HH and AdEx
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_Na [nS]" value={gNa} min={50} max={200} step={5} precision={0} onChange={setGNa} />
            <Slider label="g_K [nS]" value={gK} min={10} max={80} step={2} precision={0} onChange={setGK} />
            <Slider label="E_Na [mV]" value={ENa} min={30} max={70} step={1} precision={0} onChange={setENa} />
            <Slider label="E_K [mV]" value={EK} min={-100} max={-60} step={1} precision={0} onChange={setEK} />
            <Slider label="AdEx V_T [mV]" value={adexVT} min={-70} max={-30} step={1} precision={0} onChange={setAdexVT} />
            <Slider
              label="AdEx Delta_T [mV]"
              value={adexDeltaT}
              min={0.5}
              max={5.0}
              step={0.1}
              precision={1}
              onChange={setAdexDeltaT}
            />
            <Slider label="AdEx tau_w [ms]" value={adexTauW} min={10} max={300} step={5} precision={0} onChange={setAdexTauW} />
            <Slider label="AdEx a [nS]" value={adexA} min={0} max={10} step={0.5} precision={1} onChange={setAdexA} />
            <Slider label="AdEx b [nA]" value={adexB} min={0} max={100} step={2} precision={0} onChange={setAdexB} />
            <Slider label="AdEx V_peak [mV]" value={adexVPeak} min={0} max={40} step={1} precision={0} onChange={setAdexVPeak} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Connor-Stevens and FitzHugh
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_A [nS]" value={gA} min={0} max={80} step={2} precision={0} onChange={setGA} />
            <Slider label="FHN a [unitless]" value={fhnA} min={0.0} max={1.5} step={0.05} precision={2} onChange={setFhnA} />
            <Slider label="FHN b [unitless]" value={fhnB} min={0.0} max={2.0} step={0.05} precision={2} onChange={setFhnB} />
            <Slider label="FHN tau_w [ms]" value={fhnTauW} min={1} max={30} step={1} precision={0} onChange={setFhnTauW} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            QIF and Izhikevich
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="QIF a [nA/mV^2]" value={qifA} min={0.2} max={3.0} step={0.1} precision={1} onChange={setQifA} />
            <Slider label="QIF V_r [mV]" value={qifVr} min={-80} max={-40} step={1} precision={0} onChange={setQifVr} />
            <Slider label="QIF V_t [mV]" value={qifVt} min={-60} max={-30} step={1} precision={0} onChange={setQifVt} />
            <Slider label="Izh a [1/ms]" value={izhA} min={0.0} max={0.2} step={0.01} precision={2} onChange={setIzhA} />
            <Slider label="Izh b [nA/mV]" value={izhB} min={0.0} max={0.4} step={0.01} precision={2} onChange={setIzhB} />
            <Slider label="Izh c [mV]" value={izhC} min={-80} max={-40} step={1} precision={0} onChange={setIzhC} />
            <Slider label="Izh d [nA]" value={izhD} min={0} max={20} step={1} precision={0} onChange={setIzhD} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            MQIF
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="MQIF a [nA/mV^2]" value={mqifA} min={0.0} max={0.2} step={0.005} precision={3} onChange={setMqifA} />
            <Slider label="MQIF V_r [mV]" value={mqifVr} min={-70} max={-40} step={1} precision={0} onChange={setMqifVr} />
            <Slider label="MQIF w a [nA/mV^2]" value={mqifWA} min={0.0} max={0.2} step={0.005} precision={3} onChange={setMqifWA} />
            <Slider label="MQIF w V_r [mV]" value={mqifWVr} min={-70} max={-40} step={1} precision={0} onChange={setMqifWVr} />
            <Slider label="MQIF w tau [ms]" value={mqifWTau} min={1} max={300} step={1} precision={0} onChange={setMqifWTau} />
          </div>
        </div>

        <div className="border-t border-black/10 pt-4 dark:border-zinc-800">
          <div className="text-[11px] uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
            Heterogeneity
          </div>
          <div className="mt-3 space-y-3">
            <Slider label="g_L sd [rel]" value={gLHet} min={0.0} max={1.0} step={0.05} precision={2} onChange={setGLHet} />
            <Slider label="C_m sd [rel]" value={cMHet} min={0.0} max={1.0} step={0.05} precision={2} onChange={setCMHet} />
            <Slider label="V_th sd [mV]" value={vThHet} min={0.0} max={3.0} step={0.1} precision={1} onChange={setVThHet} />
            <Slider label="t_ref sd [ms]" value={tRefHet} min={0.0} max={2.0} step={0.1} precision={1} onChange={setTRefHet} />
          </div>
        </div>
      </div>
    </div>
  );
}
