import ParamLabel from "./ParamLabel";
import { infoText } from "./infoText";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import type { InputMode, NeuronType } from "./types";

type ParameterPanelProps = {
  neuronType: NeuronType;
  setNeuronType: (value: NeuronType) => void;
  inputMode: InputMode;
  setInputMode: (mode: InputMode) => void;
  I_ext: number;
  setIExt: (value: number) => void;
  noiseAmp: number;
  setNoiseAmp: (value: number) => void;
  sineAmp: number;
  setSineAmp: (value: number) => void;
  sineFreq: number;
  setSineFreq: (value: number) => void;
  pulseAmp: number;
  setPulseAmp: (value: number) => void;
  pulseStart: number;
  setPulseStart: (value: number) => void;
  pulseWidth: number;
  setPulseWidth: (value: number) => void;
  pulsesAmp: number;
  setPulsesAmp: (value: number) => void;
  pulsesStart: number;
  setPulsesStart: (value: number) => void;
  pulsesWidth: number;
  setPulsesWidth: (value: number) => void;
  pulsesInterval: number;
  setPulsesInterval: (value: number) => void;
  g_L: number;
  setGL: (value: number) => void;
  C_m: number;
  setCM: (value: number) => void;
  E_L: number;
  setEL: (value: number) => void;
  V_init: number;
  setVInit: (value: number) => void;
  V_th: number;
  setVTh: (value: number) => void;
  V_reset: number;
  setVReset: (value: number) => void;
  hh_C_m: number;
  setHhCM: (value: number) => void;
  hh_g_Na: number;
  setHhGNa: (value: number) => void;
  hh_g_K: number;
  setHhGK: (value: number) => void;
  hh_g_L: number;
  setHhGL: (value: number) => void;
  hh_E_Na: number;
  setHhENa: (value: number) => void;
  hh_E_K: number;
  setHhEK: (value: number) => void;
  hh_E_L: number;
  setHhEL: (value: number) => void;
  hh_V_init: number;
  setHhVInit: (value: number) => void;
  mqif_a1: number;
  setMqifA1: (value: number) => void;
  mqif_Vr1: number;
  setMqifVr1: (value: number) => void;
  mqif_g_L: number;
  setMqifGL: (value: number) => void;
  mqif_C_m: number;
  setMqifCM: (value: number) => void;
  mqif_E_L: number;
  setMqifEL: (value: number) => void;
  mqif_V_init: number;
  setMqifVInit: (value: number) => void;
  mqif_V_th: number;
  setMqifVTh: (value: number) => void;
  mqif_V_reset: number;
  setMqifVReset: (value: number) => void;
  dt: number;
  setDt: (value: number) => void;
  T: number;
  setT: (value: number) => void;
};

export default function ParameterPanel({
  neuronType,
  setNeuronType,
  inputMode,
  setInputMode,
  I_ext,
  setIExt,
  noiseAmp,
  setNoiseAmp,
  sineAmp,
  setSineAmp,
  sineFreq,
  setSineFreq,
  pulseAmp,
  setPulseAmp,
  pulseStart,
  setPulseStart,
  pulseWidth,
  setPulseWidth,
  pulsesAmp,
  setPulsesAmp,
  pulsesStart,
  setPulsesStart,
  pulsesWidth,
  setPulsesWidth,
  pulsesInterval,
  setPulsesInterval,
  g_L,
  setGL,
  C_m,
  setCM,
  E_L,
  setEL,
  V_init,
  setVInit,
  V_th,
  setVTh,
  V_reset,
  setVReset,
  hh_C_m,
  setHhCM,
  hh_g_Na,
  setHhGNa,
  hh_g_K,
  setHhGK,
  hh_g_L,
  setHhGL,
  hh_E_Na,
  setHhENa,
  hh_E_K,
  setHhEK,
  hh_E_L,
  setHhEL,
  hh_V_init,
  setHhVInit,
  mqif_a1,
  setMqifA1,
  mqif_Vr1,
  setMqifVr1,
  mqif_g_L,
  setMqifGL,
  mqif_C_m,
  setMqifCM,
  mqif_E_L,
  setMqifEL,
  mqif_V_init,
  setMqifVInit,
  mqif_V_th,
  setMqifVTh,
  mqif_V_reset,
  setMqifVReset,
  dt,
  setDt,
  T,
  setT,
}: ParameterPanelProps) {
  return (
    <div className="flex w-[300px] max-h-[calc(100vh-48px)] flex-none flex-col gap-2 overflow-auto rounded-xl border border-black bg-white p-3 dark:border-zinc-100 dark:bg-black">
      <div className="text-[11px] uppercase tracking-[0.2em] text-neutral-500 dark:text-neutral-400">
        Parameters
      </div>
      <div className="flex flex-col gap-1.5">
        <ParamLabel
          label="Neuron type"
          value={neuronType.toUpperCase()}
          info={infoText.neuronType}
        />
        <Select value={neuronType} onValueChange={(value) => setNeuronType(value as NeuronType)}>
          <SelectTrigger aria-label="Neuron type">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="lif">lif</SelectItem>
            <SelectItem value="hh">hh</SelectItem>
            <SelectItem value="mqif">mqif</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <ParamLabel label="Input mode" value={inputMode} info={infoText.inputMode} />
        <Select value={inputMode} onValueChange={(value) => setInputMode(value as InputMode)}>
          <SelectTrigger aria-label="Input mode">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="tonic">tonic</SelectItem>
            <SelectItem value="sine">sine</SelectItem>
            <SelectItem value="pulse">pulse</SelectItem>
            <SelectItem value="pulses">pulses</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <ParamLabel
          label="Tonic I_ext"
          value={I_ext.toFixed(2)}
          info={infoText.tonic}
        />
        <input
          type="range"
          min="0"
          max={neuronType === "hh" ? "20" : neuronType === "mqif" ? "10" : "2"}
          step={neuronType === "hh" ? "0.2" : neuronType === "mqif" ? "0.1" : "0.05"}
          value={I_ext}
          className="w-full accent-black dark:accent-white"
          onChange={(e) => setIExt(Number(e.target.value))}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <ParamLabel label="Noise amp" value={noiseAmp.toFixed(2)} info={infoText.noiseAmp} />
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={noiseAmp}
          className="w-full accent-black dark:accent-white"
          onChange={(e) => setNoiseAmp(Number(e.target.value))}
        />
      </div>
      {inputMode === "sine" && (
        <>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Sine amp"
              value={sineAmp.toFixed(2)}
              info={infoText.sineAmp}
            />
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={sineAmp}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setSineAmp(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Sine freq"
              value={`${sineFreq.toFixed(1)} Hz`}
              info={infoText.sineFreq}
            />
            <input
              type="range"
              min="0.5"
              max="40"
              step="0.5"
              value={sineFreq}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setSineFreq(Number(e.target.value))}
            />
          </div>
        </>
      )}
      {inputMode === "pulse" && (
        <>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulse amp"
              value={pulseAmp.toFixed(2)}
              info={infoText.pulseAmp}
            />
            <input
              type="range"
              min="0"
              max="3"
              step="0.05"
              value={pulseAmp}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulseAmp(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulse start"
              value={`${pulseStart.toFixed(0)} ms`}
              info={infoText.pulseStart}
            />
            <input
              type="range"
              min="0"
              max={Math.max(10, T - 10)}
              step="5"
              value={pulseStart}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulseStart(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulse width"
              value={`${pulseWidth.toFixed(0)} ms`}
              info={infoText.pulseWidth}
            />
            <input
              type="range"
              min="5"
              max={Math.max(20, T / 2)}
              step="5"
              value={pulseWidth}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulseWidth(Number(e.target.value))}
            />
          </div>
        </>
      )}
      {inputMode === "pulses" && (
        <>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulses amp"
              value={pulsesAmp.toFixed(2)}
              info={infoText.pulsesAmp}
            />
            <input
              type="range"
              min="0"
              max="3"
              step="0.05"
              value={pulsesAmp}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulsesAmp(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulses start"
              value={`${pulsesStart.toFixed(0)} ms`}
              info={infoText.pulsesStart}
            />
            <input
              type="range"
              min="0"
              max={Math.max(10, T - 10)}
              step="5"
              value={pulsesStart}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulsesStart(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulses width"
              value={`${pulsesWidth.toFixed(0)} ms`}
              info={infoText.pulsesWidth}
            />
            <input
              type="range"
              min="5"
              max={Math.max(20, T / 2)}
              step="5"
              value={pulsesWidth}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulsesWidth(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="Pulses interval"
              value={`${pulsesInterval.toFixed(0)} ms`}
              info={infoText.pulsesInterval}
            />
            <input
              type="range"
              min="20"
              max={Math.max(40, T)}
              step="10"
              value={pulsesInterval}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setPulsesInterval(Number(e.target.value))}
            />
          </div>
        </>
      )}
      {neuronType === "lif" && (
        <>
          <div className="flex flex-col gap-1.5">
            <ParamLabel label="g_L" value={g_L.toFixed(2)} info={infoText.g_L} />
            <input
              type="range"
              min="0.01"
              max="0.2"
              step="0.01"
              value={g_L}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setGL(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel label="C_m" value={C_m.toFixed(2)} info={infoText.C_m} />
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value={C_m}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setCM(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="E_L"
              value={`${E_L.toFixed(1)} mV`}
              info={infoText.E_L}
            />
            <input
              type="range"
              min="-75"
              max="-55"
              step="0.5"
              value={E_L}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setEL(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_init"
              value={`${V_init.toFixed(1)} mV`}
              info={infoText.V_init}
            />
            <input
              type="range"
              min="-75"
              max="-55"
              step="0.5"
              value={V_init}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setVInit(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_th"
              value={`${V_th.toFixed(1)} mV`}
              info={infoText.V_th}
            />
            <input
              type="range"
              min="-55"
              max="-40"
              step="0.5"
              value={V_th}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setVTh(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_reset"
              value={`${V_reset.toFixed(1)} mV`}
              info={infoText.V_reset}
            />
            <input
              type="range"
              min="-75"
              max="-55"
              step="0.5"
              value={V_reset}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setVReset(Number(e.target.value))}
            />
          </div>
        </>
      )}
      {neuronType === "hh" && (
        <>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="g_Na"
              value={hh_g_Na.toFixed(1)}
              info={infoText.hh_g_Na}
            />
            <input
              type="range"
              min="0"
              max="200"
              step="1"
              value={hh_g_Na}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhGNa(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="g_K"
              value={hh_g_K.toFixed(1)}
              info={infoText.hh_g_K}
            />
            <input
              type="range"
              min="0"
              max="60"
              step="1"
              value={hh_g_K}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhGK(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="g_L"
              value={hh_g_L.toFixed(2)}
              info={infoText.hh_g_L}
            />
            <input
              type="range"
              min="0"
              max="1.0"
              step="0.01"
              value={hh_g_L}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhGL(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="C_m"
              value={hh_C_m.toFixed(2)}
              info={infoText.hh_C_m}
            />
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value={hh_C_m}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhCM(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="E_Na"
              value={`${hh_E_Na.toFixed(0)} mV`}
              info={infoText.hh_E_Na}
            />
            <input
              type="range"
              min="30"
              max="70"
              step="1"
              value={hh_E_Na}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhENa(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="E_K"
              value={`${hh_E_K.toFixed(0)} mV`}
              info={infoText.hh_E_K}
            />
            <input
              type="range"
              min="-100"
              max="-50"
              step="1"
              value={hh_E_K}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhEK(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="E_L"
              value={`${hh_E_L.toFixed(1)} mV`}
              info={infoText.hh_E_L}
            />
            <input
              type="range"
              min="-70"
              max="-40"
              step="0.5"
              value={hh_E_L}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhEL(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_init"
              value={`${hh_V_init.toFixed(1)} mV`}
              info={infoText.hh_V_init}
            />
            <input
              type="range"
              min="-80"
              max="-40"
              step="0.5"
              value={hh_V_init}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setHhVInit(Number(e.target.value))}
            />
          </div>
        </>
      )}
      {neuronType === "mqif" && (
        <>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="a1"
              value={mqif_a1.toFixed(2)}
              info={infoText.mqif_a1}
            />
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={mqif_a1}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifA1(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_r1"
              value={`${mqif_Vr1.toFixed(1)} mV`}
              info={infoText.mqif_Vr1}
            />
            <input
              type="range"
              min="-80"
              max="-30"
              step="0.5"
              value={mqif_Vr1}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifVr1(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="g_L"
              value={mqif_g_L.toFixed(2)}
              info={infoText.mqif_g_L}
            />
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.01"
              value={mqif_g_L}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifGL(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="C_m"
              value={mqif_C_m.toFixed(2)}
              info={infoText.mqif_C_m}
            />
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value={mqif_C_m}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifCM(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="E_L"
              value={`${mqif_E_L.toFixed(1)} mV`}
              info={infoText.mqif_E_L}
            />
            <input
              type="range"
              min="-80"
              max="-40"
              step="0.5"
              value={mqif_E_L}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifEL(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_init"
              value={`${mqif_V_init.toFixed(1)} mV`}
              info={infoText.mqif_V_init}
            />
            <input
              type="range"
              min="-80"
              max="-40"
              step="0.5"
              value={mqif_V_init}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifVInit(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_th"
              value={`${mqif_V_th.toFixed(1)} mV`}
              info={infoText.mqif_V_th}
            />
            <input
              type="range"
              min="-60"
              max="-20"
              step="0.5"
              value={mqif_V_th}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifVTh(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <ParamLabel
              label="V_reset"
              value={`${mqif_V_reset.toFixed(1)} mV`}
              info={infoText.mqif_V_reset}
            />
            <input
              type="range"
              min="-80"
              max="-40"
              step="0.5"
              value={mqif_V_reset}
              className="w-full accent-black dark:accent-white"
              onChange={(e) => setMqifVReset(Number(e.target.value))}
            />
          </div>
        </>
      )}
      <div className="flex flex-col gap-1.5">
        <ParamLabel label="dt" value={`${dt.toFixed(2)} ms`} info={infoText.dt} />
        <input
          type="range"
          min="0.05"
          max="0.5"
          step="0.01"
          value={dt}
          className="w-full accent-black dark:accent-white"
          onChange={(e) => setDt(Number(e.target.value))}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <ParamLabel label="T" value={`${T.toFixed(0)} ms`} info={infoText.T} />
        <input
          type="range"
          min="100"
          max="800"
          step="20"
          value={T}
          className="w-full accent-black dark:accent-white"
          onChange={(e) => setT(Number(e.target.value))}
        />
      </div>
    </div>
  );
}
