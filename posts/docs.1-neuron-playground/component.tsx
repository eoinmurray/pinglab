
import { useMemo, useState } from "react";
import ParameterPanel from "./components/ParameterPanel";
import VoltagePlot from "./components/VoltagePlot";
import InputPlot from "./components/InputPlot";
import { buildInput, simulateHh, simulateLif, simulateMqif } from "./components/simulate";
import type { InputMode, NeuronType } from "./components/types";

export default function Component() {
  const [neuronType, setNeuronType] = useState<NeuronType>("lif");
  const [dt, setDt] = useState(0.1);
  const [T, setT] = useState(300);
  const [V_init, setVInit] = useState(-65);
  const [E_L, setEL] = useState(-65);
  const [g_L, setGL] = useState(0.05);
  const [C_m, setCM] = useState(1.0);
  const [hh_V_init, setHhVInit] = useState(-65);
  const [hh_E_L, setHhEL] = useState(-54.4);
  const [hh_E_Na, setHhENa] = useState(50);
  const [hh_E_K, setHhEK] = useState(-77);
  const [hh_g_L, setHhGL] = useState(0.3);
  const [hh_g_Na, setHhGNa] = useState(120);
  const [hh_g_K, setHhGK] = useState(36);
  const [hh_C_m, setHhCM] = useState(1.0);
  const [mqif_a1, setMqifA1] = useState(0.02);
  const [mqif_Vr1, setMqifVr1] = useState(-55);
  const [mqif_g_L, setMqifGL] = useState(0.1);
  const [mqif_C_m, setMqifCM] = useState(1.0);
  const [mqif_E_L, setMqifEL] = useState(-65);
  const [mqif_V_init, setMqifVInit] = useState(-65);
  const [mqif_V_th, setMqifVTh] = useState(-30);
  const [mqif_V_reset, setMqifVReset] = useState(-65);
  const [inputMode, setInputMode] = useState<InputMode>("tonic");
  const [I_ext_lif, setIExtLif] = useState(0.8);
  const [I_ext_hh, setIExtHh] = useState(8);
  const [I_ext_mqif, setIExtMqif] = useState(0.8);
  const I_ext =
    neuronType === "hh" ? I_ext_hh : neuronType === "mqif" ? I_ext_mqif : I_ext_lif;
  const setIExt = (value: number) => {
    if (neuronType === "hh") {
      setIExtHh(value);
    } else if (neuronType === "mqif") {
      setIExtMqif(value);
    } else {
      setIExtLif(value);
    }
  };
  const [sineAmp, setSineAmp] = useState(0.5);
  const [sineFreq, setSineFreq] = useState(8);
  const [pulseAmp, setPulseAmp] = useState(1.0);
  const [pulseStart, setPulseStart] = useState(80);
  const [pulseWidth, setPulseWidth] = useState(30);
  const [pulsesAmp, setPulsesAmp] = useState(1.0);
  const [pulsesStart, setPulsesStart] = useState(60);
  const [pulsesWidth, setPulsesWidth] = useState(20);
  const [pulsesInterval, setPulsesInterval] = useState(80);
  const [noiseAmp, setNoiseAmp] = useState(0.0);
  const [V_th, setVTh] = useState(-50);
  const [V_reset, setVReset] = useState(-65);

  const sim = useMemo(() => {
    const steps = Math.max(1, Math.floor(T / dt));
    const times = Array.from({ length: steps }, (_, i) => i * dt);
    const input = buildInput(
      {
        mode: inputMode,
        tonic: I_ext,
        sineAmp,
        sineFreq,
        pulseAmp,
        pulseStart,
        pulseWidth,
        pulsesAmp,
        pulsesStart,
        pulsesWidth,
        pulsesInterval,
        noiseAmp,
      },
      times
    );
    if (neuronType === "hh") {
      return simulateHh({
        dt,
        T,
        V_init: hh_V_init,
        input,
        C_m: hh_C_m,
        g_L: hh_g_L,
        g_Na: hh_g_Na,
        g_K: hh_g_K,
        E_L: hh_E_L,
        E_Na: hh_E_Na,
        E_K: hh_E_K,
      });
    }
    if (neuronType === "mqif") {
      return simulateMqif({
        dt,
        T,
        V_init: mqif_V_init,
        input,
        C_m: mqif_C_m,
        g_L: mqif_g_L,
        E_L: mqif_E_L,
        V_th: mqif_V_th,
        V_reset: mqif_V_reset,
        a_terms: [mqif_a1],
        V_r_terms: [mqif_Vr1],
      });
    }
    return simulateLif({
      dt,
      T,
      V_init,
      E_L,
      g_L,
      C_m,
      input,
      V_th,
      V_reset,
    });
  }, [
    neuronType,
    dt,
    T,
    V_init,
    E_L,
    g_L,
    C_m,
    I_ext,
    V_th,
    V_reset,
    inputMode,
    sineAmp,
    sineFreq,
    pulseAmp,
    pulseStart,
    pulseWidth,
    pulsesAmp,
    pulsesStart,
    pulsesWidth,
    pulsesInterval,
    noiseAmp,
    hh_V_init,
    hh_E_L,
    hh_E_Na,
    hh_E_K,
    hh_g_L,
    hh_g_Na,
    hh_g_K,
    hh_C_m,
    mqif_a1,
    mqif_Vr1,
    mqif_g_L,
    mqif_C_m,
    mqif_E_L,
    mqif_V_init,
    mqif_V_th,
    mqif_V_reset,
  ]);

  const width = 720;
  const heightVoltage = 250;
  const heightInput = 220;
  const margin = { top: 24, right: 28, bottom: 44, left: 64 };
  const inputDomain: [number, number] =
    neuronType === "hh" ? [0, 20] : neuronType === "mqif" ? [0, 10] : [-2, 5];
  const voltageDomain: [number, number] =
    neuronType === "hh" ? [-200, 200] : neuronType === "mqif" ? [-80, 20] : [-70, -40];

  return (
    <div className="box-border h-screen w-screen bg-white p-3 font-mono text-black dark:bg-black dark:text-zinc-100">
      <div className="flex h-full flex-row gap-3">
        <ParameterPanel
          neuronType={neuronType}
          setNeuronType={setNeuronType}
          inputMode={inputMode}
          setInputMode={setInputMode}
          I_ext={I_ext}
          setIExt={setIExt}
          noiseAmp={noiseAmp}
          setNoiseAmp={setNoiseAmp}
          sineAmp={sineAmp}
          setSineAmp={setSineAmp}
          sineFreq={sineFreq}
          setSineFreq={setSineFreq}
          pulseAmp={pulseAmp}
          setPulseAmp={setPulseAmp}
          pulseStart={pulseStart}
          setPulseStart={setPulseStart}
          pulseWidth={pulseWidth}
          setPulseWidth={setPulseWidth}
          pulsesAmp={pulsesAmp}
          setPulsesAmp={setPulsesAmp}
          pulsesStart={pulsesStart}
          setPulsesStart={setPulsesStart}
          pulsesWidth={pulsesWidth}
          setPulsesWidth={setPulsesWidth}
          pulsesInterval={pulsesInterval}
          setPulsesInterval={setPulsesInterval}
          g_L={g_L}
          setGL={setGL}
          C_m={C_m}
          setCM={setCM}
          E_L={E_L}
          setEL={setEL}
          V_init={V_init}
          setVInit={setVInit}
          V_th={V_th}
          setVTh={setVTh}
          V_reset={V_reset}
          setVReset={setVReset}
          hh_C_m={hh_C_m}
          setHhCM={setHhCM}
          hh_g_Na={hh_g_Na}
          setHhGNa={setHhGNa}
          hh_g_K={hh_g_K}
          setHhGK={setHhGK}
          hh_g_L={hh_g_L}
          setHhGL={setHhGL}
          hh_E_Na={hh_E_Na}
          setHhENa={setHhENa}
          hh_E_K={hh_E_K}
          setHhEK={setHhEK}
          hh_E_L={hh_E_L}
          setHhEL={setHhEL}
          hh_V_init={hh_V_init}
          setHhVInit={setHhVInit}
          mqif_a1={mqif_a1}
          setMqifA1={setMqifA1}
          mqif_Vr1={mqif_Vr1}
          setMqifVr1={setMqifVr1}
          mqif_g_L={mqif_g_L}
          setMqifGL={setMqifGL}
          mqif_C_m={mqif_C_m}
          setMqifCM={setMqifCM}
          mqif_E_L={mqif_E_L}
          setMqifEL={setMqifEL}
          mqif_V_init={mqif_V_init}
          setMqifVInit={setMqifVInit}
          mqif_V_th={mqif_V_th}
          setMqifVTh={setMqifVTh}
          mqif_V_reset={mqif_V_reset}
          setMqifVReset={setMqifVReset}
          dt={dt}
          setDt={setDt}
          T={T}
          setT={setT}
        />

        <div className="flex h-full flex-1 flex-col gap-2">
          <VoltagePlot
            times={sim.times}
            voltages={sim.voltages}
            T={T}
            threshold={neuronType === "hh" ? null : neuronType === "mqif" ? mqif_V_th : V_th}
            yDomain={voltageDomain}
            width={width}
            height={heightVoltage}
            margin={margin}
          />
          <InputPlot
            times={sim.times}
            input={sim.input}
            T={T}
            width={width}
            height={heightInput}
            margin={margin}
            yDomain={inputDomain}
          />
        </div>
      </div>
    </div>
  );
}
