import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear } from "@visx/scale";
import ParameterPanel, { type NeuronModel } from "./components/ParameterPanel";
import MembranePotentialPlot from "./components/MembranePotentialPlot";
import InputTracePlot from "./components/InputTracePlot";
import PopulationRatePlot from "./components/PopulationRatePlot";
import RasterPlot from "./components/RasterPlot";

type SpikesResponse = {
  times: number[];
  ids: number[];
  types: number[];
};

type RunResponse = {
  spikes: SpikesResponse;
  runtime_ms: number;
  num_steps: number;
  num_spikes: number;
  rhythmicity: number;
  mean_rate_E: number;
  mean_rate_I: number;
  population_rate_t_ms: number[];
  population_rate_hz_E: number[];
  population_rate_hz_I: number[];
  membrane_t_ms: number[];
  membrane_V_E: number[];
  membrane_V_I: number[];
  input_t_ms: number[];
  input_mean_E: number[];
  input_mean_I: number[];
};

const API_URL = "http://localhost:8000/run";

const defaultWidth = 760;
const defaultHeight = 420;
const margin = { top: 24, right: 20, bottom: 44, left: 54 };
const rateMargin = { top: 8, right: 16, bottom: 28, left: 42 };

export default function Component() {
  const [data, setData] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [neuronModel, setNeuronModel] = useState<NeuronModel>("mqif");

  const [dt, setDt] = useState(0.1);
  const [T, setT] = useState(1000);
  const [nE, setNE] = useState(800);
  const [nI, setNI] = useState(200);
  const [seed, setSeed] = useState(0);

  const [noiseStd, setNoiseStd] = useState(0.0);
  const [inputSeed, setInputSeed] = useState(0);
  const [inputType, setInputType] = useState<"ramp" | "pulse" | "pulses">("ramp");
  const [iEStart, setIEStart] = useState(2.0);
  const [iEEnd, setIEEnd] = useState(2.0);
  const [iIStart, setIIStart] = useState(2.0);
  const [iIEnd, setIIEnd] = useState(2.0);
  const [iEBase, setIEBase] = useState(0.0);
  const [iIBase, setIIBase] = useState(0.0);
  const [inputPulseT, setInputPulseT] = useState(200.0);
  const [inputPulseWidth, setInputPulseWidth] = useState(20.0);
  const [inputPulseInterval, setInputPulseInterval] = useState(100.0);
  const [inputPulseAmpE, setInputPulseAmpE] = useState(1.0);
  const [inputPulseAmpI, setInputPulseAmpI] = useState(1.0);

  const [gEeMean, setGEeMean] = useState(0.003);
  const [gEiMean, setGEiMean] = useState(0.005);
  const [gIeMean, setGIeMean] = useState(0.005);
  const [gIiMean, setGIiMean] = useState(0.005);
  const [gEeStd, setGEeStd] = useState(0.002);
  const [gEiStd, setGEiStd] = useState(0.002);
  const [gIeStd, setGIeStd] = useState(0.002);
  const [gIiStd, setGIiStd] = useState(0.002);
  const [pEe, setPEe] = useState(1.0);
  const [pEi, setPEi] = useState(1.0);
  const [pIe, setPIe] = useState(1.0);
  const [pIi, setPIi] = useState(1.0);
  const [clampMin, setClampMin] = useState(0.0);
  const [weightsSeed, setWeightsSeed] = useState(0);

  const [delayEe, setDelayEe] = useState(0.5);
  const [delayEi, setDelayEi] = useState(0.5);
  const [delayIe, setDelayIe] = useState(1.2);
  const [delayIi, setDelayIi] = useState(0.5);

  const [VInit, setVInit] = useState(-65.0);
  const [EL, setEL] = useState(-65.0);
  const [Ee, setEe] = useState(0.0);
  const [Ei, setEi] = useState(-80.0);
  const [CmE, setCmE] = useState(1.0);
  const [gLE, setGLE] = useState(0.05);
  const [CmI, setCmI] = useState(1.0);
  const [gLI, setGLI] = useState(0.1);
  const [VTh, setVTh] = useState(-50.0);
  const [VReset, setVReset] = useState(-65.0);
  const [tRefE, setTRefE] = useState(3.0);
  const [tRefI, setTRefI] = useState(1.5);
  const [tauAmpa, setTauAmpa] = useState(2.0);
  const [tauGaba, setTauGaba] = useState(6.5);

  const [gNa, setGNa] = useState(120.0);
  const [gK, setGK] = useState(36.0);
  const [ENa, setENa] = useState(50.0);
  const [EK, setEK] = useState(-77.0);

  const [adexVT, setAdexVT] = useState(-50.0);
  const [adexDeltaT, setAdexDeltaT] = useState(2.0);
  const [adexTauW, setAdexTauW] = useState(100.0);
  const [adexA, setAdexA] = useState(2.0);
  const [adexB, setAdexB] = useState(60.0);
  const [adexVPeak, setAdexVPeak] = useState(20.0);

  const [gA, setGA] = useState(47.7);

  const [fhnA, setFhnA] = useState(0.7);
  const [fhnB, setFhnB] = useState(0.8);
  const [fhnTauW, setFhnTauW] = useState(12.5);

  const [mqifA, setMqifA] = useState(0.02);
  const [mqifVr, setMqifVr] = useState(-55.0);
  const [mqifWA, setMqifWA] = useState(0.02);
  const [mqifWVr, setMqifWVr] = useState(-55.0);
  const [mqifWTau, setMqifWTau] = useState(100.0);

  const [qifA, setQifA] = useState(1.0);
  const [qifVr, setQifVr] = useState(-60.0);
  const [qifVt, setQifVt] = useState(-45.0);

  const [izhA, setIzhA] = useState(0.02);
  const [izhB, setIzhB] = useState(0.2);
  const [izhC, setIzhC] = useState(-65.0);
  const [izhD, setIzhD] = useState(8.0);

  const [gLHet, setGLHet] = useState(0.0);
  const [cMHet, setCMHet] = useState(0.0);
  const [vThHet, setVThHet] = useState(0.0);
  const [tRefHet, setTRefHet] = useState(0.0);

  const plotRef = useRef<HTMLDivElement | null>(null);
  const [plotSize, setPlotSize] = useState({
    width: defaultWidth,
    height: defaultHeight,
  });
  const ratePlotRef = useRef<HTMLDivElement | null>(null);
  const [ratePlotSize, setRatePlotSize] = useState({
    width: defaultWidth,
    height: 192,
  });
  const membraneEPlotRef = useRef<HTMLDivElement | null>(null);
  const membraneIPlotRef = useRef<HTMLDivElement | null>(null);
  const inputPlotRef = useRef<HTMLDivElement | null>(null);
  const [membraneEPlotSize, setMembraneEPlotSize] = useState({
    width: defaultWidth,
    height: 192,
  });
  const [membraneIPlotSize, setMembraneIPlotSize] = useState({
    width: defaultWidth,
    height: 192,
  });
  const [inputPlotSize, setInputPlotSize] = useState({
    width: defaultWidth,
    height: 192,
  });

  const totalN = nE + nI;

  useEffect(() => {
    if (!plotRef.current) {
      return;
    }
    const target = plotRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setPlotSize({
        width: Math.max(320, Math.floor(nextWidth)),
        height: Math.max(240, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!ratePlotRef.current) {
      return;
    }
    const target = ratePlotRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setRatePlotSize({
        width: Math.max(320, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!membraneEPlotRef.current) {
      return;
    }
    const target = membraneEPlotRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setMembraneEPlotSize({
        width: Math.max(240, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!membraneIPlotRef.current) {
      return;
    }
    const target = membraneIPlotRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setMembraneIPlotSize({
        width: Math.max(240, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!inputPlotRef.current) {
      return;
    }
    const target = inputPlotRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setInputPlotSize({
        width: Math.max(240, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let ignore = false;
    const timer = setTimeout(() => {
      const run = async () => {
        try {
          setLoading(true);
            const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              config: {
                dt,
                T,
                N_E: nE,
                N_I: nI,
                seed,
                neuron_model: neuronModel,
                delay_ee: delayEe,
                delay_ei: delayEi,
                delay_ie: delayIe,
                delay_ii: delayIi,
                V_init: VInit,
                E_L: EL,
                E_e: Ee,
                E_i: Ei,
                C_m_E: CmE,
                g_L_E: gLE,
                C_m_I: CmI,
                g_L_I: gLI,
                V_th: VTh,
                V_reset: VReset,
                t_ref_E: tRefE,
                t_ref_I: tRefI,
                tau_ampa: tauAmpa,
                tau_gaba: tauGaba,
                g_Na: gNa,
                g_K: gK,
                E_Na: ENa,
                E_K: EK,
                adex_V_T: adexVT,
                adex_delta_T: adexDeltaT,
                adex_tau_w: adexTauW,
                adex_a: adexA,
                adex_b: adexB,
                adex_V_peak: adexVPeak,
                g_A: gA,
                fhn_a: fhnA,
                fhn_b: fhnB,
                fhn_tau_w: fhnTauW,
                mqif_a: [mqifA],
                mqif_Vr: [mqifVr],
                mqif_w_a: [mqifWA],
                mqif_w_Vr: [mqifWVr],
                mqif_w_tau: [mqifWTau],
                qif_a: qifA,
                qif_Vr: qifVr,
                qif_Vt: qifVt,
                izh_a: izhA,
                izh_b: izhB,
                izh_c: izhC,
                izh_d: izhD,
                g_L_heterogeneity_sd: gLHet,
                C_m_heterogeneity_sd: cMHet,
                V_th_heterogeneity_sd: vThHet,
                t_ref_heterogeneity_sd: tRefHet,
              },
              inputs: {
                input_type: inputType,
                I_E_start: iEStart,
                I_E_end: iEEnd,
                I_I_start: iIStart,
                I_I_end: iIEnd,
                I_E_base: iEBase,
                I_I_base: iIBase,
                noise_std: noiseStd,
                seed: inputSeed,
                pulse_t_ms: inputPulseT,
                pulse_width_ms: inputPulseWidth,
                pulse_interval_ms: inputPulseInterval,
                pulse_amp_E: inputPulseAmpE,
                pulse_amp_I: inputPulseAmpI,
              },
              weights: {
                mean_ee: gEeMean,
                mean_ei: gEiMean,
                mean_ie: gIeMean,
                mean_ii: gIiMean,
                std_ee: gEeStd,
                std_ei: gEiStd,
                std_ie: gIeStd,
                std_ii: gIiStd,
                p_ee: pEe,
                p_ei: pEi,
                p_ie: pIe,
                p_ii: pIi,
                clamp_min: clampMin,
                seed: weightsSeed,
              },
            }),
          });
          if (!response.ok) {
            throw new Error(`API request failed (${response.status})`);
          }
          const json = (await response.json()) as RunResponse;
          if (!ignore) {
            setData(json);
            setError(null);
          }
        } catch (err) {
          if (!ignore) {
            setError(err instanceof Error ? err.message : "Failed to load data");
          }
        } finally {
          if (!ignore) {
            setLoading(false);
          }
        }
      };
      run();
    }, 300);
    return () => {
      ignore = true;
      clearTimeout(timer);
    };
  }, [
    dt,
    T,
    nE,
    nI,
    seed,
    neuronModel,
    delayEe,
    delayEi,
    delayIe,
    delayIi,
    VInit,
    EL,
    Ee,
    Ei,
    CmE,
    gLE,
    CmI,
    gLI,
    VTh,
    VReset,
    tRefE,
    tRefI,
    tauAmpa,
    tauGaba,
    gNa,
    gK,
    ENa,
    EK,
    adexVT,
    adexDeltaT,
    adexTauW,
    adexA,
    adexB,
    adexVPeak,
    gA,
    fhnA,
    fhnB,
    fhnTauW,
    mqifA,
    mqifVr,
    mqifWA,
    mqifWVr,
    mqifWTau,
    qifA,
    qifVr,
    qifVt,
    izhA,
    izhB,
    izhC,
    izhD,
    gLHet,
    cMHet,
    vThHet,
    tRefHet,
    noiseStd,
    inputSeed,
    inputType,
    iEStart,
    iEEnd,
    iIStart,
    iIEnd,
    iEBase,
    iIBase,
    inputPulseT,
    inputPulseWidth,
    inputPulseInterval,
    inputPulseAmpE,
    inputPulseAmpI,
    gEeMean,
    gEiMean,
    gIeMean,
    gIiMean,
    gEeStd,
    gEiStd,
    gIeStd,
    gIiStd,
    pEe,
    pEi,
    pIe,
    pIi,
    clampMin,
    weightsSeed,
  ]);

  const innerWidth = plotSize.width - margin.left - margin.right;
  const innerHeight = plotSize.height - margin.top - margin.bottom;
  const rateInnerWidth = ratePlotSize.width - rateMargin.left - rateMargin.right;
  const rateInnerHeight = ratePlotSize.height - rateMargin.top - rateMargin.bottom;
  const membraneEInnerWidth = membraneEPlotSize.width - rateMargin.left - rateMargin.right;
  const membraneEInnerHeight = membraneEPlotSize.height - rateMargin.top - rateMargin.bottom;
  const membraneIInnerWidth = membraneIPlotSize.width - rateMargin.left - rateMargin.right;
  const membraneIInnerHeight = membraneIPlotSize.height - rateMargin.top - rateMargin.bottom;
  const inputInnerWidth = inputPlotSize.width - rateMargin.left - rateMargin.right;
  const inputInnerHeight = inputPlotSize.height - rateMargin.top - rateMargin.bottom;

  const xScale = useMemo(
    () =>
      scaleLinear({
        domain: [0, T],
        range: [0, innerWidth],
      }),
    [T, innerWidth]
  );
  const yScale = useMemo(
    () =>
      scaleLinear({
        domain: [0, totalN],
        range: [innerHeight, 0],
      }),
    [totalN, innerHeight]
  );

  const spikeCounts = useMemo(() => {
    const spikes = data?.spikes;
    if (!spikes) {
      return { eCount: 0, iCount: 0 };
    }
    let eCount = 0;
    let iCount = 0;
    for (let i = 0; i < spikes.types.length; i += 1) {
      if (spikes.types[i] === 0) {
        eCount += 1;
      } else if (spikes.types[i] === 1) {
        iCount += 1;
      }
    }
    return { eCount, iCount };
  }, [data]);

  const Panel = <ParameterPanel
    neuronModel={neuronModel}
    setNeuronModel={setNeuronModel}
    dt={dt}
    setDt={setDt}
    T={T}
    setT={setT}
    nE={nE}
    setNE={setNE}
    nI={nI}
    setNI={setNI}
    seed={seed}
    setSeed={setSeed}
    noiseStd={noiseStd}
    setNoiseStd={setNoiseStd}
    inputSeed={inputSeed}
    setInputSeed={setInputSeed}
    inputType={inputType}
    setInputType={setInputType}
    iEStart={iEStart}
    setIEStart={setIEStart}
    iEEnd={iEEnd}
    setIEEnd={setIEEnd}
    iIStart={iIStart}
    setIIStart={setIIStart}
    iIEnd={iIEnd}
    setIIEnd={setIIEnd}
    iEBase={iEBase}
    setIEBase={setIEBase}
    iIBase={iIBase}
    setIIBase={setIIBase}
    inputPulseT={inputPulseT}
    setInputPulseT={setInputPulseT}
    inputPulseWidth={inputPulseWidth}
    setInputPulseWidth={setInputPulseWidth}
    inputPulseInterval={inputPulseInterval}
    setInputPulseInterval={setInputPulseInterval}
    inputPulseAmpE={inputPulseAmpE}
    setInputPulseAmpE={setInputPulseAmpE}
    inputPulseAmpI={inputPulseAmpI}
    setInputPulseAmpI={setInputPulseAmpI}
    gEeMean={gEeMean}
    setGEeMean={setGEeMean}
    gEiMean={gEiMean}
    setGEiMean={setGEiMean}
    gIeMean={gIeMean}
    setGIeMean={setGIeMean}
    gIiMean={gIiMean}
    setGIiMean={setGIiMean}
    gEeStd={gEeStd}
    setGEeStd={setGEeStd}
    gEiStd={gEiStd}
    setGEiStd={setGEiStd}
    gIeStd={gIeStd}
    setGIeStd={setGIeStd}
    gIiStd={gIiStd}
    setGIiStd={setGIiStd}
    pEe={pEe}
    setPEe={setPEe}
    pEi={pEi}
    setPEi={setPEi}
    pIe={pIe}
    setPIe={setPIe}
    pIi={pIi}
    setPIi={setPIi}
    clampMin={clampMin}
    setClampMin={setClampMin}
    weightsSeed={weightsSeed}
    setWeightsSeed={setWeightsSeed}
    delayEe={delayEe}
    setDelayEe={setDelayEe}
    delayEi={delayEi}
    setDelayEi={setDelayEi}
    delayIe={delayIe}
    setDelayIe={setDelayIe}
    delayIi={delayIi}
    setDelayIi={setDelayIi}
    VInit={VInit}
    setVInit={setVInit}
    VTh={VTh}
    setVTh={setVTh}
    VReset={VReset}
    setVReset={setVReset}
    EL={EL}
    setEL={setEL}
    Ee={Ee}
    setEe={setEe}
    Ei={Ei}
    setEi={setEi}
    CmE={CmE}
    setCmE={setCmE}
    gLE={gLE}
    setGLE={setGLE}
    CmI={CmI}
    setCmI={setCmI}
    gLI={gLI}
    setGLI={setGLI}
    tRefE={tRefE}
    setTRefE={setTRefE}
    tRefI={tRefI}
    setTRefI={setTRefI}
    tauAmpa={tauAmpa}
    setTauAmpa={setTauAmpa}
    tauGaba={tauGaba}
    setTauGaba={setTauGaba}
    gNa={gNa}
    setGNa={setGNa}
    gK={gK}
    setGK={setGK}
    ENa={ENa}
    setENa={setENa}
    EK={EK}
    setEK={setEK}
    adexVT={adexVT}
    setAdexVT={setAdexVT}
    adexDeltaT={adexDeltaT}
    setAdexDeltaT={setAdexDeltaT}
    adexTauW={adexTauW}
    setAdexTauW={setAdexTauW}
    adexA={adexA}
    setAdexA={setAdexA}
    adexB={adexB}
    setAdexB={setAdexB}
    adexVPeak={adexVPeak}
    setAdexVPeak={setAdexVPeak}
    gA={gA}
    setGA={setGA}
    fhnA={fhnA}
    setFhnA={setFhnA}
    fhnB={fhnB}
    setFhnB={setFhnB}
    fhnTauW={fhnTauW}
    setFhnTauW={setFhnTauW}
    qifA={qifA}
    setQifA={setQifA}
    qifVr={qifVr}
    setQifVr={setQifVr}
    qifVt={qifVt}
    setQifVt={setQifVt}
    izhA={izhA}
    setIzhA={setIzhA}
    izhB={izhB}
    setIzhB={setIzhB}
    izhC={izhC}
    setIzhC={setIzhC}
    izhD={izhD}
    setIzhD={setIzhD}
    mqifA={mqifA}
    setMqifA={setMqifA}
    mqifVr={mqifVr}
    setMqifVr={setMqifVr}
    mqifWA={mqifWA}
    setMqifWA={setMqifWA}
    mqifWVr={mqifWVr}
    setMqifWVr={setMqifWVr}
    mqifWTau={mqifWTau}
    setMqifWTau={setMqifWTau}
    gLHet={gLHet}
    setGLHet={setGLHet}
    cMHet={cMHet}
    setCMHet={setCMHet}
    vThHet={vThHet}
    setVThHet={setVThHet}
    tRefHet={tRefHet}
    setTRefHet={setTRefHet}
  />

  return (
    <div
      className="overflow-hidden p-2"
      style={{ height: "100vh", width: "100vw" }}
    >
      <div
        className="flex min-h-0 gap-3 overflow-hidden rounded-2xl bg-slate-100 p-3 dark:bg-[#0f1015]"
        style={{ height: "100%", width: "100%" }}
      >
        <main
          className="flex min-h-0 min-w-0 flex-1 overflow-hidden rounded-xl border border-slate-200/70 bg-white/70 dark:border-white/10 dark:bg-white/5"
          style={{ height: "100%" }}
        >
          <div className="grid h-full w-full min-h-0 grid-rows-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1fr)] gap-2 p-2">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-zinc-400">
              <div className="flex flex-wrap items-center gap-2 text-sm font-mono text-slate-700 dark:text-zinc-200">
                <span className="rounded-xl bg-white/70 px-4 py-2 text-sm dark:bg-white/5">
                  {loading
                    ? "Running..."
                    : data
                    ? `${spikeCounts.eCount} E / ${spikeCounts.iCount} I`
                    : "No data"}
                </span>
                <span className="rounded-xl bg-white/70 px-4 py-2 text-sm dark:bg-white/5">
                  {data
                    ? `Rhythmicity ${data.rhythmicity.toFixed(3)}`
                    : "Rhythmicity --"}
                </span>
                <span className="rounded-xl bg-white/70 px-4 py-2 text-sm dark:bg-white/5">
                  {data
                    ? `Rates ${data.mean_rate_E.toFixed(2)} Hz E / ${data.mean_rate_I.toFixed(2)} Hz I`
                    : "Rates --"}
                </span>
              </div>
              <div className="flex items-center gap-3 rounded-xl bg-white/70 px-4 py-2.5 text-sm font-semibold uppercase tracking-wide text-slate-600 dark:bg-white/5 dark:text-zinc-300">
                <span className="inline-flex items-center gap-1">
                  <span className="h-3 w-3 rounded-sm bg-current" />
                  E
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: "#d9480f" }} />
                  I
                </span>
              </div>
            </div>
            <div className="min-h-0">
              <div className="flex h-full w-full flex-col gap-2">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-zinc-400">
                  Raster plot
                </div>
                <div ref={plotRef} className="min-h-0 flex-1">
                  {error ? (
                    <div className="flex h-full items-center justify-center rounded-lg border border-red-500/50 bg-red-500/10 px-4 py-6 text-sm text-red-600 dark:border-red-400/60 dark:text-red-300">
                      Cannot connect to simulation api, is the server running?
                    </div>
                  ) : (
                    <RasterPlot
                      width={plotSize.width}
                      height={plotSize.height}
                      margin={margin}
                      innerWidth={innerWidth}
                      innerHeight={innerHeight}
                      xScale={xScale}
                      yScale={yScale}
                      spikes={data?.spikes ?? null}
                    />
                  )}
                </div>
              </div>
            </div>
            <div className="flex min-h-0 w-full flex-col gap-2 rounded-lg bg-white/70 p-3 dark:bg-white/5">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-zinc-400">
                Population rate (Hz)
              </div>
              <div ref={ratePlotRef} className="min-h-0 flex-1">
                {data && data.population_rate_hz_E.length > 0 ? (
                  <PopulationRatePlot
                    width={ratePlotSize.width}
                    height={ratePlotSize.height}
                    margin={rateMargin}
                    innerWidth={rateInnerWidth}
                    innerHeight={rateInnerHeight}
                    tMs={data.population_rate_t_ms}
                    rateHzE={data.population_rate_hz_E}
                    rateHzI={data.population_rate_hz_I}
                    maxTMs={T}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-slate-500 dark:text-zinc-400">
                    No population rate data yet.
                  </div>
                )}
              </div>
            </div>
            <div className="flex min-h-0 w-full flex-col gap-2">
              <div className="flex min-h-0 flex-1 w-full flex-col gap-2 rounded-lg bg-white/70 p-3 dark:bg-white/5">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-zinc-400">
                  Membrane potential E (mV)
                </div>
                <div ref={membraneEPlotRef} className="min-h-0 flex-1">
                  {data && data.membrane_t_ms.length > 0 ? (
                    <MembranePotentialPlot
                      width={membraneEPlotSize.width}
                      height={membraneEPlotSize.height}
                      margin={rateMargin}
                      innerWidth={membraneEInnerWidth}
                      innerHeight={membraneEInnerHeight}
                      tMs={data.membrane_t_ms}
                      vE={data.membrane_V_E}
                      maxTMs={T}
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-xs text-slate-500 dark:text-zinc-400">
                      No membrane E data yet.
                    </div>
                  )}
                </div>
              </div>
              <div className="flex min-h-0 flex-1 w-full flex-col gap-2 rounded-lg bg-white/70 p-3 dark:bg-white/5">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-zinc-400">
                  Membrane potential I (mV)
                </div>
                <div ref={membraneIPlotRef} className="min-h-0 flex-1">
                  {data && data.membrane_t_ms.length > 0 ? (
                    <MembranePotentialPlot
                      width={membraneIPlotSize.width}
                      height={membraneIPlotSize.height}
                      margin={rateMargin}
                      innerWidth={membraneIInnerWidth}
                      innerHeight={membraneIInnerHeight}
                      tMs={data.membrane_t_ms}
                      vI={data.membrane_V_I}
                      maxTMs={T}
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-xs text-slate-500 dark:text-zinc-400">
                      No membrane I data yet.
                    </div>
                  )}
                </div>
              </div>
              <div className="flex min-h-0 flex-1 w-full flex-col gap-2 rounded-lg bg-white/70 p-3 dark:bg-white/5">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-zinc-400">
                  Input current
                </div>
                <div ref={inputPlotRef} className="min-h-0 flex-1">
                  {data && data.input_t_ms.length > 0 ? (
                    <InputTracePlot
                      width={inputPlotSize.width}
                      height={inputPlotSize.height}
                      margin={rateMargin}
                      innerWidth={inputInnerWidth}
                      innerHeight={inputInnerHeight}
                      tMs={data.input_t_ms}
                      inputE={data.input_mean_E}
                      inputI={data.input_mean_I}
                      maxTMs={T}
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-xs text-slate-500 dark:text-zinc-400">
                      No input data yet.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
        <aside
          className="flex min-h-0 w-[320px] shrink-0 flex-col overflow-hidden rounded-xl bg-white/70 dark:border-white/10 dark:bg-white/5"
          style={{ height: "100%" }}
        >
          <div
            className="min-h-0 flex-1 text-sm text-slate-600 dark:text-zinc-300"
            style={{ overflowY: "auto" }}
          >
            {Panel}
          </div>
        </aside>
      </div>
    </div>
  );
}
