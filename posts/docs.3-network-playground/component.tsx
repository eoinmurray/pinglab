import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear } from "@visx/scale";
import ParameterPanel, { type NeuronModel } from "./components/ParameterPanel";
import CorrelationPlot from "./components/CorrelationPlot";
import MembranePotentialPlot from "./components/MembranePotentialPlot";
import PopulationRatePlot from "./components/PopulationRatePlot";
import RasterPlot from "./components/RasterPlot";
import WeightsHistogramPlot from "./components/WeightsHistogramPlot";

type WeightDistName = "normal" | "lognormal" | "gamma" | "exponential";

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
  isi_cv_E: number;
  input_t_ms: number[];
  input_mean_E: number[];
  input_mean_I: number[];
  weights_hist_bins: number[];
  weights_hist_counts_ee: number[];
  weights_hist_counts_ei: number[];
  weights_hist_counts_ie: number[];
  weights_hist_counts_ii: number[];
  autocorr_lags_ms: number[];
  autocorr_corr: number[];
  xcorr_lags_ms: number[];
  xcorr_corr: number[];
};

const API_URL = "http://localhost:8000/run";
const CONFIG_API = "http://localhost:8000/configs";

const defaultWidth = 760;
const defaultHeight = 420;
const margin = { top: 24, right: 20, bottom: 44, left: 24 };
const rateMargin = { top: 8, right: 16, bottom: 28, left: 42 };
const histMargin = { top: 10, right: 10, bottom: 20, left: 28 };

export default function Component() {
  const [data, setData] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [configList, setConfigList] = useState<string[]>([]);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [saveName, setSaveName] = useState("");
  const [configStatus, setConfigStatus] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"single" | "scans">("single");

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

  const [eeDist, setEeDist] = useState<WeightDistName>("normal");
  const [eiDist, setEiDist] = useState<WeightDistName>("normal");
  const [ieDist, setIeDist] = useState<WeightDistName>("normal");
  const [iiDist, setIiDist] = useState<WeightDistName>("normal");
  const [eeMean, setEeMean] = useState(0.02);
  const [eiMean, setEiMean] = useState(0.015);
  const [ieMean, setIeMean] = useState(0.015);
  const [iiMean, setIiMean] = useState(0.02);
  const [eeStd, setEeStd] = useState(0.0);
  const [eiStd, setEiStd] = useState(0.0);
  const [ieStd, setIeStd] = useState(0.0);
  const [iiStd, setIiStd] = useState(0.0);
  const [eeSigma, setEeSigma] = useState(1.0);
  const [eiSigma, setEiSigma] = useState(1.0);
  const [ieSigma, setIeSigma] = useState(1.0);
  const [iiSigma, setIiSigma] = useState(1.0);
  const [eeShape, setEeShape] = useState(1.0);
  const [eiShape, setEiShape] = useState(1.0);
  const [ieShape, setIeShape] = useState(1.0);
  const [iiShape, setIiShape] = useState(1.0);
  const [eeScale, setEeScale] = useState(0.01);
  const [eiScale, setEiScale] = useState(0.01);
  const [ieScale, setIeScale] = useState(0.01);
  const [iiScale, setIiScale] = useState(0.01);
  const [pEe, setPEe] = useState(0.02);
  const [pEi, setPEi] = useState(0.18);
  const [pIe, setPIe] = useState(0.04);
  const [pIi, setPIi] = useState(0.06);
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
  const [membraneEPlotSize, setMembraneEPlotSize] = useState({
    width: defaultWidth,
    height: 192,
  });
  const membraneIPlotRef = useRef<HTMLDivElement | null>(null);
  const [membraneIPlotSize, setMembraneIPlotSize] = useState({
    width: defaultWidth,
    height: 192,
  });
  const autocorrRef = useRef<HTMLDivElement | null>(null);
  const xcorrRef = useRef<HTMLDivElement | null>(null);
  const [autocorrSize, setAutocorrSize] = useState({ width: defaultWidth, height: 192 });
  const [xcorrSize, setXcorrSize] = useState({ width: defaultWidth, height: 192 });
  const histEeRef = useRef<HTMLDivElement | null>(null);
  const histEiRef = useRef<HTMLDivElement | null>(null);
  const histIeRef = useRef<HTMLDivElement | null>(null);
  const histIiRef = useRef<HTMLDivElement | null>(null);
  const [histEeSize, setHistEeSize] = useState({ width: 200, height: 120 });
  const [histEiSize, setHistEiSize] = useState({ width: 200, height: 120 });
  const [histIeSize, setHistIeSize] = useState({ width: 200, height: 120 });
  const [histIiSize, setHistIiSize] = useState({ width: 200, height: 120 });
  const totalN = nE + nI;

  const buildWeightParams = (
    dist: WeightDistName,
    mean: number,
    std: number,
    sigma: number,
    shape: number,
    scale: number
  ) => {
    switch (dist) {
      case "normal":
        return { mean, std };
      case "lognormal":
        return { mean, sigma };
      case "gamma":
        return { shape, scale };
      case "exponential":
        return { scale };
      default:
        return { mean, std };
    }
  };

  const requestPayload = useMemo(
    () => ({
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
        ee: { p: pEe, dist: { name: eeDist, params: buildWeightParams(eeDist, eeMean, eeStd, eeSigma, eeShape, eeScale) } },
        ei: { p: pEi, dist: { name: eiDist, params: buildWeightParams(eiDist, eiMean, eiStd, eiSigma, eiShape, eiScale) } },
        ie: { p: pIe, dist: { name: ieDist, params: buildWeightParams(ieDist, ieMean, ieStd, ieSigma, ieShape, ieScale) } },
        ii: { p: pIi, dist: { name: iiDist, params: buildWeightParams(iiDist, iiMean, iiStd, iiSigma, iiShape, iiScale) } },
        clamp_min: clampMin,
        seed: weightsSeed,
      },
    }),
    [
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
      inputType,
      iEStart,
      iEEnd,
      iIStart,
      iIEnd,
      iEBase,
      iIBase,
      noiseStd,
      inputSeed,
      inputPulseT,
      inputPulseWidth,
      inputPulseInterval,
      inputPulseAmpE,
      inputPulseAmpI,
      eeDist,
      eiDist,
      ieDist,
      iiDist,
      eeMean,
      eiMean,
      ieMean,
      iiMean,
      eeStd,
      eiStd,
      ieStd,
      iiStd,
      eeSigma,
      eiSigma,
      ieSigma,
      iiSigma,
      eeShape,
      eiShape,
      ieShape,
      iiShape,
      eeScale,
      eiScale,
      ieScale,
      iiScale,
      pEe,
      pEi,
      pIe,
      pIi,
      clampMin,
      weightsSeed,
    ]
  );

  const applyConfig = (payload: any) => {
    if (!payload) {
      return;
    }
    const cfg = payload.config ?? payload.base ?? {};
    const inputs = payload.inputs ?? payload.default_inputs ?? {};
    const weights = payload.weights ?? {};

    if (cfg.dt !== undefined) setDt(cfg.dt);
    if (cfg.T !== undefined) setT(cfg.T);
    if (cfg.N_E !== undefined) setNE(cfg.N_E);
    if (cfg.N_I !== undefined) setNI(cfg.N_I);
    if (cfg.seed !== undefined) setSeed(cfg.seed);
    if (cfg.neuron_model !== undefined) setNeuronModel(cfg.neuron_model as NeuronModel);
    if (cfg.delay_ee !== undefined) setDelayEe(cfg.delay_ee);
    if (cfg.delay_ei !== undefined) setDelayEi(cfg.delay_ei);
    if (cfg.delay_ie !== undefined) setDelayIe(cfg.delay_ie);
    if (cfg.delay_ii !== undefined) setDelayIi(cfg.delay_ii);
    if (cfg.V_init !== undefined) setVInit(cfg.V_init);
    if (cfg.E_L !== undefined) setEL(cfg.E_L);
    if (cfg.E_e !== undefined) setEe(cfg.E_e);
    if (cfg.E_i !== undefined) setEi(cfg.E_i);
    if (cfg.C_m_E !== undefined) setCmE(cfg.C_m_E);
    if (cfg.g_L_E !== undefined) setGLE(cfg.g_L_E);
    if (cfg.C_m_I !== undefined) setCmI(cfg.C_m_I);
    if (cfg.g_L_I !== undefined) setGLI(cfg.g_L_I);
    if (cfg.V_th !== undefined) setVTh(cfg.V_th);
    if (cfg.V_reset !== undefined) setVReset(cfg.V_reset);
    if (cfg.t_ref_E !== undefined) setTRefE(cfg.t_ref_E);
    if (cfg.t_ref_I !== undefined) setTRefI(cfg.t_ref_I);
    if (cfg.tau_ampa !== undefined) setTauAmpa(cfg.tau_ampa);
    if (cfg.tau_gaba !== undefined) setTauGaba(cfg.tau_gaba);
    if (cfg.g_Na !== undefined) setGNa(cfg.g_Na);
    if (cfg.g_K !== undefined) setGK(cfg.g_K);
    if (cfg.E_Na !== undefined) setENa(cfg.E_Na);
    if (cfg.E_K !== undefined) setEK(cfg.E_K);
    if (cfg.adex_V_T !== undefined) setAdexVT(cfg.adex_V_T);
    if (cfg.adex_delta_T !== undefined) setAdexDeltaT(cfg.adex_delta_T);
    if (cfg.adex_tau_w !== undefined) setAdexTauW(cfg.adex_tau_w);
    if (cfg.adex_a !== undefined) setAdexA(cfg.adex_a);
    if (cfg.adex_b !== undefined) setAdexB(cfg.adex_b);
    if (cfg.adex_V_peak !== undefined) setAdexVPeak(cfg.adex_V_peak);
    if (cfg.g_A !== undefined) setGA(cfg.g_A);
    if (cfg.fhn_a !== undefined) setFhnA(cfg.fhn_a);
    if (cfg.fhn_b !== undefined) setFhnB(cfg.fhn_b);
    if (cfg.fhn_tau_w !== undefined) setFhnTauW(cfg.fhn_tau_w);
    if (cfg.mqif_a?.[0] !== undefined) setMqifA(cfg.mqif_a[0]);
    if (cfg.mqif_Vr?.[0] !== undefined) setMqifVr(cfg.mqif_Vr[0]);
    if (cfg.mqif_w_a?.[0] !== undefined) setMqifWA(cfg.mqif_w_a[0]);
    if (cfg.mqif_w_Vr?.[0] !== undefined) setMqifWVr(cfg.mqif_w_Vr[0]);
    if (cfg.mqif_w_tau?.[0] !== undefined) setMqifWTau(cfg.mqif_w_tau[0]);
    if (cfg.qif_a !== undefined) setQifA(cfg.qif_a);
    if (cfg.qif_Vr !== undefined) setQifVr(cfg.qif_Vr);
    if (cfg.qif_Vt !== undefined) setQifVt(cfg.qif_Vt);
    if (cfg.izh_a !== undefined) setIzhA(cfg.izh_a);
    if (cfg.izh_b !== undefined) setIzhB(cfg.izh_b);
    if (cfg.izh_c !== undefined) setIzhC(cfg.izh_c);
    if (cfg.izh_d !== undefined) setIzhD(cfg.izh_d);
    if (cfg.g_L_heterogeneity_sd !== undefined) setGLHet(cfg.g_L_heterogeneity_sd);
    if (cfg.C_m_heterogeneity_sd !== undefined) setCMHet(cfg.C_m_heterogeneity_sd);
    if (cfg.V_th_heterogeneity_sd !== undefined) setVThHet(cfg.V_th_heterogeneity_sd);
    if (cfg.t_ref_heterogeneity_sd !== undefined) setTRefHet(cfg.t_ref_heterogeneity_sd);

    if (inputs.input_type !== undefined) setInputType(inputs.input_type);
    if (inputs.I_E_start !== undefined) setIEStart(inputs.I_E_start);
    if (inputs.I_E_end !== undefined) setIEEnd(inputs.I_E_end);
    if (inputs.I_I_start !== undefined) setIIStart(inputs.I_I_start);
    if (inputs.I_I_end !== undefined) setIIEnd(inputs.I_I_end);
    if (inputs.I_E_base !== undefined) setIEBase(inputs.I_E_base);
    if (inputs.I_I_base !== undefined) setIIBase(inputs.I_I_base);
    if (inputs.noise_std !== undefined) setNoiseStd(inputs.noise_std);
    if (inputs.I_E !== undefined) {
      setInputType("ramp");
      setIEStart(inputs.I_E);
      setIEEnd(inputs.I_E);
      setIEBase(inputs.I_E);
    }
    if (inputs.I_I !== undefined) {
      setInputType("ramp");
      setIIStart(inputs.I_I);
      setIIEnd(inputs.I_I);
      setIIBase(inputs.I_I);
    }
    if (inputs.noise !== undefined) setNoiseStd(inputs.noise);
    if (inputs.seed !== undefined) setInputSeed(inputs.seed);
    if (inputs.pulse_t_ms !== undefined) setInputPulseT(inputs.pulse_t_ms);
    if (inputs.pulse_width_ms !== undefined) setInputPulseWidth(inputs.pulse_width_ms);
    if (inputs.pulse_interval_ms !== undefined) setInputPulseInterval(inputs.pulse_interval_ms);
    if (inputs.pulse_amp_E !== undefined) setInputPulseAmpE(inputs.pulse_amp_E);
    if (inputs.pulse_amp_I !== undefined) setInputPulseAmpI(inputs.pulse_amp_I);

    const ee = weights.ee ?? {};
    const ei = weights.ei ?? {};
    const ie = weights.ie ?? {};
    const ii = weights.ii ?? {};
    if (ee.dist?.name !== undefined) setEeDist(ee.dist.name);
    if (ei.dist?.name !== undefined) setEiDist(ei.dist.name);
    if (ie.dist?.name !== undefined) setIeDist(ie.dist.name);
    if (ii.dist?.name !== undefined) setIiDist(ii.dist.name);

    if (ee.dist?.params?.mean !== undefined) setEeMean(ee.dist.params.mean);
    if (ei.dist?.params?.mean !== undefined) setEiMean(ei.dist.params.mean);
    if (ie.dist?.params?.mean !== undefined) setIeMean(ie.dist.params.mean);
    if (ii.dist?.params?.mean !== undefined) setIiMean(ii.dist.params.mean);

    if (ee.dist?.params?.std !== undefined) setEeStd(ee.dist.params.std);
    if (ei.dist?.params?.std !== undefined) setEiStd(ei.dist.params.std);
    if (ie.dist?.params?.std !== undefined) setIeStd(ie.dist.params.std);
    if (ii.dist?.params?.std !== undefined) setIiStd(ii.dist.params.std);

    if (ee.dist?.params?.sigma !== undefined) setEeSigma(ee.dist.params.sigma);
    if (ei.dist?.params?.sigma !== undefined) setEiSigma(ei.dist.params.sigma);
    if (ie.dist?.params?.sigma !== undefined) setIeSigma(ie.dist.params.sigma);
    if (ii.dist?.params?.sigma !== undefined) setIiSigma(ii.dist.params.sigma);

    if (ee.dist?.params?.shape !== undefined) setEeShape(ee.dist.params.shape);
    if (ei.dist?.params?.shape !== undefined) setEiShape(ei.dist.params.shape);
    if (ie.dist?.params?.shape !== undefined) setIeShape(ie.dist.params.shape);
    if (ii.dist?.params?.shape !== undefined) setIiShape(ii.dist.params.shape);

    if (ee.dist?.params?.scale !== undefined) setEeScale(ee.dist.params.scale);
    if (ei.dist?.params?.scale !== undefined) setEiScale(ei.dist.params.scale);
    if (ie.dist?.params?.scale !== undefined) setIeScale(ie.dist.params.scale);
    if (ii.dist?.params?.scale !== undefined) setIiScale(ii.dist.params.scale);

    if (ee.p !== undefined) setPEe(ee.p);
    if (ei.p !== undefined) setPEi(ei.p);
    if (ie.p !== undefined) setPIe(ie.p);
    if (ii.p !== undefined) setPIi(ii.p);
    if (weights.clamp_min !== undefined) setClampMin(weights.clamp_min);
    if (weights.seed !== undefined) setWeightsSeed(weights.seed);
  };

  const refreshConfigs = async () => {
    try {
      const response = await fetch(CONFIG_API);
      if (!response.ok) {
        throw new Error(`Config list failed (${response.status})`);
      }
      const json = (await response.json()) as { configs?: string[] };
      setConfigList(json.configs ?? []);
    } catch (err) {
      setConfigStatus(err instanceof Error ? err.message : "Failed to load configs");
    }
  };

  useEffect(() => {
    refreshConfigs();
  }, []);

  useEffect(() => {
    if (!configList.length || selectedConfig) {
      return;
    }
    const first = configList[0];
    setSelectedConfig(first);
    setSaveName(first.replace(/\\.yaml$/i, ""));
    handleLoadConfig(first);
  }, [configList, selectedConfig]);

  const handleSaveConfig = async () => {
    if (!saveName.trim()) {
      setConfigStatus("Enter a config name");
      return;
    }
    try {
      const response = await fetch(`${CONFIG_API}/${encodeURIComponent(saveName.trim())}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      if (!response.ok) {
        throw new Error(`Save failed (${response.status})`);
      }
      setConfigStatus("Saved");
      await refreshConfigs();
      setSelectedConfig(saveName.trim().endsWith(".yaml") ? saveName.trim() : `${saveName.trim()}.yaml`);
    } catch (err) {
      setConfigStatus(err instanceof Error ? err.message : "Save failed");
    }
  };

  const handleLoadConfig = async (nameOverride?: string) => {
    const name = nameOverride ?? selectedConfig;
    if (!name) {
      return;
    }
    try {
      const response = await fetch(`${CONFIG_API}/${encodeURIComponent(name)}`);
      if (!response.ok) {
        throw new Error(`Load failed (${response.status})`);
      }
      const json = (await response.json()) as { config?: any };
      applyConfig(json.config);
      setConfigStatus(`Loaded ${name}`);
    } catch (err) {
      setConfigStatus(err instanceof Error ? err.message : "Load failed");
    }
  };

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
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
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
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
        width: Math.max(240, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
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
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
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
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
    if (!autocorrRef.current) {
      return;
    }
    const target = autocorrRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setAutocorrSize({
        width: Math.max(240, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
    if (!xcorrRef.current) {
      return;
    }
    const target = xcorrRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setXcorrSize({
        width: Math.max(240, Math.floor(nextWidth)),
        height: Math.max(120, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
    if (!histEeRef.current) {
      return;
    }
    const target = histEeRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setHistEeSize({
        width: Math.max(140, Math.floor(nextWidth)),
        height: Math.max(90, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
    if (!histEiRef.current) {
      return;
    }
    const target = histEiRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setHistEiSize({
        width: Math.max(140, Math.floor(nextWidth)),
        height: Math.max(90, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
    if (!histIeRef.current) {
      return;
    }
    const target = histIeRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setHistIeSize({
        width: Math.max(140, Math.floor(nextWidth)),
        height: Math.max(90, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "single") {
      return;
    }
    if (!histIiRef.current) {
      return;
    }
    const target = histIiRef.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect;
      setHistIiSize({
        width: Math.max(140, Math.floor(nextWidth)),
        height: Math.max(90, Math.floor(nextHeight)),
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [activeTab]);

  useEffect(() => {
    let ignore = false;
    const timer = setTimeout(() => {
      const run = async () => {
        try {
          setLoading(true);
            const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestPayload),
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
  }, [requestPayload]);

  const innerWidth = plotSize.width - margin.left - margin.right;
  const innerHeight = plotSize.height - margin.top - margin.bottom;
  const rateInnerWidth = ratePlotSize.width - rateMargin.left - rateMargin.right;
  const rateInnerHeight = ratePlotSize.height - rateMargin.top - rateMargin.bottom;
  const membraneEInnerWidth = membraneEPlotSize.width - rateMargin.left - rateMargin.right;
  const membraneEInnerHeight = membraneEPlotSize.height - rateMargin.top - rateMargin.bottom;
  const membraneIInnerWidth = membraneIPlotSize.width - rateMargin.left - rateMargin.right;
  const membraneIInnerHeight = membraneIPlotSize.height - rateMargin.top - rateMargin.bottom;
  const autocorrInnerWidth = autocorrSize.width - rateMargin.left - rateMargin.right;
  const autocorrInnerHeight = autocorrSize.height - rateMargin.top - rateMargin.bottom;
  const xcorrInnerWidth = xcorrSize.width - rateMargin.left - rateMargin.right;
  const xcorrInnerHeight = xcorrSize.height - rateMargin.top - rateMargin.bottom;
  const histEeInnerWidth = histEeSize.width - histMargin.left - histMargin.right;
  const histEeInnerHeight = histEeSize.height - histMargin.top - histMargin.bottom;
  const histEiInnerWidth = histEiSize.width - histMargin.left - histMargin.right;
  const histEiInnerHeight = histEiSize.height - histMargin.top - histMargin.bottom;
  const histIeInnerWidth = histIeSize.width - histMargin.left - histMargin.right;
  const histIeInnerHeight = histIeSize.height - histMargin.top - histMargin.bottom;
  const histIiInnerWidth = histIiSize.width - histMargin.left - histMargin.right;
  const histIiInnerHeight = histIiSize.height - histMargin.top - histMargin.bottom;

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
    configList={configList}
    selectedConfig={selectedConfig}
    onSelectConfig={(value) => {
      setSelectedConfig(value);
      if (value) {
        setSaveName(value.replace(/\\.yaml$/i, ""));
      }
      if (value) {
        handleLoadConfig(value);
      }
    }}
    onLoadConfig={() => handleLoadConfig(selectedConfig)}
    saveName={saveName}
    setSaveName={setSaveName}
    onSaveConfig={handleSaveConfig}
    configStatus={configStatus}
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
    eeDist={eeDist}
    setEeDist={setEeDist}
    eiDist={eiDist}
    setEiDist={setEiDist}
    ieDist={ieDist}
    setIeDist={setIeDist}
    iiDist={iiDist}
    setIiDist={setIiDist}
    eeMean={eeMean}
    setEeMean={setEeMean}
    eiMean={eiMean}
    setEiMean={setEiMean}
    ieMean={ieMean}
    setIeMean={setIeMean}
    iiMean={iiMean}
    setIiMean={setIiMean}
    eeStd={eeStd}
    setEeStd={setEeStd}
    eiStd={eiStd}
    setEiStd={setEiStd}
    ieStd={ieStd}
    setIeStd={setIeStd}
    iiStd={iiStd}
    setIiStd={setIiStd}
    eeSigma={eeSigma}
    setEeSigma={setEeSigma}
    eiSigma={eiSigma}
    setEiSigma={setEiSigma}
    ieSigma={ieSigma}
    setIeSigma={setIeSigma}
    iiSigma={iiSigma}
    setIiSigma={setIiSigma}
    eeShape={eeShape}
    setEeShape={setEeShape}
    eiShape={eiShape}
    setEiShape={setEiShape}
    ieShape={ieShape}
    setIeShape={setIeShape}
    iiShape={iiShape}
    setIiShape={setIiShape}
    eeScale={eeScale}
    setEeScale={setEeScale}
    eiScale={eiScale}
    setEiScale={setEiScale}
    ieScale={ieScale}
    setIeScale={setIeScale}
    iiScale={iiScale}
    setIiScale={setIiScale}
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
        <main
          className="flex min-h-0 min-w-0 flex-1 overflow-hidden rounded-xl border border-slate-200/70 bg-white/70 dark:border-white/10 dark:bg-white/5"
          style={{ height: "100%" }}
        >
          <div className="flex h-full w-full min-h-0 flex-col gap-2 p-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="inline-flex items-center gap-1 rounded-full border border-slate-200/70 bg-white/70 p-1 text-xs font-medium text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-zinc-300">
                <button
                  type="button"
                  onClick={() => setActiveTab("single")}
                  className={`rounded-full px-3 py-1.5 transition ${
                    activeTab === "single"
                      ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
                      : "text-slate-600 hover:bg-slate-200/60 dark:text-zinc-300 dark:hover:bg-white/10"
                  }`}
                >
                  Single
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("scans")}
                  className={`rounded-full px-3 py-1.5 transition ${
                    activeTab === "scans"
                      ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
                      : "text-slate-600 hover:bg-slate-200/60 dark:text-zinc-300 dark:hover:bg-white/10"
                  }`}
                >
                  Scans
                </button>
              </div>
              <div className="text-xs text-slate-500 dark:text-zinc-400">
                {activeTab === "single" ? "Live sim" : "Scan workspace"}
              </div>
            </div>
            {error ? (
              <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
                {error}
              </div>
            ) : null}
            {activeTab === "single" ? (
              <div className="flex min-h-0 flex-1 flex-col gap-2">
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
                        ? `Rates ${data.mean_rate_E.toFixed(2)} Hz E / ${data.mean_rate_I.toFixed(2)} Hz I`
                        : "Rates --"}
                    </span>
                    <span className="rounded-xl bg-white/70 px-4 py-2 text-sm dark:bg-white/5">
                      {data
                        ? `ISI CV (E) ${data.isi_cv_E.toFixed(3)}`
                        : "ISI CV (E) --"}
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
                <div className="grid min-h-0 flex-1 grid-cols-3 grid-rows-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)] gap-2">
                  <div className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5">
                  <div ref={plotRef} className="h-full w-full">
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
                  </div>
                </div>
                <div className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5">
                  <div ref={membraneEPlotRef} className="h-full w-full">
                    <MembranePotentialPlot
                      width={membraneEPlotSize.width}
                      height={membraneEPlotSize.height}
                      margin={rateMargin}
                      innerWidth={membraneEInnerWidth}
                      innerHeight={membraneEInnerHeight}
                      tMs={data?.membrane_t_ms ?? []}
                      vE={data?.membrane_V_E ?? []}
                      maxTMs={T}
                    />
                  </div>
                </div>
                <div className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5">
                  <div ref={ratePlotRef} className="h-full w-full">
                    <PopulationRatePlot
                      width={ratePlotSize.width}
                      height={ratePlotSize.height}
                      margin={rateMargin}
                      innerWidth={rateInnerWidth}
                      innerHeight={rateInnerHeight}
                      tMs={data?.population_rate_t_ms ?? []}
                      rateHzE={data?.population_rate_hz_E ?? []}
                      rateHzI={data?.population_rate_hz_I ?? []}
                      maxTMs={T}
                    />
                  </div>
                </div>
                <div className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5">
                  <div ref={membraneIPlotRef} className="h-full w-full">
                    <MembranePotentialPlot
                      width={membraneIPlotSize.width}
                      height={membraneIPlotSize.height}
                      margin={rateMargin}
                      innerWidth={membraneIInnerWidth}
                      innerHeight={membraneIInnerHeight}
                      tMs={data?.membrane_t_ms ?? []}
                      vI={data?.membrane_V_I ?? []}
                      maxTMs={T}
                    />
                  </div>
                </div>
                <div className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5">
                  <div ref={autocorrRef} className="h-full w-full">
                    <CorrelationPlot
                      width={autocorrSize.width}
                      height={autocorrSize.height}
                      margin={rateMargin}
                      innerWidth={autocorrInnerWidth}
                      innerHeight={autocorrInnerHeight}
                      lagsMs={data?.autocorr_lags_ms ?? []}
                      values={data?.autocorr_corr ?? []}
                      xLabel="Lag (ms)"
                      yLabel="Autocorr"
                      color="#0f172a"
                      yMin={-1}
                      yMax={1}
                    />
                  </div>
                </div>
                <div className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5">
                  <div ref={xcorrRef} className="h-full w-full">
                    <CorrelationPlot
                      width={xcorrSize.width}
                      height={xcorrSize.height}
                      margin={rateMargin}
                      innerWidth={xcorrInnerWidth}
                      innerHeight={xcorrInnerHeight}
                      lagsMs={data?.xcorr_lags_ms ?? []}
                      values={data?.xcorr_corr ?? []}
                      xLabel="Lag (ms)"
                      yLabel="Xcorr"
                      color="#d9480f"
                      yMin={-1}
                      yMax={1}
                    />
                  </div>
                </div>
                <div
                  className="min-h-0"
                  style={{
                    gridColumn: "1 / -1",
                    display: "grid",
                    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                    gap: "8px",
                  }}
                >
                  <div
                    ref={histEeRef}
                    className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5"
                    style={{ height: "100%" }}
                  >
                    <WeightsHistogramPlot
                      width={histEeSize.width}
                      height={histEeSize.height}
                      margin={histMargin}
                      innerWidth={histEeInnerWidth}
                      innerHeight={histEeInnerHeight}
                      bins={data?.weights_hist_bins ?? []}
                      counts={data?.weights_hist_counts_ee ?? []}
                      color="#0f172a"
                      label="EE"
                    />
                  </div>
                  <div
                    ref={histEiRef}
                    className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5"
                    style={{ height: "100%" }}
                  >
                    <WeightsHistogramPlot
                      width={histEiSize.width}
                      height={histEiSize.height}
                      margin={histMargin}
                      innerWidth={histEiInnerWidth}
                      innerHeight={histEiInnerHeight}
                      bins={data?.weights_hist_bins ?? []}
                      counts={data?.weights_hist_counts_ei ?? []}
                      color="#2563eb"
                      label="EI"
                    />
                  </div>
                  <div
                    ref={histIeRef}
                    className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5"
                    style={{ height: "100%" }}
                  >
                    <WeightsHistogramPlot
                      width={histIeSize.width}
                      height={histIeSize.height}
                      margin={histMargin}
                      innerWidth={histIeInnerWidth}
                      innerHeight={histIeInnerHeight}
                      bins={data?.weights_hist_bins ?? []}
                      counts={data?.weights_hist_counts_ie ?? []}
                      color="#e11d48"
                      label="IE"
                    />
                  </div>
                  <div
                    ref={histIiRef}
                    className="min-h-0 rounded-xl border border-slate-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5"
                    style={{ height: "100%" }}
                  >
                    <WeightsHistogramPlot
                      width={histIiSize.width}
                      height={histIiSize.height}
                      margin={histMargin}
                      innerWidth={histIiInnerWidth}
                      innerHeight={histIiInnerHeight}
                      bins={data?.weights_hist_bins ?? []}
                      counts={data?.weights_hist_counts_ii ?? []}
                      color="#f97316"
                      label="II"
                    />
                  </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center rounded-xl border border-dashed border-slate-300/70 bg-white/40 text-sm text-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-zinc-400">
                Scan plots will appear here.
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
