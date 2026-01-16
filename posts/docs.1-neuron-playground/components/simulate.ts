import type { InputParams, SimResult } from "./types";

export function buildInput(params: InputParams, times: number[]): number[] {
  const input = new Array<number>(times.length);
  for (let i = 0; i < times.length; i += 1) {
    const t = times[i];
    let I = params.tonic;
    if (params.mode === "sine") {
      I += params.sineAmp * Math.sin(2 * Math.PI * params.sineFreq * (t / 1000));
    } else if (params.mode === "pulse") {
      if (t >= params.pulseStart && t <= params.pulseStart + params.pulseWidth) {
        I += params.pulseAmp;
      }
    } else if (params.mode === "pulses") {
      if (t >= params.pulsesStart) {
        const dt = t - params.pulsesStart;
        const cycle = params.pulsesInterval;
        if (cycle > 0) {
          const phase = dt % cycle;
          if (phase <= params.pulsesWidth) {
            I += params.pulsesAmp;
          }
        }
      }
    }
    if (params.noiseAmp > 0) {
      I += params.noiseAmp * (Math.random() * 2 - 1);
    }
    input[i] = I;
  }
  return input;
}

export function simulateLif(params: {
  dt: number;
  T: number;
  V_init: number;
  E_L: number;
  g_L: number;
  C_m: number;
  input: number[];
  V_th: number;
  V_reset: number;
}): SimResult {
  const steps = Math.max(1, Math.floor(params.T / params.dt));
  const times = new Array<number>(steps);
  const voltages = new Array<number>(steps);
  let V = params.V_init;
  let spikes = 0;

  for (let i = 0; i < steps; i += 1) {
    times[i] = i * params.dt;
    const I_ext = params.input[i] ?? 0;
    const dVdt = (params.g_L * (params.E_L - V) + I_ext) / params.C_m;
    V = V + params.dt * dVdt;
    if (V >= params.V_th) {
      spikes += 1;
      V = params.V_reset;
    }
    voltages[i] = V;
  }

  return { times, voltages, input: params.input, spikes };
}

function safeDivNexp(x: number, y: number): number {
  const ratio = x / y;
  if (Math.abs(ratio) < 1e-6) {
    return y;
  }
  return x / (1 - Math.exp(-ratio));
}

function alphaM(V: number): number {
  return 0.1 * safeDivNexp(V + 40, 10);
}

function betaM(V: number): number {
  return 4 * Math.exp(-(V + 65) / 18);
}

function alphaH(V: number): number {
  return 0.07 * Math.exp(-(V + 65) / 20);
}

function betaH(V: number): number {
  return 1 / (1 + Math.exp(-(V + 35) / 10));
}

function alphaN(V: number): number {
  return 0.01 * safeDivNexp(V + 55, 10);
}

function betaN(V: number): number {
  return 0.125 * Math.exp(-(V + 65) / 80);
}

export function simulateHh(params: {
  dt: number;
  T: number;
  V_init: number;
  input: number[];
  C_m: number;
  g_L: number;
  g_Na: number;
  g_K: number;
  E_L: number;
  E_Na: number;
  E_K: number;
}): SimResult {
  const steps = Math.max(1, Math.floor(params.T / params.dt));
  const times = new Array<number>(steps);
  const voltages = new Array<number>(steps);
  let V = params.V_init;
  let spikes = 0;
  let prevV = V;

  const vClamped = Math.min(60, Math.max(-100, V));
  let m = alphaM(vClamped) / (alphaM(vClamped) + betaM(vClamped));
  let h = alphaH(vClamped) / (alphaH(vClamped) + betaH(vClamped));
  let n = alphaN(vClamped) / (alphaN(vClamped) + betaN(vClamped));

  for (let i = 0; i < steps; i += 1) {
    times[i] = i * params.dt;
    const I_ext = params.input[i] ?? 0;
    const vRates = Math.min(60, Math.max(-100, V));

    const aM = alphaM(vRates);
    const bM = betaM(vRates);
    const aH = alphaH(vRates);
    const bH = betaH(vRates);
    const aN = alphaN(vRates);
    const bN = betaN(vRates);

    m = m + params.dt * (aM * (1 - m) - bM * m);
    h = h + params.dt * (aH * (1 - h) - bH * h);
    n = n + params.dt * (aN * (1 - n) - bN * n);

    m = Math.min(1, Math.max(0, m));
    h = Math.min(1, Math.max(0, h));
    n = Math.min(1, Math.max(0, n));

    const I_Na = params.g_Na * m ** 3 * h * (V - params.E_Na);
    const I_K = params.g_K * n ** 4 * (V - params.E_K);
    const I_L = params.g_L * (V - params.E_L);

    const dVdt = (I_ext - I_Na - I_K - I_L) / params.C_m;
    V = V + params.dt * dVdt;
    voltages[i] = V;

    if (prevV < 0 && V >= 0) {
      spikes += 1;
    }
    prevV = V;
  }

  return { times, voltages, input: params.input, spikes };
}

export function simulateMqif(params: {
  dt: number;
  T: number;
  V_init: number;
  input: number[];
  C_m: number;
  g_L: number;
  E_L: number;
  V_th: number;
  V_reset: number;
  a_terms: number[];
  V_r_terms: number[];
}): SimResult {
  if (params.a_terms.length !== params.V_r_terms.length) {
    throw new Error("mqif a_terms and V_r_terms must have the same length");
  }

  const steps = Math.max(1, Math.floor(params.T / params.dt));
  const times = new Array<number>(steps);
  const voltages = new Array<number>(steps);
  let V = params.V_init;
  let spikes = 0;

  for (let i = 0; i < steps; i += 1) {
    times[i] = i * params.dt;
    const I_ext = params.input[i] ?? 0;
    let quadTerm = 0;
    for (let j = 0; j < params.a_terms.length; j += 1) {
      const a = params.a_terms[j];
      const Vr = params.V_r_terms[j];
      const dv = V - Vr;
      quadTerm += a * dv * dv;
    }
    const dVdt = (params.g_L * (params.E_L - V) + quadTerm + I_ext) / params.C_m;
    const V_new = V + params.dt * dVdt;
    const spiked = V_new >= params.V_th;
    if (spiked) {
      spikes += 1;
      V = params.V_reset;
    } else {
      V = V_new;
    }
    voltages[i] = V;
  }

  return { times, voltages, input: params.input, spikes };
}
