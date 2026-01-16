export type SimResult = {
  times: number[];
  voltages: number[];
  input: number[];
  spikes: number;
};

export type InputMode = "tonic" | "sine" | "pulse" | "pulses";

export type InputParams = {
  mode: InputMode;
  tonic: number;
  noiseAmp: number;
  sineAmp: number;
  sineFreq: number;
  pulseAmp: number;
  pulseStart: number;
  pulseWidth: number;
  pulsesAmp: number;
  pulsesStart: number;
  pulsesWidth: number;
  pulsesInterval: number;
};

export type NeuronType = "lif" | "hh" | "mqif";
