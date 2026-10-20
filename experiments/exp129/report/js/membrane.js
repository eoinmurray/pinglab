export const DT = 0.01;
export const REVERSALS = { na: 50, k: -77, leak: -54.3 };

// Classical squid-axon kinetics at 6.3 °C, using inside-minus-outside voltage.
function vtrap(x, scale) {
  return Math.abs(x / scale) < 1e-6 ? scale * (1 - x / scale / 2) : x / Math.expm1(x / scale);
}
export function gateRates(v) {
  return [
    [0.1 * vtrap(-(v + 40), 10), 4 * Math.exp(-(v + 65) / 18)],
    [0.07 * Math.exp(-(v + 65) / 20), 1 / (Math.exp(-(v + 35) / 10) + 1)],
    [0.01 * vtrap(-(v + 55), 10), 0.125 * Math.exp(-(v + 65) / 80)],
  ];
}
export function initialState() {
  return [-65, ...gateRates(-65).map(([a, b]) => a / (a + b))];
}
export function ionicCurrents(state, params) {
  const [v, m, h, n] = state;
  const gNa = 120 * params.sodium * m ** 3 * h;
  const gK = 36 * params.potassium * n ** 4;
  return { gNa, gK, na: gNa * (v - REVERSALS.na), k: gK * (v - REVERSALS.k),
    leak: 0.3 * (v - REVERSALS.leak) };
}
export const SYNAPTIC_DECAY = 2, SPIKE_DELAY = 4, SPIKE_INTERVAL = 40;
export const synapticCurrent = (conductance, voltage) => conductance * (0 - voltage);
function derivative(state, params, conductance) {
  const currents = ionicCurrents(state, params);
  return [(synapticCurrent(conductance, state[0]) - currents.na - currents.k - currents.leak) / params.capacitance,
    ...gateRates(state[0]).map(([a, b], index) => a * (1 - state[index + 1]) - b * state[index + 1])];
}
export function advanceNeuron(state, params, conductance, dt = DT) {
  const add = (values, change, scale) => values.map((v, index) => v + scale * change[index]);
  const midpoint = conductance * Math.exp(-dt / (2 * SYNAPTIC_DECAY));
  const endpoint = conductance * Math.exp(-dt / SYNAPTIC_DECAY);
  const a = derivative(state, params, conductance);
  const b = derivative(add(state, a, dt / 2), params, midpoint);
  const c = derivative(add(state, b, dt / 2), params, midpoint);
  const d = derivative(add(state, c, dt), params, endpoint);
  return state.map((v, index) => v + dt / 6 * (a[index] + 2 * b[index] + 2 * c[index] + d[index]));
}

export async function mountMembrane(host) {
  const { Application, Graphics } = await import('pixi.js');
  const surface = host.querySelector('.scene'), app = new Application();
  await app.init({ width: 1240, height: 360, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  surface.appendChild(app.canvas); surface.querySelector('.status')?.remove();
  const graphics = new Graphics(); app.stage.addChild(graphics);
  const labels = document.createElement('div'); labels.className = 'membrane-math-labels';
  labels.setAttribute('aria-hidden', 'true'); surface.appendChild(labels);
  const events = new AbortController(), get = name => host.querySelector(`[data-${name}]`);
  const blue = 0x176d91, rose = 0xbc3866, amber = 0xa87922, ink = 0x263343, grey = 0xdce3e8;
  let state = initialState(), time = 0, carry = 0, params, history = [], particles = [], flow = { na: 0, k: 0, syn: 0 }, synapse = 0, nextArrival = 5, arrivals = 0;
  let width, membrane, plots, voltageLabel, pulseLabel, leftTimes, rightTimes, sample = 0;
  let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches, visible = true, disposed = false;
  function label(markup, x, y, size = 18, color = ink, centered = false) {
    const node = document.createElement('span'); node.innerHTML = markup;
    Object.assign(node.style, { position: 'absolute', left: `${100 * x / width}%`, top: `${100 * y / 360}%`,
      fontSize: `${size * surface.clientWidth / width}px`, lineHeight: '1.2', whiteSpace: 'nowrap',
      color: `#${color.toString(16).padStart(6, '0')}`, transform: centered ? 'translateX(-50%)' : '' });
    labels.appendChild(node); return node;
  }
  function line(x1, y1, x2, y2, color = grey, thickness = 1.5, alpha = 1) {
    graphics.moveTo(x1, y1).lineTo(x2, y2).stroke({ color, width: thickness, alpha });
  }
  function layout() {
    labels.replaceChildren(); leftTimes = []; rightTimes = [];
    width = matchMedia('(max-width: 600px)').matches ? 620 : 1240;
    app.renderer.resize(width, 360); app.canvas.style.width = '100%'; app.canvas.style.height = '100%';
    const panel = width / 3, scene = 2 * panel;
    const thickness = 28 / params.capacitance;
    membrane = { left: 22, right: scene - 24, top: 180 - thickness / 2, bottom: 180 + thickness / 2,
      source: { x: scene * 0.22, y: 46 },
      channels: [{ name: 'syn', x: scene * 0.22, color: amber }, { name: 'na', x: scene * 0.50, color: blue }, { name: 'k', x: scene * 0.78, color: rose }] };
    label('A', 16, 14);
    label('Outside the neuron', scene * 0.65, 14, 18, ink, true);
    label('Inside the neuron', scene / 2, 309, 18, ink, true);
    label('Na⁺', 28, 334, 17, blue); label('K⁺', 88, 334, 17, rose); label('Synapse', 148, 334, 17, amber);
    label('Presynaptic neuron', membrane.source.x + 24, 37, 16, rose);
    for (const channel of membrane.channels) {
      label(channel.name === 'syn' ? (width === 620 ? 'Synapse' : 'Synaptic channel') : channel.name === 'na' ? 'Na⁺ channel' : 'K⁺ channel', channel.x, 130, 16, channel.color, true);
    }
    voltageLabel = label('', membrane.left + 10, 309, 17);
    pulseLabel = label('', scene * 0.78, 334, 16, ink, true);
    plots = [
      { x: scene + 54, y: 48, w: panel - 74, h: 100, min: -85, max: 50, ticks: [-80, -20, 40], field: 'v', color: ink },
      { x: scene + 54, y: 220, w: panel - 74, h: 100, min: -1000, max: 1000, ticks: [-1000, 0, 1000], field: 'na', color: blue },
    ];
    label('B  <i>V</i><sub>m</sub> (mV)', plots[0].x - 37, 14);
    label('C  <i>I</i> (μA/cm²)', plots[1].x - 37, 184);
    for (const plot of plots) {
      for (const value of plot.ticks) {
        const node = label(String(value).replace('-', '−'), plot.x - 10,
          plot.y + (plot.max - value) / (plot.max - plot.min) * plot.h - 10, 16);
        node.style.transform = 'translateX(-100%)';
      }
      leftTimes.push(label('', plot.x - 4, plot.y + plot.h + 12, 17));
      label('<i>t</i> (ms)', plot.x + plot.w / 2 - 24, plot.y + plot.h + 12, 17);
      const right = label('', plot.x + plot.w, plot.y + plot.h + 12, 17);
      right.style.transform = 'translateX(-100%)'; rightTimes.push(right);
    }
  }
  function ion(x, y, color, alpha = 0.8) { graphics.circle(x, y, 4.5).fill({ color, alpha }); }
  function drawMembrane(currents) {
    const { left, right, top, bottom, channels, source } = membrane;
    const synapticChannel = channels[0];
    line(source.x, source.y + 14, synapticChannel.x, top - 12);
    const phase = (time + 1e-8) % SPIKE_INTERVAL, travelling = phase >= 1 && phase < 5;
    graphics.circle(source.x, source.y, 14).fill({ color: rose, alpha: travelling ? 0.12 : 0.04 }).stroke({ color: rose, width: 1.5 });
    if (travelling) {
      const progress = (phase - 1) / SPIKE_DELAY;
      graphics.circle(source.x + (synapticChannel.x - source.x) * progress, source.y + 14 + (top - 12 - source.y - 14) * progress, 5).fill(rose);
    }
    const arrivalAge = phase - 5;
    if (arrivalAge >= 0 && arrivalAge < 0.8) graphics.circle(synapticChannel.x, top - 8, 9).stroke({ color: amber, width: 2, alpha: 1 - arrivalAge / 0.8 });
    graphics.roundRect(left, top, right - left, bottom - top, 8).fill({ color: grey, alpha: 0.35 });
    // Paired lipid heads mark the insulating membrane; channels are the only traversable paths.
    for (let x = left + 6; x < right - 6; x += 12) {
      if (channels.some(channel => Math.abs(x - channel.x) < 20)) continue;
      graphics.circle(x, top, 3).fill(grey); graphics.circle(x, bottom, 3).fill(grey);
      line(x, top + 4, x, bottom - 4, grey, 1, 0.5);
    }
    const spacing = (right - left - 32) / 15;
    for (let row = 0; row < 2; row++) for (let col = 0; col < 16; col++) {
      const x = left + 16 + col * spacing;
      // These reservoirs illustrate Na-rich outside and K-rich inside, without counting concentration.
      ion(x, 78 + row * 28, (col + row) % 5 === 0 ? rose : blue, 0.45);
      ion(x, 250 + row * 28, (col + row) % 5 === 0 ? blue : rose, 0.45);
    }
    for (const channel of channels) {
      const conductance = channel.name === 'syn' ? synapse : channel.name === 'na' ? currents.gNa : currents.gK;
      const opening = Math.min(1, conductance / (channel.name === 'syn' ? 0.2 : 25));
      const aperture = 5 + 20 * opening;
      graphics.rect(channel.x - 21, top - 5, 42, bottom - top + 10).fill(0xffffff);
      graphics.roundRect(channel.x - aperture / 2 - 6, top - 6, 6, bottom - top + 12, 3).fill(channel.color);
      graphics.roundRect(channel.x + aperture / 2, top - 6, 6, bottom - top + 12, 3).fill(channel.color);
      line(channel.x - aperture / 2, 180, channel.x + aperture / 2, 180, channel.color, 3, 1 - opening);
    }
    for (const particle of particles) {
      const channel = channels.find(c => c.name === particle.name);
      const progress = Math.min(1, (time - particle.start) / 1);
      const start = particle.inward ? 116 : 238, end = particle.inward ? 238 : 116;
      ion(channel.x, start + (end - start) * progress, channel.color, 1);
    }
    const v = state[0]; voltageLabel.textContent = `${v.toFixed(1)} mV`.replace('-', '−');
    pulseLabel.textContent = travelling ? 'Spike arriving' : arrivalAge >= 0 && arrivalAge < 2 ? (params.weight > 0 ? 'Synapse open' : 'Spike arrived') : '';
  }
  function trace(plot, field, color, axes = true) {
    const cutoff = Math.max(0, time - 60);
    const mapY = value => plot.y + plot.h * (plot.max - value) / (plot.max - plot.min);
    if (axes) {
      line(plot.x, plot.y, plot.x, plot.y + plot.h); line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h);
      for (const tick of plot.ticks) line(plot.x - 5, mapY(tick), plot.x + plot.w, mapY(tick), grey, 1, 0.5);
    }
    if (axes) for (let at = 5 + Math.max(0, Math.ceil((cutoff - 5) / SPIKE_INTERVAL)) * SPIKE_INTERVAL; at <= time; at += SPIKE_INTERVAL) {
      const x = plot.x + (at - cutoff) / 60 * plot.w;
      line(x, plot.y, x, plot.y + plot.h, amber, 1, 0.2);
    }
    let previous;
    for (const point of history) {
      if (point.t < cutoff) continue;
      const x = plot.x + (point.t - cutoff) / 60 * plot.w, y = mapY(point[field]);
      if (previous) {
        const dy = y - previous.y;
        let start = 0, end = 1;
        if (dy === 0) { if (y < plot.y || y > plot.y + plot.h) end = -1; }
        else {
          const a = (plot.y - previous.y) / dy, b = (plot.y + plot.h - previous.y) / dy;
          start = Math.max(0, Math.min(a, b)); end = Math.min(1, Math.max(a, b));
        }
        if (start <= end) graphics.moveTo(previous.x + (x - previous.x) * start, previous.y + dy * start)
          .lineTo(previous.x + (x - previous.x) * end, previous.y + dy * end);
      }
      previous = { x, y };
    }
    graphics.stroke({ color, width: 2 });
  }
  function draw() {
    if (!membrane) return;
    const currents = ionicCurrents(state, params);
    graphics.clear(); drawMembrane(currents); trace(plots[0], 'v', ink);
    trace(plots[1], 'na', blue); trace(plots[1], 'k', rose, false); trace(plots[1], 'syn', amber, false);
    const cutoff = Math.max(0, time - 60);
    leftTimes.forEach(node => { node.textContent = cutoff.toFixed(0); });
    rightTimes.forEach(node => { node.textContent = (cutoff + 60).toFixed(0); });
    Object.assign(host.dataset, { time: String(time), voltage: String(state[0]), sodiumCurrent: String(currents.na),
      potassiumCurrent: String(currents.k), sodiumConductance: String(currents.gNa), potassiumConductance: String(currents.gK),
      playing: String(playing), ionsInFlight: String(particles.length), synapticConductance: String(synapse),
      synapticCurrent: String(synapticCurrent(synapse, state[0])), arrivals: String(arrivals), spikeTravelling: String((time % 40) >= 1 && (time % 40) < 5) });
  }
  function record() {
    const c = ionicCurrents(state, params);
    history.push({ t: time, v: state[0], na: -c.na, k: -c.k, syn: synapticCurrent(synapse, state[0]) });
  }
  function step() {
    if (time + 1e-8 >= nextArrival) {
      synapse += params.weight; nextArrival += SPIKE_INTERVAL; arrivals++;
    }
    state = advanceNeuron(state, params, synapse);
    synapse *= Math.exp(-DT / SYNAPTIC_DECAY); time += DT;
    const currents = ionicCurrents(state, params);
    particles = particles.filter(p => time - p.start < 1);
    for (const name of ['na', 'k', 'syn']) {
      const outward = name === 'syn' ? -synapticCurrent(synapse, state[0]) : currents[name];
      flow[name] += Math.abs(outward) * DT / (name === 'syn' ? 2 : 25);
      while (flow[name] >= 1) {
        particles.push({ name, inward: outward < 0, start: time }); flow[name] -= 1;
      }
    }
    if (++sample % 5 === 0) {
      record(); history = history.filter(p => p.t >= time - 60.1);
    }
  }
  function refresh() { draw(); app.render(); }
  function restart(preview = !playing) {
    params = { capacitance: Number(get('capacitance').value), weight: Number(get('weight').value),
      sodium: Number(get('sodium').value) / 100, potassium: Number(get('potassium').value) / 100 };
    get('capacitance-value').value = `${params.capacitance.toFixed(1)} μF/cm²`;
    get('weight-value').value = `${params.weight.toFixed(2)} mS/cm²`;
    get('sodium-value').value = `${params.sodium * 100}%`; get('potassium-value').value = `${params.potassium * 100}%`;
    state = initialState();
    for (let k = 0; k < 200 / DT; k++) state = advanceNeuron(state, params, 0);
    time = 0; carry = 0; history = []; particles = []; flow = { na: 0, k: 0, syn: 0 }; synapse = 0; nextArrival = 5; arrivals = 0; sample = 0;
    record(); if (preview) for (let k = 0; k < 60 / DT; k++) step();
    layout(); refresh();
  }
  function playback() {
    get('pause').textContent = playing ? 'Pause' : 'Play';
    if (playing && visible && !document.hidden) app.start(); else app.stop();
    refresh();
  }
  for (const name of ['capacitance', 'weight', 'sodium', 'potassium']) get(name).addEventListener('input', () => restart(), { signal: events.signal });
  get('restart').addEventListener('click', () => restart(false), { signal: events.signal });
  get('pause').addEventListener('click', () => { playing = !playing; carry = 0; playback(); }, { signal: events.signal });
  document.addEventListener('visibilitychange', () => { carry = 0; playback(); }, { signal: events.signal });
  const observer = new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); carry = 0; playback(); }); observer.observe(host);
  const resize = new ResizeObserver(() => { if (!disposed && params) { layout(); refresh(); } }); resize.observe(surface);
  app.ticker.add(ticker => {
    carry += Math.min(ticker.elapsedMS, 100) * 0.005;
    while (carry >= DT) { step(); carry -= DT; }
    draw();
  });
  get('pause').disabled = false; get('restart').disabled = false; restart(); playback();
  return () => { disposed = true; events.abort(); observer.disconnect(); resize.disconnect(); labels.remove(); app.destroy(true, { children: true }); };
}
