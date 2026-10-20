export const REST = -65, THRESHOLD = -50, DT = 0.25;
export const CAPACITANCE_PF = 200;
export const SOURCE_RATES_HZ = [40, 30, 20];

export function weightedInput(weights, arrivals) {
  return weights.reduce((sum, weight, i) => sum + weight * arrivals[i], 0);
}

export function makeSpikeTrain(rateHz, seed) {
  let state = seed >>> 0;
  function interval() {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    const uniform = ((state >>> 0) + 0.5) / 4294967296;
    return -Math.log1p(-uniform) * 1000 / rateHz;
  }
  let nextEvent = interval();
  return throughTime => {
    const events = [];
    while (nextEvent <= throughTime) {
      events.push(nextEvent);
      nextEvent += interval();
    }
    return events;
  };
}

// Exact subthreshold integration for current held constant during each step.
export function advanceVoltage(voltage, tau, current = 0, dt = DT) {
  const decay = Math.exp(-dt / tau);
  // nA × 1000 gives pA; pA × ms / pF gives mV.
  const drive = current * 1000 * tau / CAPACITANCE_PF;
  const candidate = REST + (voltage - REST) * decay + drive * -Math.expm1(-dt / tau);
  const spike = candidate >= THRESHOLD;
  return { voltage: spike ? REST : candidate, candidate, spike };
}

export async function mountNeuron(host) {
  const { Application, Graphics } = await import('pixi.js');
  const surface = host.querySelector('.scene');
  const app = new Application();
  await app.init({ width: 1240, height: 360, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(window.devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  app.canvas.style.width = '100%';
  app.canvas.style.height = '100%';
  surface.appendChild(app.canvas);
  surface.querySelector('.status')?.remove();
  const graphics = new Graphics();
  app.stage.addChild(graphics);
  const labels = document.createElement('div');
  labels.className = 'neuron-labels';
  surface.appendChild(labels);
  const controls = new AbortController();
  const get = name => host.querySelector(`[data-${name}]`);
  const synapses = host.dataset.synapses === 'true';
  const weights = synapses ? [...host.querySelectorAll('[data-weight]')].map(input => Number(input.value)) : [];
  const spikeTrains = SOURCE_RATES_HZ.map((rate, i) => makeSpikeTrain(rate, [0x9e3779b9, 0x243f6a88, 0xb7e15162][i]));
  const sourceFiredAt = [-Infinity, -Infinity, -Infinity];
  let time = 0, voltage = REST, count = 0, nextInput = 8, firedAt = -Infinity;
  let pulses = [], spikes = [], history = [{ t: 0, v: REST }];
  let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let visible = true, disposed = false, carry = 0, width = 1240, height = 360;
  let strength = synapses ? 0 : Number(get('strength').value), tau = synapses ? 30 : Number(get('leak').value);
  const blue = 0x176d91, rose = 0xbc3866, ink = 0x263343, grey = 0xdce3e8;

  function label(value, x, y, size = 20, color = ink) {
    const node = document.createElement('span');
    node.textContent = value;
    Object.assign(node.style, { position: 'absolute', left: `${100 * x / width}%`, top: `${100 * y / height}%`,
      fontSize: `${size * surface.clientWidth / width}px`, lineHeight: '1.2', whiteSpace: 'nowrap',
      color: `#${color.toString(16).padStart(6, '0')}` });
    labels.appendChild(node);
    return node;
  }

  let voltageLabel, leftTimeLabel, rightTimeLabel, weightLabels = [];
  function mathLabel(markup, x, y, size = 20, color = ink) {
    const node = label('', x, y, size, color);
    node.innerHTML = markup;
    return node;
  }
  function geometry() {
    const mobile = width === 620;
    const geometry = mobile
      ? { cx: 148, cy: 160, radius: 48, inputX: 22, outputX: 280, gaugeH: 66, plot: { x: 354, y: 98, w: 238, h: 150 } }
      : { cx: 290, cy: 164, radius: 66, inputX: 65, outputX: 540, gaugeH: 92, plot: { x: 730, y: 68, w: 450, h: 218 } };
    if (synapses) {
      geometry.cx = mobile ? 205 : 380;
      geometry.radius = mobile ? 42 : 60;
      geometry.sources = [72, 164, 256].map(y => ({ x: mobile ? 115 : 160, y }));
    }
    return geometry;
  }
  function layout() {
    labels.replaceChildren();
    const { cx, cy, radius, inputX, outputX, gaugeH, plot, sources } = geometry();
    if (synapses) {
      weightLabels = sources.map((source, i) => {
        mathLabel(`<i>s</i><sub>${i + 1}</sub>`, source.x - 12, source.y - 42, 18, blue);
        const highlighted = SOURCE_RATES_HZ[i] === Math.max(...SOURCE_RATES_HZ);
        const rate = label(`${SOURCE_RATES_HZ[i]} Hz`, width === 620 ? 8 : 22, source.y - 13, 23, highlighted ? blue : 0x687787);
        rate.style.fontWeight = highlighted ? '600' : '400';
        return mathLabel(`<i>w</i><sub>${i + 1}</sub>`, width === 620 ? 142 : 230, (width === 620 ? [52, 190, 274] : [52, 132, 274])[i], 18);
      });
    } else mathLabel('Input <i>I</i><sub>j</sub>', inputX, cy - radius - 42, 20, blue);
    label('Spike', outputX - 36, cy - radius - 42, 20, rose);
    const membraneLabel = mathLabel('Membrane voltage <i>V</i><sub>m</sub>', cx, synapses && width === 620 ? 312 : cy + radius + 18, 18);
    membraneLabel.style.transform = 'translateX(-50%)';
    voltageLabel = label('', cx - 36, synapses && width === 620 ? 338 : cy + radius + 46, 18);
    mathLabel('<i>V</i><sub>m</sub> (mV)', plot.x - 37, plot.y - 54, 18);
    for (const voltage of [-45, -50, -55, -60, -65]) {
      const y = plot.y + (-45 - voltage) / 20 * plot.h;
      label(`−${Math.abs(voltage)}`, plot.x - 43, y - 10, 17);
    }
    mathLabel('<i>V</i><sub>th</sub>', plot.x + plot.w - 32, plot.y + plot.h / 4 - 28, 18, rose);
    leftTimeLabel = label('0', plot.x - 4, plot.y + plot.h + 12, 17);
    mathLabel('<i>t</i> (ms)', plot.x + plot.w / 2 - 24, plot.y + plot.h + 12, 17);
    rightTimeLabel = label('120', plot.x + plot.w - 30, plot.y + plot.h + 12, 17);
  }
  function draw() {
    const { cx, cy, radius, inputX, outputX, gaugeH, plot, sources } = geometry();
    graphics.clear();
    if (synapses) {
      for (const [i, source] of sources.entries()) {
        graphics.moveTo(source.x + 16, source.y).lineTo(cx - radius, cy).stroke({ color: blue, width: weights[i] ? 0.6 + weights[i] * 0.6875 : 1, alpha: weights[i] ? 0.5 : 0.1 });
        const flash = Math.max(0, 1 - (time - sourceFiredAt[i]) / 4);
        graphics.circle(source.x, source.y, 16).fill({ color: blue, alpha: 0.05 + 0.65 * flash }).stroke({ color: blue, width: 1.5 });
      }
    } else graphics.moveTo(inputX, cy).lineTo(cx - radius, cy).stroke({ color: grey, width: 2 });
    graphics.moveTo(cx + radius, cy).lineTo(outputX, cy).stroke({ color: grey, width: 2 });
    graphics.moveTo(outputX - 12, cy - 8).lineTo(outputX, cy).lineTo(outputX - 12, cy + 8).stroke({ color: grey, width: 2 });
    graphics.circle(cx, cy, radius).fill({ color: blue, alpha: 0.025 }).stroke({ color: blue, width: 1.5 });
    const glow = Math.max(0, 1 - (time - firedAt) / 5);
    if (glow > 0) graphics.circle(cx, cy, radius + 7).stroke({ color: rose, width: 3, alpha: glow });
    const gaugeX = cx - 16, gaugeY = cy - gaugeH / 2;
    graphics.roundRect(gaugeX, gaugeY, 32, gaugeH, 5).fill({ color: grey, alpha: 0.6 });
    const fillH = Math.max(2, (voltage - REST) / (THRESHOLD - REST) * gaugeH);
    graphics.rect(gaugeX + 2, gaugeY + gaugeH - fillH, 28, fillH).fill({ color: blue, alpha: 0.8 });
    graphics.moveTo(gaugeX - 8, gaugeY).lineTo(gaugeX + 40, gaugeY).stroke({ color: rose, width: 2 });
    for (const pulse of pulses) {
      const progress = Math.min(1, (time - pulse.sent) / 6);
      const start = synapses ? { x: sources[pulse.source].x + 16, y: sources[pulse.source].y } : { x: inputX, y: cy };
      graphics.circle(start.x + progress * (cx - radius - start.x), start.y + progress * (cy - start.y), synapses ? 8 : 5).fill(blue);
    }
    for (const t of spikes) {
      const progress = (time - t) / 6;
      graphics.circle(cx + radius + progress * (outputX - cx - radius), cy, 6).fill(rose);
    }
    graphics.moveTo(plot.x, plot.y).lineTo(plot.x, plot.y + plot.h).lineTo(plot.x + plot.w, plot.y + plot.h).stroke({ color: grey, width: 1.5 });
    for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
      const y = plot.y + plot.h * fraction;
      graphics.moveTo(plot.x - 5, y).lineTo(plot.x, y).stroke({ color: grey, width: 1.5 });
    }
    for (let x = plot.x; x < plot.x + plot.w; x += 12) {
      graphics.moveTo(x, plot.y + plot.h / 4).lineTo(Math.min(x + 6, plot.x + plot.w), plot.y + plot.h / 4).stroke({ color: rose, width: 1.5 });
    }
    const cutoff = Math.max(0, time - 120);
    let started = false;
    for (const point of history) {
      if (point.t < cutoff) continue;
      const x = plot.x + (point.t - cutoff) / 120 * plot.w;
      const y = plot.y + plot.h - (Math.min(point.v, THRESHOLD) - REST) / (-45 - REST) * plot.h;
      if (!started) { graphics.moveTo(x, y); started = true; }
      else graphics.lineTo(x, y);
    }
    graphics.stroke({ color: blue, width: 2 });
    if (voltageLabel) voltageLabel.textContent = `${voltage.toFixed(1)} mV`;
    if (leftTimeLabel) leftTimeLabel.textContent = String(Math.round(cutoff));
    if (rightTimeLabel) rightTimeLabel.textContent = String(Math.round(cutoff + 120));
    host.dataset.voltage = String(voltage);
    host.dataset.spikes = String(count);
    host.dataset.time = String(time);
  }

  function sendPulse(source, sent = time) { pulses.push({ sent, arrival: sent + 6, strength, source }); }
  function step() {
    if (synapses) {
      for (let i = 0; i < 3; i++) for (const sent of spikeTrains[i](time)) {
        sendPulse(i, sent); sourceFiredAt[i] = sent;
      }
    } else if (time >= nextInput) { sendPulse(); nextInput += 12; }
    const incoming = pulses.filter(p => p.arrival <= time);
    pulses = pulses.filter(p => p.arrival > time);
    const arrivals = weights.map((_, i) => incoming.filter(p => p.source === i).length);
    const current = synapses ? weightedInput(weights, arrivals) : incoming.reduce((sum, pulse) => sum + pulse.strength, 0);
    const result = advanceVoltage(voltage, tau, current);
    time += DT;
    history.push({ t: time, v: result.candidate });
    voltage = result.voltage;
    if (result.spike) history.push({ t: time, v: REST });
    history = history.filter(p => p.t >= time - 121);
    if (result.spike) { count++; firedAt = time; spikes.push(time); }
    spikes = spikes.filter(t => t >= time - 6);

  }

  function updatePlayback() {
    get('pause').textContent = playing ? 'Pause' : 'Play';
    if (playing && visible && !document.hidden) app.start();
    else app.stop();
  }
  const on = (name, event, callback) => get(name).addEventListener(event, callback, { signal: controls.signal });
  if (!synapses) on('strength', 'input', () => { strength = Number(get('strength').value); get('strength-value').value = `${strength} nA`; });
  if (!synapses) on('leak', 'input', () => { tau = Number(get('leak').value); get('leak-value').value = `${tau} ms`; });
  for (const input of host.querySelectorAll('[data-weight]')) {
    input.addEventListener('input', () => {
      const i = Number(input.dataset.weight);
      weights[i] = Number(input.value);
      host.querySelector(`[data-weight-value="${i}"]`).value = `${weights[i]} nA`;
      draw(); app.render();
    }, { signal: controls.signal });
  }
  on('pause', 'click', () => { playing = !playing; carry = 0; updatePlayback(); });
  document.addEventListener('visibilitychange', updatePlayback, { signal: controls.signal });
  const observer = new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); carry = 0; updatePlayback(); });
  observer.observe(host);
  const resize = new ResizeObserver(() => {
    if (disposed) return;
    const mobile = matchMedia('(max-width: 600px)').matches;
    width = mobile ? 620 : 1240; height = 360;
    app.renderer.resize(width, height);
    app.canvas.style.width = '100%'; app.canvas.style.height = '100%';
    layout(); draw(); app.render();
  });
  resize.observe(surface);
  app.ticker.add(ticker => {
    carry += Math.min(ticker.elapsedMS, 100) * 0.04;
    while (carry >= DT) { step(); carry -= DT; }
    draw();
  });
  for (const button of host.querySelectorAll('button')) button.disabled = false;
  layout(); draw(); app.render(); updatePlayback();
  return () => {
    disposed = true; controls.abort(); observer.disconnect(); resize.disconnect(); labels.remove();
    app.destroy(true, { children: true });
  };
}
