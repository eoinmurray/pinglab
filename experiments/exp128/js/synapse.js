export const DT = 0.25, DELAY = 8, INTERVAL = 30;
export const THRESHOLD = -50, RESET = -65;
export const REST = -65, CAPACITANCE = 100, LEAK = 10;
export const G_MAX = 14, V_MIN = -80, V_MAX = -20;
export const synapticCurrent = (g, e, v) => g * (e - v);
export const decayConductance = (g, tau, dt = DT) => g * Math.exp(-dt / tau);

// Exact passive-membrane update for a conductance held fixed over this step.
export function advanceVoltage(voltage, g, reversal, dt = DT) {
  const total = LEAK + g;
  const equilibrium = (LEAK * REST + g * reversal) / total;
  return equilibrium + (voltage - equilibrium) * Math.exp(-dt * total / CAPACITANCE);
}

export function advanceDualVoltage(voltage, gE, gI, reversalE = 0, reversalI = -75, dt = DT) {
  const total = LEAK + gE + gI;
  const equilibrium = (LEAK * REST + gE * reversalE + gI * reversalI) / total;
  return equilibrium + (voltage - equilibrium) * Math.exp(-dt * total / CAPACITANCE);
}

export async function mountSynapse(host) {
  const { Application, Graphics } = await import('pixi.js');
  const surface = host.querySelector('.scene'), app = new Application();
  await app.init({ width: 1240, height: 360, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  surface.appendChild(app.canvas); surface.querySelector('.status')?.remove();
  const graphics = new Graphics(); app.stage.addChild(graphics);
  const mathLabels = document.createElement('div');
  mathLabels.className = 'synapse-math-labels'; mathLabels.setAttribute('aria-hidden', 'true'); surface.appendChild(mathLabels);
  const controls = new AbortController(), get = name => host.querySelector(`[data-${name}]`);
  const dual = host.dataset.dual === 'true';
  const voltageMin = dual ? -70 : V_MIN;
  const voltageMax = dual ? -45 : V_MAX;
  const voltageTicks = dual ? [-45, -50, -55, -60, -65, -70] : [-20, -35, -50, -65, -80];
  const voltageY = value => voltagePlot.y + voltagePlot.h * (voltageMax - value) / (voltageMax - voltageMin);
  const blue = 0x176d91, rose = 0xbc3866, ink = 0x263343, grey = 0xdce3e8;
  let time = 0, g = 0, gI = 0, voltage = REST, carry = 0, nextArrival = 14, nextArrivalI = 24;
  let count = 0, firedAt = -Infinity, outputSpikes = [];
  let history = [], width, height, scheme, conductancePlot, voltagePlot;
  let neuronLabel, countLabel, reversalLabels = [], leftTimes = [], rightTimes = [];
  let reversal, reversalI = -75, weight, weightI = 0, tau;
  let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let visible = true, disposed = false;
  function label(text, x, y, size = 14, color = ink) {
    const node = mathLabel('', x, y, size);
    node.textContent = text;
    node.style.color = `#${color.toString(16).padStart(6, '0')}`;
    return node;
  }
  function mathLabel(markup, x, y, size = 20, color = ink) {
    const node = document.createElement('span');
    node.innerHTML = markup;
    Object.assign(node.style, { position: 'absolute', left: `${100 * x / width}%`, top: `${100 * y / height}%`, fontSize: `${size * surface.clientWidth / width}px`, lineHeight: '1.2', whiteSpace: 'nowrap', overflow: 'visible', color: `#${color.toString(16).padStart(6, '0')}` });
    mathLabels.appendChild(node); return node;
  }
  function line(x1, y1, x2, y2, color = grey, thickness = 1.5) {
    graphics.moveTo(x1, y1).lineTo(x2, y2).stroke({ color, width: thickness });
  }
  function layout() {
    mathLabels.replaceChildren(); leftTimes = []; rightTimes = [];
    const mobile = matchMedia('(max-width: 600px)').matches;
    width = mobile ? 620 : 1240; height = 360;
    surface.style.removeProperty('height');
    app.renderer.resize(width, height);
    app.canvas.style.width = '100%'; app.canvas.style.height = '100%';
    const panelWidth = width / 3;
    const cx = panelWidth / 2 - (mobile ? 12 : 32), cellHeight = mobile ? 96 : 132;
    const neuronY = 188;
    const membraneY = neuronY - cellHeight / 2;
    scheme = { w: panelWidth, cx, spikeY: dual ? 44 : 30, my: membraneY, ny: neuronY, cellHeight, cellWidth: mobile ? (dual ? 150 : 112) : 180 };
    conductancePlot = { x: panelWidth + 54, y: 68, w: panelWidth - 74, h: 218 };
    voltagePlot = { x: 2 * panelWidth + 54, y: 68, w: panelWidth - 74, h: 218 };
    label('A', 16, 14, 18);
    if (dual) {
      const offset = mobile ? 26 : 32;
      for (const [name, x, color] of [['E', cx - offset, blue], ['I', cx + offset, rose]]) {
        label(name, x, 4, 18, color).style.transform = 'translateX(-50%)';
        const channelLabel = mathLabel(`<i>g</i><sub>${name}</sub>`, x + (name === 'E' ? -24 : 24), scheme.my - 32, 18, color);
        if (name === 'E') channelLabel.style.transform = 'translateX(-100%)';
      }
    } else {
      label('Spike', cx + 16, scheme.spikeY - 8, 20, blue);
      mathLabel('<i>g</i><sub>syn</sub>', cx + 32, scheme.my - 30, 18, blue);
    }
    const neuronName = mathLabel('Membrane voltage <i>V</i><sub>m</sub>', cx, scheme.my + cellHeight + 18, 18);
    neuronName.style.transform = 'translateX(-50%)';
    neuronLabel = label('', cx, scheme.ny, 18);
    neuronLabel.style.transform = 'translate(-50%, -50%)';
    countLabel = label('', cx, scheme.ny + 24, 15, rose); countLabel.style.transform = 'translateX(-50%)';
    if (!mobile) label('Output', panelWidth - 100, scheme.ny - 30, 16, rose);
    mathLabel(dual ? 'B  <i>g</i><sub>E</sub>, <i>g</i><sub>I</sub> (nS)' : 'B  <i>g</i><sub>syn</sub> (nS)', conductancePlot.x - 37, 14, 18);
    mathLabel('C  <i>V</i><sub>m</sub> (mV)', voltagePlot.x - 37, 14, 18);
    for (const [plot, ticks, min, max] of [[conductancePlot, [14, 7, 0], 0, 14], [voltagePlot, voltageTicks, voltageMin, voltageMax]]) {
      for (const value of ticks) {
        const tick = label(String(value).replace('-', '−'), plot.x - 12, plot.y + (max - value) / (max - min) * plot.h - 10, 17);
        tick.style.transform = 'translateX(-100%)';
      }
      leftTimes.push(label('', plot.x - 4, plot.y + plot.h + 12, 17));
      mathLabel('<i>t</i> (ms)', plot.x + plot.w / 2 - 24, plot.y + plot.h + 12, 17);
      const rightTime = label('', plot.x + plot.w, plot.y + plot.h + 12, 17);
      rightTime.style.transform = 'translateX(-100%)'; rightTimes.push(rightTime);
    }
    mathLabel('<i>V</i><sub>th</sub>', voltagePlot.x + voltagePlot.w - 25, voltageY(THRESHOLD) - 28, 18, rose);
    mathLabel('<i>E</i><sub>L</sub>', voltagePlot.x + voltagePlot.w - 25, voltageY(REST) - 28, 18);
    reversalLabels = (dual ? ['E', 'I'] : ['syn']).map((name, index) =>
      mathLabel(`<i>E</i><sub>${name}</sub>`, voltagePlot.x + 8, voltagePlot.y, 17, index ? rose : blue));
  }

  function drawTrace(plot, field, min, max, color, axes = true) {
    const cutoff = Math.max(0, time - 120);
    if (axes) {
    line(plot.x, plot.y, plot.x, plot.y + plot.h);
    line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h);
    for (const fraction of field === 'g' ? [0, 0.5, 1] : voltageTicks.map(value => (voltageMax - value) / (voltageMax - voltageMin))) {
      const y = plot.y + plot.h * fraction;
      line(plot.x - 5, y, plot.x, y);
    }
    }
    let previous;
    for (const point of history) {
      if (point.t < cutoff) continue;
      const x = plot.x + (point.t - cutoff) / 120 * plot.w;
      const y = plot.y + plot.h * (1 - (point[field] - min) / (max - min));
      if (previous) {
        const dx = x - previous.x, dy = y - previous.y;
        let start = 0, end = 1;
        if (dy === 0) {
          if (y < plot.y || y > plot.y + plot.h) end = -1;
        } else {
          const top = (plot.y - previous.y) / dy;
          const bottom = (plot.y + plot.h - previous.y) / dy;
          start = Math.max(0, Math.min(top, bottom));
          end = Math.min(1, Math.max(top, bottom));
        }
        if (start <= end) {
          graphics.moveTo(previous.x + dx * start, previous.y + dy * start);
          graphics.lineTo(previous.x + dx * end, previous.y + dy * end);
        }
      }
      previous = { x, y };
    }
    graphics.stroke({ color, width: 2, alpha: field === 'v' ? 0.35 : 1 });
  }
  function draw() {
    if (!scheme) return; graphics.clear();
    const { w, cx: mid, spikeY, my, ny, cellHeight, cellWidth } = scheme;
    const color = voltage < REST - 0.01 ? rose : blue;
    const offset = width === 620 ? 26 : 32;
    const channels = dual
      ? [{ x: mid - offset, g, color: blue, first: 6, interval: 30 }, { x: mid + offset, g: gI, color: rose, first: 16, interval: 40 }]
      : [{ x: mid, g, color: blue, first: 6, interval: INTERVAL }];
    for (const channel of channels) {
      channel.aperture = 1 + 30 * Math.min(1, channel.g / G_MAX);
      channel.left = channel.x - channel.aperture / 2 - 7;
      channel.right = channel.x + channel.aperture / 2 + 7;
      const startY = dual ? spikeY + 16 : spikeY;
      line(channel.x, startY, channel.x, my - 22, grey, 1.5);
      if (dual) {
        const age = time >= channel.first ? (time - channel.first) % channel.interval : -1;
        const flash = age >= 0 ? Math.max(0, 1 - age / 3) : 0;
        graphics.circle(channel.x, spikeY, 16).fill({ color: channel.color, alpha: 0.05 + flash * 0.6 })
          .stroke({ color: channel.color, width: 1.5 });
      }
    }
    const radius = dual ? 30 : 40;
    graphics.roundRect(mid - cellWidth / 2, my - 5, cellWidth, cellHeight + 10, radius)
      .fill({ color, alpha: 0.025 });
    // The outline joins each channel wall directly, leaving each pore open.
    function membraneOutline(left, top, span, tall, corner) {
      const right = left + span, bottom = top + tall;
      graphics.moveTo(channels.at(-1).right, top)
        .lineTo(right - corner, top).quadraticCurveTo(right, top, right, top + corner)
        .lineTo(right, bottom - corner).quadraticCurveTo(right, bottom, right - corner, bottom)
        .lineTo(left + corner, bottom).quadraticCurveTo(left, bottom, left, bottom - corner)
        .lineTo(left, top + corner).quadraticCurveTo(left, top, left + corner, top)
        .lineTo(channels[0].left, top).stroke({ color: grey, width: 2 });
      for (let i = 1; i < channels.length; i++) line(channels[i - 1].right, top, channels[i].left, top, grey, 2);
    }
    membraneOutline(mid - cellWidth / 2, my - 5, cellWidth, cellHeight + 10, radius);
    membraneOutline(mid - cellWidth / 2 + 10, my + 5, cellWidth - 20, cellHeight - 10, radius - 10);
    for (const channel of channels) {
      graphics.roundRect(channel.left, my - 20, 7, 40, 3).fill(channel.color);
      graphics.roundRect(channel.x + channel.aperture / 2, my - 20, 7, 40, 3).fill(channel.color);
      const age = time >= channel.first ? (time - channel.first) % channel.interval : -1;
      if (age >= 0 && age < DELAY) {
        const startY = dual ? spikeY + 16 : spikeY;
        graphics.circle(channel.x, startY + (my - startY) * age / DELAY, 5).fill(channel.color);
      }
    }

    const outputX = w - 16, startX = mid + cellWidth / 2;
    line(startX, ny, outputX, ny, grey, 2);
    line(outputX - 10, ny - 6, outputX, ny); line(outputX - 10, ny + 6, outputX, ny);
    for (const at of outputSpikes) graphics.circle(startX + (outputX - startX) * (time - at) / DELAY, ny, 5).fill(rose);
    const flash = Math.max(0, 1 - (time - firedAt) / 4);
    if (flash > 0) graphics.roundRect(mid - cellWidth / 2 - 5, my - 10, cellWidth + 10, cellHeight + 20, radius + 5).stroke({ color: rose, width: 2, alpha: flash });
    neuronLabel.textContent = `${voltage.toFixed(1)} mV`;
    countLabel.textContent = `${count} spikes`;
    const restY = voltageY(REST);
    for (let x = voltagePlot.x; x < voltagePlot.x + voltagePlot.w; x += 12)
      line(x, restY, Math.min(x + 6, voltagePlot.x + voltagePlot.w), restY, grey, 1);
    const thresholdY = voltageY(THRESHOLD);
    for (let x = voltagePlot.x; x < voltagePlot.x + voltagePlot.w; x += 12) line(x, thresholdY, Math.min(x + 6, voltagePlot.x + voltagePlot.w), thresholdY, rose, 1);
    drawTrace(conductancePlot, 'g', 0, G_MAX, blue);
    if (dual) drawTrace(conductancePlot, 'gI', 0, G_MAX, rose, false);
    drawTrace(voltagePlot, 'v', voltageMin, voltageMax, color);
    let previousLabelY = -Infinity;
    for (const [index, value] of (dual ? [reversal, reversalI] : [reversal]).entries()) {
      const above = value > voltageMax, below = value < voltageMin;
      const y = voltageY(Math.max(voltageMin, Math.min(voltageMax, value)));
      for (let x = voltagePlot.x; x < voltagePlot.x + voltagePlot.w; x += 14)
        line(x, y, Math.min(x + 7, voltagePlot.x + voltagePlot.w), y, index ? rose : blue, 1.5);
      let labelY = above ? y + 6 : y - 24;
      if (Math.abs(labelY - previousLabelY) < 22) labelY = previousLabelY + 22;
      previousLabelY = labelY;
      const node = reversalLabels[index];
      node.style.top = `${100 * labelY / height}%`;
      node.innerHTML = `<i>E</i><sub>${dual ? (index ? 'I' : 'E') : 'syn'}</sub> ${above ? '↑ ' : below ? '↓ ' : ''}${String(value).replace('-', '−')} mV`;
    }
    const cutoff = Math.max(0, time - 120);
    leftTimes.forEach(node => { node.textContent = cutoff.toFixed(0); });
    rightTimes.forEach(node => { node.textContent = (cutoff + 120).toFixed(0); });
    host.dataset.conductance = String(g); host.dataset.voltage = String(voltage);
    host.dataset.conductanceI = String(gI);
    host.dataset.current = String(synapticCurrent(g, reversal, voltage) + synapticCurrent(gI, reversalI, voltage)); host.dataset.time = String(time); host.dataset.spikes = String(count);
  }
  function step() {
    const midpointG = decayConductance(g, tau, DT / 2);
    voltage = dual ? advanceDualVoltage(voltage, midpointG, decayConductance(gI, 20, DT / 2), reversal, reversalI) : advanceVoltage(voltage, midpointG, reversal);
    g = decayConductance(g, tau); gI = decayConductance(gI, 20); time += DT;
    if (voltage >= THRESHOLD) {
      history.push({ t: time, g, gI, v: THRESHOLD });
      voltage = RESET; count++; firedAt = time; outputSpikes.push(time);
    }
    outputSpikes = outputSpikes.filter(at => at >= time - DELAY);
    if (time >= nextArrival) {
      history.push({ t: time, g, gI, v: voltage });
      g += weight; nextArrival += INTERVAL;
    }
    if (dual && time >= nextArrivalI) {
      history.push({ t: time, g, gI, v: voltage });
      gI += weightI; nextArrivalI += 40;
    }
    history.push({ t: time, g, gI, v: voltage }); history = history.filter(p => p.t >= time - 121);
  }
  function refresh() { draw(); app.render(); }
  function preview() {
    reversal = Number(get('reversal').value); reversalI = dual ? Number(get('reversal-i').value) : -75; weight = Number(get('weight').value); tau = dual ? 12 : Number(get('tau').value);
    weightI = dual ? Number(get('weight-i').value) : 0;
    get('weight-value').value = `${weight} nS`;
    get('reversal-value').value = `${reversal} mV`;
    if (dual) { get('weight-i-value').value = `${weightI} nS`; get('reversal-i-value').value = `${reversalI} mV`; }
    else get('tau-value').value = `${tau} ms`;
    count = 0; firedAt = -Infinity; outputSpikes = []; time = 0; g = 0; gI = 0; voltage = REST; carry = 0; nextArrival = 14; nextArrivalI = 24; history = [{ t: 0, g: 0, gI: 0, v: REST }];
    // Recompute a complete preview immediately, even when playback is paused.
    for (let k = 0; k < 120 / DT; k++) step();
    refresh();
  }
  function playback() {
    get('pause').textContent = playing ? 'Pause' : 'Play';
    if (playing && visible && !document.hidden) app.start(); else app.stop();
  }
  for (const name of dual ? ['weight', 'weight-i', 'reversal', 'reversal-i'] : ['reversal', 'weight', 'tau']) get(name).addEventListener('input', preview, { signal: controls.signal });
  get('pause').addEventListener('click', () => { playing = !playing; carry = 0; playback(); }, { signal: controls.signal });
  document.addEventListener('visibilitychange', () => { carry = 0; playback(); }, { signal: controls.signal });
  const observer = new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); carry = 0; playback(); }); observer.observe(host);
  const resize = new ResizeObserver(() => { if (!disposed) { layout(); refresh(); } }); resize.observe(surface);
  app.ticker.add(ticker => { carry += Math.min(ticker.elapsedMS, 100) * 0.03; while (carry >= DT) { step(); carry -= DT; } draw(); });
  get('pause').disabled = false; layout(); preview(); playback();
  return () => { disposed = true; controls.abort(); observer.disconnect(); resize.disconnect(); mathLabels.remove(); app.destroy(true, { children: true }); };
}
