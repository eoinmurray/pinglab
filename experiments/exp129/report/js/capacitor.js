export const CHARGE_SCALE = 120e-6;

// SI units throughout the model; the view converts to μF, kΩ, μC and mA.
export function current(voltage, battery, resistance, mode) {
  return mode === 'hold' ? 0 : ((mode === 'charge' ? battery : 0) - voltage) / resistance;
}
export function advanceVoltage(voltage, battery, resistance, capacitance, mode, dt) {
  if (mode === 'hold') return voltage;
  const target = mode === 'charge' ? battery : 0;
  return target + (voltage - target) * Math.exp(-dt / (resistance * capacitance));
}

export async function mountCapacitor(host) {
  const { Application, Graphics } = await import('pixi.js');
  const surface = host.querySelector('.scene'), app = new Application();
  await app.init({ width: 1240, height: 360, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  surface.appendChild(app.canvas); surface.querySelector('.status')?.remove();
  const graphics = new Graphics(); app.stage.addChild(graphics);
  const labels = document.createElement('div'); labels.className = 'capacitor-math-labels';
  labels.setAttribute('aria-hidden', 'true'); surface.appendChild(labels);
  const events = new AbortController(), get = name => host.querySelector(`[data-${name}]`);
  const blue = 0x176d91, rose = 0xbc3866, ink = 0x263343, grey = 0xdce3e8;
  let width, height = 360, circuit, plots, voltageLabel, chargeLabels, modeLabel, leftTimes, rightTimes;
  let time = 0, voltage = 0, displayCount = 0, scheduledCount = 0, transfers = [], mode = 'charge', capacitance, resistance, battery;
  let history = [], visible = true, disposed = false;
  let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  function label(markup, x, y, size = 18, color = ink, centered = false) {
    const node = document.createElement('span'); node.innerHTML = markup;
    Object.assign(node.style, { position: 'absolute', left: `${100 * x / width}%`, top: `${100 * y / height}%`,
      fontSize: `${size * surface.clientWidth / width}px`, lineHeight: '1.2', whiteSpace: 'nowrap',
      color: `#${color.toString(16).padStart(6, '0')}`, transform: centered ? 'translateX(-50%)' : '' });
    labels.appendChild(node); return node;
  }
  function line(x1, y1, x2, y2, color = grey, thickness = 1.5, alpha = 1) {
    graphics.moveTo(x1, y1).lineTo(x2, y2).stroke({ color, width: thickness, alpha });
  }
  function layout() {
    labels.replaceChildren(); leftTimes = []; rightTimes = [];
    const mobile = matchMedia('(max-width: 600px)').matches;
    width = mobile ? 620 : 1240; app.renderer.resize(width, height);
    app.canvas.style.width = '100%'; app.canvas.style.height = '100%';
    const panel = width / 3;
    const sceneWidth = 2 * panel;
    circuit = { panel: sceneWidth, left: sceneWidth * 0.14, mid: sceneWidth * 0.65, half: sceneWidth * 0.2 };
    label('A', 16, 14, 18);
    modeLabel = label('', sceneWidth / 2, 32, 18, ink, true);
    label('R', (circuit.left + circuit.mid) / 2, 48, 17, ink, true);
    label('Gap', circuit.mid + circuit.half + 12, 191, 15);
    voltageLabel = label('', circuit.mid - circuit.half - 15, 191, 16);
    voltageLabel.style.transform = 'translateX(-100%)';
    chargeLabels = [label('', circuit.mid + circuit.half + 12, 120, 16, rose), label('', circuit.mid + circuit.half + 12, 240, 16, blue)];
    label('<i>V</i><sub>b</sub>', circuit.left + 34, 179, 17);
    label('Electrons', sceneWidth / 2, 335, 17, blue, true);
    plots = [
      { x: sceneWidth + 54, y: 48, w: panel - 74, h: 100, min: 0, max: 6, field: 'v', color: blue, ticks: [0, 3, 6] },
      { x: sceneWidth + 54, y: 220, w: panel - 74, h: 100, min: -0.6, max: 0.6, field: 'i', color: rose, ticks: [-0.6, 0, 0.6] },
    ];
    label('B  <i>V</i><sub>C</sub> (V)', plots[0].x - 37, 14);
    label('C  <i>I</i> (mA)', plots[1].x - 37, 184);
    for (const plot of plots) {
      for (const value of plot.ticks) {
        const node = label(String(value).replace('-', '−'), plot.x - 12,
          plot.y + (plot.max - value) / (plot.max - plot.min) * plot.h - 10, 17);
        node.style.transform = 'translateX(-100%)';
      }
      leftTimes.push(label('', plot.x - 4, plot.y + plot.h + 12, 17));
      label('<i>t</i> (s)', plot.x + plot.w / 2 - 15, plot.y + plot.h + 12, 17);
      const right = label('', plot.x + plot.w, plot.y + plot.h + 12, 17);
      right.style.transform = 'translateX(-100%)'; rightTimes.push(right);
    }
  }
  function electron(x, y, alpha = 1, radius = 5) {
    if (alpha <= 0) return;
    graphics.circle(x, y, radius).fill({ color: blue, alpha });
  }
  function gridSlot(y, column, row) {
    const { mid, half } = circuit;
    return [mid - half + (column + 0.5) * (2 * half / 16), y + 16 + row * 32];
  }
  function transferSlot(y, upper, index) {
    const column = upper ? 7 - Math.floor(index / 2) : 8 + Math.floor(index / 2);
    return gridSlot(y, column, index % 2);
  }
  function plate(y, upper, count) {
    const { mid, half } = circuit;
    graphics.roundRect(mid - half, y, 2 * half, 64, 4)
      .fill({ color: upper ? rose : blue, alpha: 0.04 })
      .stroke({ color: grey, width: 1.5 });
    for (let column = 1; column < 16; column++) {
      const x = mid - half + column * 2 * half / 16;
      line(x, y, x, y + 64, grey, 1, 0.6);
    }
    line(mid - half, y + 32, mid + half, y + 32, grey, 1, 0.6);
    const radius = Math.min(4.5, half / 16 * 0.85);
    // Each plate starts half occupied: the left eight columns of one common grid.
    for (let row = 0; row < 2; row++) for (let column = 0; column < 8; column++) {
      const transferIndex = 2 * (7 - column) + row;
      if (upper && transferIndex < count) continue;
      electron(...gridSlot(y, column, row), 0.85, radius);
    }
    if (!upper) for (let k = 0; k < count; k++) {
      electron(...transferSlot(y, false, k), 0.85, radius);
    }
  }
  function path(points, color = grey, thickness = 2) {
    graphics.moveTo(...points[0]); for (const point of points.slice(1)) graphics.lineTo(...point);
    graphics.stroke({ color, width: thickness });
  }
  function along(points, fraction) {
    const lengths = points.slice(1).map((p, k) => Math.hypot(p[0] - points[k][0], p[1] - points[k][1]));
    let distance = fraction * lengths.reduce((a, b) => a + b, 0);
    for (let k = 0; k < lengths.length; k++) {
      if (distance <= lengths[k]) {
        const f = distance / lengths[k];
        return [points[k][0] + f * (points[k + 1][0] - points[k][0]), points[k][1] + f * (points[k + 1][1] - points[k][1])];
      }
      distance -= lengths[k];
    }
    return points.at(-1);
  }
  function drawCircuit() {
    const { left, mid } = circuit;
    const q = capacitance * voltage;
    const gap = 160 / (capacitance * 1e6);
    const upperY = 200 - gap / 2 - 64, lowerY = 200 + gap / 2;
    const batteryScale = 0.65 + 0.7 * (battery - 1) / 5;
    const sourceTop = 185 - 24 * batteryScale, sourceBottom = 185 + 24 * batteryScale;
    const top = [[mid, upperY], [mid, 78], [left, 78], [left, sourceTop]];
    const bottom = [[left, sourceBottom], [left, 316], [mid, 316], [mid, lowerY + 64]];
    path(top); path(bottom);
    const rx = (left + mid) / 2;
    const resistorWidth = 24 + 72 * (resistance / 1e3 - 10) / 90;
    graphics.rect(rx - resistorWidth / 2, 71, resistorWidth, 14).fill(0xffffff).stroke({ color: ink, width: 1.5 });
    if (mode === 'charge') {
      const positiveY = 185 - 11 * batteryScale, negativeY = 185 + 11 * batteryScale;
      line(left - 13 * batteryScale, positiveY, left + 13 * batteryScale, positiveY, ink, 2);
      line(left - 7 * batteryScale, negativeY, left + 7 * batteryScale, negativeY, ink, 2);
      line(left, sourceTop, left, positiveY); line(left, negativeY, left, sourceBottom);
      const signX = left - 13 * batteryScale - 8;
      line(signX - 3, positiveY, signX + 3, positiveY, rose, 1.5);
      line(signX, positiveY - 3, signX, positiveY + 3, rose, 1.5);
      line(signX - 3, negativeY, signX + 3, negativeY, blue, 1.5);
    } else if (mode === 'hold') {
      line(left, sourceTop, left + 12, 185, ink, 2);
      graphics.circle(left, sourceTop, 3).fill(ink); graphics.circle(left, sourceBottom, 3).fill(ink);
    } else line(left, sourceTop, left, sourceBottom, ink, 2);
    const alpha = voltage / 6;
    for (const side of [-1, 1]) {
      const x = mid + side * circuit.half * 0.82;
      const startY = upperY + 65, endY = lowerY - 1, tip = Math.min(5, gap / 3);
      line(x, startY, x, endY, rose, 1.4, alpha * 0.7);
      line(x - 3, endY - tip, x, endY, rose, 1.4, alpha * 0.7);
      line(x + 3, endY - tip, x, endY, rose, 1.4, alpha * 0.7);
    }
    const outgoing = transfers.length;
    const upperCount = displayCount + (mode === 'charge' ? outgoing : 0);
    const lowerCount = displayCount - (mode === 'discharge' ? outgoing : 0);
    plate(upperY, true, upperCount); plate(lowerY, false, lowerCount);
    for (const transfer of transfers) {
      const progress = Math.min(1, (time - transfer.start) / 0.4);
      const upperSlot = transferSlot(upperY, true, transfer.index);
      const lowerSlot = transferSlot(lowerY, false, transfer.index);
      if (mode === 'charge') {
        electron(...along([upperSlot, ...top], progress), 0.85, 4.5);
        electron(...along([...bottom, lowerSlot], progress), 0.85, 4.5);
      } else if (mode === 'discharge') {
        const route = [lowerSlot, ...bottom.toReversed(), [left, sourceTop], ...top.toReversed().slice(1), upperSlot];
        electron(...along(route, progress), 0.85, 4.5);
      }
    }
    Object.assign(host.dataset, { upperElectrons: String(16 - upperCount), lowerElectrons: String(16 + lowerCount),
      inFlight: String(transfers.length), settledTransfers: String(displayCount) });
    modeLabel.textContent = mode === 'charge' ? 'Charging' : mode === 'hold' ? 'Holding' : 'Discharging';
    voltageLabel.textContent = `${voltage.toFixed(2)} V`;
    chargeLabels[0].style.top = `${100 * (upperY + 22) / height}%`;
    chargeLabels[1].style.top = `${100 * (lowerY + 22) / height}%`;
    Object.assign(host.dataset, { plateGap: String(gap), resistorWidth: String(resistorWidth), batteryScale: String(batteryScale) });
    chargeLabels[0].textContent = `+${(q * 1e6).toFixed(1)} μC`;
    chargeLabels[1].textContent = `−${(q * 1e6).toFixed(1)} μC`;
  }
  function drawPlot(plot) {
    const cutoff = Math.max(0, time - 10);
    const y = value => plot.y + plot.h * (plot.max - value) / (plot.max - plot.min);
    line(plot.x, plot.y, plot.x, plot.y + plot.h);
    line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h);
    for (const tick of plot.ticks) line(plot.x - 5, y(tick), plot.x + plot.w, y(tick), grey, 1, 0.6);
    let previous;
    for (const point of history) {
      if (point.t < cutoff) continue;
      const p = { x: plot.x + (point.t - cutoff) / 10 * plot.w, y: y(point[plot.field]) };
      if (previous) graphics.moveTo(previous.x, previous.y).lineTo(p.x, p.y);
      previous = p;
    }
    graphics.stroke({ color: plot.color, width: 2 });
  }
  function draw() {
    if (!circuit) return;
    graphics.clear(); drawCircuit(); plots.forEach(drawPlot);
    const cutoff = Math.max(0, time - 10);
    leftTimes.forEach(node => { node.textContent = cutoff.toFixed(1); });
    rightTimes.forEach(node => { node.textContent = (cutoff + 10).toFixed(1); });
    const i = current(voltage, battery, resistance, mode), q = capacitance * voltage;
    Object.assign(host.dataset, { time: String(time), voltage: String(voltage), current: String(i), charge: String(q),
      tau: String(resistance * capacitance), playing: String(playing), state: mode });
  }
  function record() { history.push({ t: time, v: voltage, i: current(voltage, battery, resistance, mode) * 1e3 }); }
  function settleDisplay() {
    displayCount = Math.round(12 * capacitance * voltage / CHARGE_SCALE);
    scheduledCount = displayCount; transfers = [];
  }
  function step(dt) {
    voltage = advanceVoltage(voltage, battery, resistance, capacitance, mode, dt);
    time += dt;
    const target = Math.round(12 * capacitance * voltage / CHARGE_SCALE);
    while (scheduledCount !== target) {
      const direction = target > scheduledCount ? 1 : -1;
      const index = direction > 0 ? scheduledCount : scheduledCount - 1;
      scheduledCount += direction;
      transfers.push({ index, to: scheduledCount, start: time });
    }
    transfers = transfers.filter(transfer => {
      if (time - transfer.start < 0.4) return true;
      displayCount = transfer.to; return false;
    });
    record(); history = history.filter(p => p.t >= time - 10.05);
  }
  function refresh() { draw(); app.render(); }
  function playback() {
    get('pause').textContent = playing ? 'Pause' : 'Play';
    if (playing && visible && !document.hidden) app.start(); else app.stop();
    refresh();
  }
  function restart() {
    capacitance = Number(get('capacitance').value) * 1e-6;
    resistance = Number(get('resistance').value) * 1e3;
    battery = Number(get('battery').value);
    get('capacitance-value').value = `${Number(get('capacitance').value)} μF`;
    get('resistance-value').value = `${Number(get('resistance').value)} kΩ`;
    get('battery-value').value = `${battery} V`;
    mode = 'charge'; get('mode').value = mode; time = 0; voltage = 0; displayCount = 0; scheduledCount = 0; transfers = []; history = []; record(); refresh();
  }
  for (const name of ['capacitance', 'resistance', 'battery']) get(name).addEventListener('input', restart, { signal: events.signal });
  get('mode').addEventListener('change', () => { record(); mode = get('mode').value; settleDisplay(); record(); refresh(); }, { signal: events.signal });
  get('restart').addEventListener('click', restart, { signal: events.signal });
  get('pause').addEventListener('click', () => { playing = !playing; playback(); }, { signal: events.signal });
  document.addEventListener('visibilitychange', playback, { signal: events.signal });
  const observer = new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); playback(); }); observer.observe(host);
  const resize = new ResizeObserver(() => { if (!disposed) { layout(); refresh(); } }); resize.observe(surface);
  app.ticker.add(ticker => {
    const elapsed = Math.min(ticker.elapsedMS, 100) / 1000;
    const steps = Math.max(1, Math.ceil(elapsed / 0.01));
    for (let k = 0; k < steps; k++) step(elapsed / steps);
    draw();
  });
  get('pause').disabled = false; get('restart').disabled = false;
  layout(); restart(); playback();
  return () => { disposed = true; events.abort(); observer.disconnect(); resize.disconnect(); labels.remove(); app.destroy(true, { children: true }); };
}
