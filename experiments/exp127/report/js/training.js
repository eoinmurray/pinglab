export async function mountTraining(host) {
  const [{ Application, Graphics }, response] = await Promise.all([import('pixi.js'), fetch(host.dataset.source)]);
  if (!response.ok) throw new Error(`Training recording: HTTP ${response.status}`);
  const data = await response.json();
  if (data.schema !== 'exp127.training/v1' || data.steps.length !== data.window_length || !data.losses.every(Number.isFinite)) throw new Error('Invalid training trace');
  const surface = host.querySelector('.scene'), app = new Application();
  await app.init({ width: 1240, height: 400, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(window.devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  surface.appendChild(app.canvas); surface.querySelector('.status')?.remove();
  const graphics = new Graphics(); app.stage.addChild(graphics);
  const labels = document.createElement('div'); labels.className = 'training-labels'; surface.appendChild(labels);
  const controls = new AbortController(), get = name => host.querySelector(`[data-${name}]`);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const blue = 0x176d91, purple = 0x846da7, rose = 0xbc3866, ink = 0x263343, muted = 0x687787, grey = 0xdce3e8, colors = [blue, purple];
  const maxWeight = Math.max(...data.steps.flatMap(s => [...s.weights_before.flat(), ...s.weights_after.flat()]));
  const maxGradient = Math.max(...data.steps.flatMap(s => s.gradients.flat().map(Math.abs)));
  const maxLoss = Math.max(...data.losses) * 1.1;
  let width = 1240, height = 400, index = 0, exampleIndex = 0, elapsed = 0, playing = !reduced, visible = true, disposed = false;
  let iterationLabel, lossLabel, countLabels = [];
  const cycle = 9000;
  const weightLabels = Array.from({ length: 3 }, (_, i) => Array.from({ length: 2 }, (_, j) => host.querySelector(`[data-weight="${i}-${j}"]`)));
  const changeLabels = Array.from({ length: 3 }, (_, i) => Array.from({ length: 2 }, (_, j) => host.querySelector(`[data-change="${i}-${j}"]`)));
  function stage() { return elapsed < 3000 ? 0 : elapsed < 4000 ? 1 : elapsed < 5700 ? 2 : 3; }
  function geometry() {
    return width === 640
      ? { sources: [95, 180, 265].map(y => ({ x: 115, y })), targets: [135, 265].map(y => ({ x: 390, y })), radius: 38, plot: { x: 80, y: 390, w: 520, h: 170 } }
      : { sources: [105, 200, 295].map(y => ({ x: 160, y })), targets: [135, 275].map(y => ({ x: 510, y })), radius: 44, plot: { x: 770, y: 90, w: 420, h: 215 } };
  }
  function label(value, x, y, size = 20, color = ink) {
    const node = document.createElement('span'); node.textContent = value;
    Object.assign(node.style, { position: 'absolute', left: `${x / width * 100}%`, top: `${y / height * 100}%`, fontSize: `${size * surface.clientWidth / width}px`, color: `#${color.toString(16).padStart(6, '0')}`, whiteSpace: 'nowrap', lineHeight: '1.2' });
    labels.appendChild(node); return node;
  }
  function layout() {
    labels.replaceChildren(); const g = geometry(), item = data.examples[exampleIndex], p = g.plot;
    label(`Training example: ${exampleIndex ? 'B' : 'A'}`, 22, 18, 22);
    const largest = Math.max(...item.rates_hz);
    g.sources.forEach((source, i) => {
      const name = label(`s${i + 1}`, source.x, source.y - 45, 20, blue); name.style.transform = 'translateX(-50%)';
      const rate = label(`${Math.round(item.rates_hz[i])} Hz`, width === 640 ? 8 : 22, source.y - 13, 23, item.rates_hz[i] === largest ? blue : muted);
      rate.style.fontWeight = item.rates_hz[i] === largest ? '600' : '400';
    });
    countLabels = g.targets.map((target, i) => {
      const name = label(`Class ${i ? 'B' : 'A'}`, target.x, target.y - g.radius - 32, 20, colors[i]); name.style.transform = 'translateX(-50%)';
      const count = label('', target.x + g.radius + 16, target.y - 12, 20, colors[i]); return count;
    });
    label('Training loss · full batch', p.x, p.y - 55, 22);
    for (const fraction of [0, 0.5, 1]) label((maxLoss * fraction).toFixed(2), p.x - 56, p.y + p.h * (1 - fraction) - 10, 16, muted);
    label('1', p.x - 4, p.y + p.h + 12, 17, muted);
    label(String(data.losses.length), p.x + p.w - 16, p.y + p.h + 12, 17, muted);
    label('Iteration', p.x + p.w / 2 - 38, p.y + p.h + 12, 18, muted);
    iterationLabel = label('', 22, width === 640 ? 306 : 358, 19, muted);
    lossLabel = label('', p.x, p.y + p.h + 50, 19, blue);
  }
  function draw() {
    const g = geometry(), p = g.plot, step = data.steps[index], item = data.examples[exampleIndex], current = step.examples[exampleIndex], phase = stage();
    const time = Math.min(data.duration_ms, elapsed / 3000 * data.duration_ms);
    const blend = phase === 3 ? Math.min(1, (elapsed - 5700) / 2500) : 0;
    const weights = step.weights_before.map((row, i) => row.map((w, j) => w + blend * (step.weights_after[i][j] - w)));
    for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) {
      weightLabels[i][j].textContent = weights[i][j].toFixed(3);
      weightLabels[i][j].parentElement.dataset.updating = String(phase === 3);
      const change = step.weights_after[i][j] - step.weights_before[i][j];
      changeLabels[i][j].textContent = phase === 3 ? `${change >= 0 ? '+' : '−'}${Math.abs(change).toFixed(3)}` : '';
    }
    host.dataset.weights = JSON.stringify(weights);
    graphics.clear();
    for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) {
      const source = g.sources[i], target = g.targets[j];
      graphics.moveTo(source.x + 16, source.y).lineTo(target.x - g.radius, target.y).stroke({ color: phase === 2 ? rose : colors[j], width: 0.75 + weights[i][j] / maxWeight * 10, alpha: phase === 2 ? 0.55 : phase === 3 ? 0.75 : 0.45 });
      if (phase === 2 && step.gradients[i][j] !== 0) {
        const progress = ((elapsed - 4000) / 850) % 1;
        graphics.circle(target.x - g.radius + progress * (source.x + 16 - target.x + g.radius), target.y + progress * (source.y - target.y), 4 + Math.abs(step.gradients[i][j]) / maxGradient * 5).fill(rose);
      }
    }
    if (phase === 0) for (const [t, i] of item.inputs) {
      const progress = (time - t) / 8;
      if (progress < 0 || progress > 1) continue;
      const source = g.sources[i];
      for (let j = 0; j < 2; j++) {
        if (!weights[i][j]) continue;
        const target = g.targets[j];
        graphics.circle(source.x + 16 + progress * (target.x - g.radius - source.x - 16), source.y + progress * (target.y - source.y), 8).fill(colors[j]);
      }
    }
    g.sources.forEach((source, i) => {
      const flash = phase === 0 && item.inputs.some(([t, cell]) => cell === i && time >= t && time - t < 5);
      graphics.circle(source.x, source.y, 16).fill({ color: blue, alpha: flash ? 0.75 : 0.04 }).stroke({ color: blue, width: 1.5 });
    });
    const counts = [0, 0]; for (const [t, cell] of current.outputs) if (t <= time) counts[cell]++;
    g.targets.forEach((target, i) => {
      graphics.circle(target.x, target.y, g.radius).fill({ color: colors[i], alpha: 0.025 }).stroke({ color: colors[i], width: 1.5 });
      const flash = phase === 0 && current.outputs.some(([t, cell]) => cell === i && time >= t && time - t < 5);
      if (flash) graphics.circle(target.x, target.y, g.radius + 5).stroke({ color: colors[i], width: 3, alpha: 0.8 });
      const voltage = current.voltages_mv[Math.min(current.voltages_mv.length - 1, Math.floor(time / data.dt_ms))][i];
      const gh = g.radius * 1.4, gw = g.radius * 0.5, x = target.x - gw / 2, y = target.y - gh / 2;
      graphics.roundRect(x, y, gw, gh, 4).fill({ color: grey, alpha: 0.65 });
      const filled = Math.max(2, Math.max(0, Math.min(1, (voltage - data.rest_mv) / (data.threshold_mv - data.rest_mv))) * gh);
      graphics.rect(x + 2, y + gh - filled, gw - 4, filled).fill({ color: colors[i], alpha: 0.85 });
      graphics.moveTo(x - 6, y).lineTo(x + gw + 6, y).stroke({ color: rose, width: 1.5 });
      countLabels[i].textContent = String(counts[i]);
    });
    const plotX = iteration => p.x + (iteration - 1) / (data.losses.length - 1) * p.w;
    const plotY = loss => p.y + p.h * (1 - loss / maxLoss);
    const first = plotX(data.window_start + 1), last = plotX(data.window_start + data.window_length);
    graphics.rect(first, p.y, Math.max(3, last - first), p.h).fill({ color: blue, alpha: 0.09 });
    for (const fraction of [0, 0.5, 1]) graphics.moveTo(p.x, p.y + p.h * fraction).lineTo(p.x + p.w, p.y + p.h * fraction).stroke({ color: grey, width: 1 });
    graphics.moveTo(p.x, p.y).lineTo(p.x, p.y + p.h).lineTo(p.x + p.w, p.y + p.h).stroke({ color: grey, width: 1.5 });
    data.losses.forEach((loss, i) => { if (i === 0) graphics.moveTo(plotX(i + 1), plotY(loss)); else graphics.lineTo(plotX(i + 1), plotY(loss)); });
    graphics.stroke({ color: blue, width: 2.5 });
    const x = plotX(step.iteration), y = plotY(step.loss);
    graphics.moveTo(x, p.y).lineTo(x, p.y + p.h).stroke({ color: blue, alpha: 0.4, width: 1.5 });
    graphics.circle(x, y, 6).fill(phase === 1 ? rose : blue);
    const previousLoss = index > 0 ? data.steps[index - 1].loss : 0;
    const lossProgress = phase === 0 ? 0 : phase === 1 ? Math.min(1, (elapsed - 3000) / 1000) : 1;
    const displayedLoss = previousLoss + (step.loss - previousLoss) * lossProgress;
    const lossWidth = p.w * Math.min(1, displayedLoss / maxLoss);
    graphics.roundRect(p.x, p.y + p.h + 80, p.w, 10, 4).fill({ color: grey, alpha: 0.55 });
    if (lossWidth > 0) graphics.roundRect(p.x, p.y + p.h + 80, lossWidth, 10, Math.min(4, lossWidth / 2)).fill({ color: rose, alpha: 0.85 });
    host.dataset.lossFraction = String(lossWidth / p.w);
    lossLabel.textContent = phase === 0 && index === 0 ? 'Batch loss: …' : `Batch loss: ${displayedLoss.toFixed(3)}`;
    iterationLabel.textContent = `Iteration ${step.iteration} / ${data.losses.length} · ${data.samples} examples`;
    for (const node of host.querySelectorAll('[data-phase]')) node.dataset.active = String(Number(node.dataset.phase) === phase);
    for (const button of host.querySelectorAll('button[data-example]')) button.setAttribute('aria-pressed', String(Number(button.dataset.example) === exampleIndex));
    host.dataset.phase = String(phase); host.dataset.iteration = String(step.iteration); host.dataset.loss = String(step.loss); host.dataset.example = String(exampleIndex);
  }
  function updatePlayback() { get('pause').textContent = playing ? 'Pause' : 'Play'; if (playing && visible && !document.hidden) app.start(); else app.stop(); }
  function reset() { index = 0; elapsed = 0; playing = !reduced; draw(); app.render(); updatePlayback(); }
  get('replay').addEventListener('click', reset, { signal: controls.signal });
  get('next').addEventListener('click', () => { index = (index + 1) % data.steps.length; elapsed = 0; draw(); app.render(); }, { signal: controls.signal });
  get('pause').addEventListener('click', () => { if (!playing && index === data.steps.length - 1 && elapsed >= cycle) { index = 0; elapsed = 0; } playing = !playing; updatePlayback(); }, { signal: controls.signal });
  for (const button of host.querySelectorAll('button[data-example]')) button.addEventListener('click', () => { exampleIndex = Number(button.dataset.example); elapsed = 0; layout(); draw(); app.render(); }, { signal: controls.signal });
  document.addEventListener('visibilitychange', updatePlayback, { signal: controls.signal });
  const observer = new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); updatePlayback(); }); observer.observe(host);
  const resize = new ResizeObserver(() => { if (disposed) return; const mobile = matchMedia('(max-width: 600px)').matches; width = mobile ? 640 : 1240; height = mobile ? 660 : 400; app.renderer.resize(width, height); app.canvas.style.width = '100%'; app.canvas.style.height = '100%'; layout(); draw(); app.render(); }); resize.observe(surface);
  app.ticker.add(ticker => { elapsed = Math.min(cycle, elapsed + Math.min(ticker.elapsedMS, 100)); if (elapsed >= cycle) { if (index < data.steps.length - 1) { index++; elapsed = 0; } else playing = false; } draw(); updatePlayback(); });
  for (const button of host.querySelectorAll('button')) button.disabled = false;
  app.canvas.style.width = '100%'; app.canvas.style.height = '100%'; layout(); draw(); app.render(); updatePlayback();
  return () => { disposed = true; controls.abort(); observer.disconnect(); resize.disconnect(); labels.remove(); app.destroy(true, { children: true }); };
}
