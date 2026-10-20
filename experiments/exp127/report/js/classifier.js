export async function mountClassifier(host) {
  const [{ Application, Graphics }, response] = await Promise.all([import('pixi.js'), fetch(host.dataset.source)]);
  if (!response.ok) throw new Error(`Classifier data: HTTP ${response.status}`);
  const data = await response.json();
  if (data.schema !== 'exp127.classifier/v2' || data.examples.length !== 2 || data.duration_ms !== 200) throw new Error('Invalid classifier recording');
  for (const mode of ['before', 'after']) {
    if (data.modes[mode].weights.length !== 3 || data.modes[mode].weights.some(row => row.length !== 2)) throw new Error('Invalid weight axes');
    for (const example of data.examples) {
      const voltage = example.modes[mode].voltages_mv;
      if (voltage.length !== data.duration_ms / data.dt_ms || voltage.some(row => row.length !== 2 || row.some(value => !Number.isFinite(value)))) throw new Error('Invalid voltage recording');
    }
  }
  const surface = host.querySelector('.scene'), app = new Application();
  await app.init({ width: 1240, height: 400, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(window.devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  surface.appendChild(app.canvas);
  surface.querySelector('.status')?.remove();
  const graphics = new Graphics();
  app.stage.addChild(graphics);
  const labels = document.createElement('div');
  labels.className = 'classifier-labels';
  surface.appendChild(labels);
  const controls = new AbortController(), get = name => host.querySelector(`[data-${name}]`);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ink = 0x263343, blue = 0x176d91, purple = 0x846da7, muted = 0x687787, grey = 0xdce3e8;
  const colors = [blue, purple];
  const maxWeight = Math.max(...Object.values(data.modes).flatMap(mode => mode.weights.flat()));
  let mode = 'before', exampleIndex = 0, time = 0, playing = !reducedMotion;
  let visible = true, disposed = false, width = 1240, height = 400;
  let countLabels = [], predictionLabel, clockLabel, snapshotLabel;
  const example = () => data.examples[exampleIndex];
  const snapshot = () => example().modes[mode];

  function label(value, x, y, size = 20, color = ink) {
    const node = document.createElement('span');
    node.textContent = value;
    Object.assign(node.style, { position: 'absolute', left: `${x / width * 100}%`, top: `${y / height * 100}%`,
      fontSize: `${size * surface.clientWidth / width}px`, color: `#${color.toString(16).padStart(6, '0')}`,
      whiteSpace: 'nowrap', lineHeight: '1.2' });
    labels.appendChild(node);
    return node;
  }
  function geometry() {
    return width === 640
      ? { sources: [100, 200, 300].map(y => ({ x: 115, y })), targets: [155, 285].map(y => ({ x: 252, y })),
          sourceRadius: 14, targetRadius: 38, barX: 402, barW: 205, barY: [155, 265] }
      : { sources: [110, 205, 300].map(y => ({ x: 160, y })), targets: [135, 295].map(y => ({ x: 510, y })),
          sourceRadius: 20, targetRadius: 48, barX: 770, barW: 370, barY: [150, 255] };
  }
  function layout() {
    labels.replaceChildren();
    const g = geometry(), item = example();
    const largestRate = Math.max(...item.rates_hz);
    label('3 inputs → 2 outputs', 22, 18, 22);
    snapshotLabel = label('', 22, height - 32, 18, muted);
    clockLabel = label('', g.barX, height - 32, 18, muted);
    g.sources.forEach((source, i) => {
      const name = label(`s${i + 1}`, source.x, source.y - g.sourceRadius - 32, 20, blue);
      name.style.transform = 'translateX(-50%)';
      const highlighted = item.rates_hz[i] === largestRate;
      const rate = label(`${Math.round(item.rates_hz[i])} Hz`, width === 640 ? 8 : 22, source.y - 13, 23, highlighted ? blue : muted);
      rate.style.fontWeight = highlighted ? '600' : '400';
    });
    g.targets.forEach((target, i) => {
      const name = label(`Class ${data.class_names[i]}`, target.x, target.y - g.targetRadius - 34, 20, colors[i]);
      name.style.transform = 'translateX(-50%)';
    });
    label('Output spike counts', g.barX, 18, 22);
    label(`True class: ${data.class_names[item.label]}`, g.barX, 64, 20);
    countLabels = g.barY.map((y, i) => {
      label(`Class ${data.class_names[i]}`, g.barX, y - 33, 20, colors[i]);
      return label('0', g.barX + g.barW - 25, y - 33, 20, colors[i]);
    });
    predictionLabel = label('', g.barX, 326, 20);
  }
  function updateSelection() {
    for (const button of host.querySelectorAll('[data-mode]')) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    for (const button of host.querySelectorAll('button[data-example]')) button.setAttribute('aria-pressed', String(Number(button.dataset.example) === exampleIndex));
    host.dataset.mode = mode;
    host.dataset.example = String(exampleIndex);
  }
  function draw() {
    const g = geometry(), item = example(), current = snapshot();
    const weights = data.modes[mode].weights;
    const counts = [0, 0];
    for (const [t, cell] of current.outputs) if (t <= time) counts[cell]++;
    graphics.clear();
    for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) {
      const source = g.sources[i], target = g.targets[j], weight = weights[i][j];
      graphics.moveTo(source.x + g.sourceRadius, source.y).lineTo(target.x - g.targetRadius, target.y)
        .stroke({ color: colors[j], width: weight ? 0.75 + weight / maxWeight * 5.5 : 1, alpha: weight ? 0.38 : 0.08 });
    }
    for (const [t, i] of item.inputs) {
      const progress = (time - t) / data.delay_ms;
      if (progress < 0 || progress > 1) continue;
      const source = g.sources[i];
      for (let j = 0; j < 2; j++) {
        if (weights[i][j] === 0) continue;
        const target = g.targets[j];
        graphics.circle(source.x + g.sourceRadius + progress * (target.x - g.targetRadius - source.x - g.sourceRadius),
          source.y + progress * (target.y - source.y), 8).fill(colors[j]);
      }
    }
    g.sources.forEach((source, i) => {
      const flash = item.inputs.some(([t, cell]) => cell === i && time >= t && time - t < 3);
      graphics.circle(source.x, source.y, g.sourceRadius).fill({ color: blue, alpha: flash ? 0.8 : 0.04 }).stroke({ color: blue, width: 1.5 });
    });
    g.targets.forEach((target, i) => {
      const flash = current.outputs.some(([t, cell]) => cell === i && time >= t && time - t < 3);
      graphics.circle(target.x, target.y, g.targetRadius).fill({ color: colors[i], alpha: 0.025 }).stroke({ color: colors[i], width: 1.5 });
      if (flash) graphics.circle(target.x, target.y, g.targetRadius + 6).stroke({ color: colors[i], width: 3, alpha: 0.8 });
      const step = Math.min(current.voltages_mv.length - 1, Math.floor(time / data.dt_ms));
      const voltage = current.voltages_mv[step][i];
      const gaugeH = g.targetRadius * 1.4, gaugeW = g.targetRadius * 0.5;
      const x = target.x - gaugeW / 2, y = target.y - gaugeH / 2;
      graphics.roundRect(x, y, gaugeW, gaugeH, 4).fill({ color: grey, alpha: 0.65 });
      const fraction = Math.max(0, Math.min(1, (voltage - data.rest_mv) / (data.threshold_mv - data.rest_mv)));
      const fillH = Math.max(2, fraction * gaugeH);
      graphics.rect(x + 2, y + gaugeH - fillH, gaugeW - 4, fillH).fill({ color: colors[i], alpha: 0.85 });
      graphics.moveTo(x - 7, y).lineTo(x + gaugeW + 7, y).stroke({ color: 0xbc3866, width: 1.5 });
    });
    const maxCount = Math.max(1, ...Object.values(item.modes).flatMap(value => value.counts));
    for (let i = 0; i < 2; i++) {
      const y = g.barY[i];
      graphics.roundRect(g.barX, y, g.barW, 23, 4).fill({ color: grey, alpha: 0.55 });
      if (counts[i]) graphics.roundRect(g.barX, y, counts[i] / maxCount * g.barW, 23, 4).fill({ color: colors[i], alpha: 0.8 });
      countLabels[i].textContent = String(counts[i]);
    }
    const complete = time >= data.duration_ms;
    const tied = counts[0] === counts[1], prediction = counts[1] > counts[0] ? 1 : 0;
    predictionLabel.textContent = !complete && counts[0] + counts[1] === 0 ? 'No output spikes yet'
      : tied && !complete ? 'Prediction: tied' : `Prediction: ${data.class_names[prediction]}${tied ? ' (tie)' : ''}`;
    predictionLabel.style.color = complete || !tied ? `#${colors[prediction].toString(16)}` : `#${muted.toString(16)}`;
    clockLabel.textContent = `${Math.floor(time)} / ${data.duration_ms} ms`;
    snapshotLabel.textContent = mode === 'before' ? 'Initial weights' : `Trained · epoch ${data.selected_epoch}`;
    host.dataset.time = String(time);
    host.dataset.counts = JSON.stringify(counts);
    host.dataset.prediction = predictionLabel.textContent;
    host.dataset.voltages = JSON.stringify(current.voltages_mv[Math.min(current.voltages_mv.length - 1, Math.floor(time / data.dt_ms))]);
  }
  function updatePlayback() {
    get('pause').textContent = playing ? 'Pause' : 'Play';
    if (playing && visible && !document.hidden) app.start(); else app.stop();
  }
  function reset() { time = 0; playing = !reducedMotion; layout(); updateSelection(); draw(); app.render(); updatePlayback(); }
  for (const button of host.querySelectorAll('[data-mode]')) button.addEventListener('click', () => { mode = button.dataset.mode; reset(); }, { signal: controls.signal });
  for (const button of host.querySelectorAll('button[data-example]')) button.addEventListener('click', () => {
    const selected = Number(button.dataset.example);
    if (selected === exampleIndex) return;
    exampleIndex = selected;
    reset();
  }, { signal: controls.signal });
  get('replay').addEventListener('click', reset, { signal: controls.signal });
  get('pause').addEventListener('click', () => {
    if (!playing && time >= data.duration_ms) time = 0;
    playing = !playing; updatePlayback();
  }, { signal: controls.signal });
  document.addEventListener('visibilitychange', updatePlayback, { signal: controls.signal });
  const observer = new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); updatePlayback(); });
  observer.observe(host);
  const resize = new ResizeObserver(() => {
    if (disposed) return;
    const mobile = matchMedia('(max-width: 600px)').matches;
    width = mobile ? 640 : 1240; height = mobile ? 440 : 400;
    app.renderer.resize(width, height);
    app.canvas.style.width = '100%'; app.canvas.style.height = '100%';
    layout(); draw(); app.render();
  });
  resize.observe(surface);
  app.ticker.add(ticker => {
    time = Math.min(data.duration_ms, time + Math.min(ticker.elapsedMS, 100) * 0.02);
    draw();
    if (time >= data.duration_ms) { playing = false; updatePlayback(); }
  });
  for (const control of host.querySelectorAll('button')) control.disabled = false;
  app.canvas.style.width = '100%'; app.canvas.style.height = '100%';
  layout(); updateSelection(); draw(); app.render(); updatePlayback();
  return () => { disposed = true; controls.abort(); observer.disconnect(); resize.disconnect(); labels.remove(); app.destroy(true, { children: true }); };
}
