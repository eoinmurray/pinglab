// Replay only: events and connectivity come from a validated present export.
import { createPistons } from './pistons.js';
import { createAfferentRate } from './afferent-rate.js';

export async function mountCircuit(host) {
  const [{ Application, Container, Graphics }, response] = await Promise.all([
    import('pixi.js'), fetch(host.dataset.source),
  ]);
  if (!response.ok) throw new Error(`Circuit data: HTTP ${response.status}`);
  const data = await response.json();
  if (data.schema !== 'exp099.circuit/v3' || !(data.duration_ms > 0)) throw new Error('Invalid circuit recording');
  const surface = host.querySelector('.scene');
  const sceneWidth = 1240, sceneHeight = 540;
  const app = new Application();
  await app.init({ width: sceneWidth, height: sceneHeight, backgroundAlpha: 0, antialias: true,
    resolution: Math.min(window.devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  surface.appendChild(app.canvas);
  surface.querySelector('.status')?.remove();
  const scene = new Container();
  app.stage.addChild(scene);
  // Browser text avoids GPU glyph-texture clipping at fractional/retina scales.
  const labels = document.createElement('div');
  labels.className = 'circuit-labels';
  surface.appendChild(labels);
  const colors = { e: 0x176d91, i: 0xbc3866, ink: 0x263343, muted: 0x687787 };
  function text(value, x, y, size = 15, color = colors.ink) {
    const label = document.createElement('span');
    label.textContent = value;
    Object.assign(label.style, { position: 'absolute', left: `${x / sceneWidth * 100}%`, top: `${y / sceneHeight * 100}%`,
      fontSize: `calc(var(--circuit-scale, 1) * ${size}px)`,
      color: `#${color.toString(16).padStart(6, '0')}`, whiteSpace: 'nowrap', lineHeight: '1.2' });
    labels.appendChild(label);
    return label;
  }
  const clock = text('0 ms', 784, 8, 24);
  clock.style.transform = 'translateX(-100%)';
  clock.setAttribute('role', 'timer');
  text('A', 0, 12, 14);
  text('E private', 16, 12, 19, colors.e);
  text('E population', 432, 12, 19, colors.e);
  text('I private', 16, 336, 19, colors.i);
  text('I population', 432, 336, 19, colors.i);
  text(`${data.afferents_per_neuron} independent sources / target`, 16, 282, 12);
  text(`${data.population_sizes.e.toLocaleString()} neurons · ${data.groups.e.indices.length} shown`, 432, 282, 12);
  text(`${data.afferents_per_neuron} independent sources / target`, 16, 510, 12);
  text(`${data.population_sizes.i.toLocaleString()} neurons · ${data.groups.i.indices.length} shown`, 432, 510, 12);
  const boxes = { private_e: [16, 40, 352, 232], e: [432, 40, 352, 232],
    private_i: [16, 368, 352, 132], i: [432, 368, 352, 132] };
  const frames = new Graphics();
  for (const [key, [x, y, w, h]] of Object.entries(boxes)) {
    frames.roundRect(x, y, w, h, 9).fill({ color: colors[key.endsWith('e') ? 'e' : 'i'], alpha: 0.025 })
      .stroke({ color: 0xdce3e8, width: 1 });
  }
  scene.addChild(frames);
  const particles = new Graphics(), cells = new Container();
  const recurrentPulses = new Graphics();
  scene.addChild(particles, cells, recurrentPulses);
  const groups = {};
  for (const [key, group] of Object.entries(data.groups)) {
    const [x, y, w, h] = boxes[key];
    const cols = key.endsWith('e') ? 16 : 10;
    const rows = Math.ceil(group.indices.length / cols);
    const color = colors[key.endsWith('e') ? 'e' : 'i'];
    const nodes = group.indices.map((identity, index) => {
      const node = { x: x + 14 + index % cols * (w - 28) / (cols - 1),
        y: y + 12 + Math.floor(index / cols) * (h - 24) / Math.max(1, rows - 1), color,
        view: new Container(), core: new Graphics(), halo: new Graphics(), intensity: 0 };
      node.view.position.set(node.x, node.y);
      node.core.circle(0, 0, key.startsWith('private') ? 2.1 : 3).fill(color);
      node.halo.circle(0, 0, 7).fill({ color, alpha: 0.15 });
      node.halo.alpha = 0;
      node.view.addChild(node.halo, node.core); cells.addChild(node.view);
      return node;
    });
    groups[key] = { nodes, events: group.events, cursor: 0 };
  }
  const paths = data.edges.map(([src, a, dst, b]) => ({ source: groups[src].nodes[a],
    target: groups[dst].nodes[b], src, index: a, color: 0x626b74, private: false, crossPopulation: src !== dst }));
  for (const pop of ['e', 'i']) {
    groups[pop].nodes.forEach((target, index) => paths.push({ source: groups[`private_${pop}`].nodes[index],
      target, src: `private_${pop}`, index, color: 0x626b74, private: true }));
  }
  const outgoing = {};
  for (const path of paths) {
    (outgoing[`${path.src}:${path.index}`] ??= []).push(path);
  }
  const drawPistons = createPistons({ data, scene, text, Graphics, colors });
  const drawAfferentRate = createAfferentRate({ data, scene, text, Graphics, colors });
  let time = 0, visible = true, paused = false, scrubbing = false;
  const pauseButton = host.querySelector('[data-pause]');
  const speedSlider = host.querySelector('[data-speed]');
  const speedValue = host.querySelector('[data-speed-value]');
  const scrubber = host.querySelector('[data-scrub]');
  scrubber.min = String(data.start_ms);
  scrubber.max = String(data.start_ms + data.duration_ms);
  scrubber.step = String(data.dt_ms);
  const pulses = [];
  const delay = data.delay_ms;
  function resetLoop() {
    pulses.length = 0;
    for (const group of Object.values(groups)) { group.cursor = 0; for (const node of group.nodes) node.intensity = 0; }
  }
  function draw() {
    particles.clear();
    recurrentPulses.clear();
    for (const [key, group] of Object.entries(groups)) {
      for (const node of group.nodes) node.intensity = 0;
      while (group.cursor < group.events.length && group.events[group.cursor][0] * data.dt_ms <= time) {
        const [step, index, count] = group.events[group.cursor++];
        const eventTime = step * data.dt_ms;
        // Ignore old events after a slow frame; never move them to the current time.
        if (time - eventTime < delay + 3) {
          for (const path of outgoing[`${key}:${index}`] ?? []) pulses.push({ path, time: eventTime, count });
        }
      }
      // Exact recorded times with a 3 ms display trail, preserving input multiplicity.
      for (let k = group.cursor - 1; k >= 0; k--) {
        const [step, index, count] = group.events[k];
        const age = time - step * data.dt_ms;
        if (age > 3) break;
        group.nodes[index].intensity = Math.max(group.nodes[index].intensity, Math.min(1, count / 2) * (1 - age / 3));
      }
      for (const node of group.nodes) {
        node.core.alpha = 0.22 + node.intensity * 0.78;
        node.core.scale.set(1 + node.intensity * 0.7);
        node.halo.alpha = node.intensity;
      }
    }
    for (let k = pulses.length - 1; k >= 0; k--) {
      const pulse = pulses[k], age = time - pulse.time;
      if (age > delay + 3) { pulses.splice(k, 1); continue; }
      const { source, target, color, crossPopulation } = pulse.path;
      const fraction = Math.min(1, age / delay);
      const x = source.x + (target.x - source.x) * fraction;
      const y = source.y + (target.y - source.y) * fraction;
      const fade = age <= delay ? 1 : 1 - (age - delay) / 3;
      if (crossPopulation) {
        // Draw E↔I traffic last and strongly; recorded timing is unchanged.
        const tail = Math.max(0, fraction - 0.2);
        recurrentPulses.moveTo(source.x + (target.x - source.x) * tail,
          source.y + (target.y - source.y) * tail).lineTo(x, y)
          .stroke({ color, alpha: fade * 0.9, width: 2.5 });
        recurrentPulses.circle(x, y, 7).fill({ color, alpha: fade * 0.13 });
        recurrentPulses.circle(x, y, 3.5).fill({ color, alpha: fade });
        recurrentPulses.circle(x, y, 1.2).fill({ color: 0xffffff, alpha: fade });
      } else {
        const privateInput = pulse.path.private;
        if (privateInput) {
          particles.circle(x, y, 2 + Math.min(0.7, (pulse.count - 1) * 0.35))
            .fill({ color, alpha: fade * 0.55 });
        } else {
          particles.circle(x, y, 2).fill({ color, alpha: fade * 0.65 });
        }
      }
    }
    clock.textContent = `${Math.floor(data.start_ms + time)} ms`;
    drawPistons(time);
    drawAfferentRate(time);
    scrubber.value = String(data.start_ms + time);
    scrubber.setAttribute('aria-valuetext', `${(data.start_ms + time).toFixed(1)} milliseconds`);
  }
  app.ticker.add(ticker => {
    time += Math.min(ticker.deltaMS, 100) * data.duration_ms / 25000 * Number(speedSlider.value);
    if (time >= data.duration_ms) { time %= data.duration_ms; resetLoop(); }
    draw();
  });
  const updatePlayback = () => {
    if (!paused && !scrubbing && visible && !document.hidden) app.start(); else app.stop();
  };
  const onSeek = () => {
    time = Math.max(0, Math.min(data.duration_ms, Number(scrubber.value) - data.start_ms));
    resetLoop();
    // Rebuild only the recorded events within the visible pulse trail.
    for (const group of Object.values(groups)) {
      let low = 0, high = group.events.length;
      const firstVisible = time - delay - 3;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (group.events[middle][0] * data.dt_ms < firstVisible) low = middle + 1;
        else high = middle;
      }
      group.cursor = low;
    }
    draw(); app.render();
  };
  const onScrubStart = () => { scrubbing = true; updatePlayback(); };
  const onScrubEnd = () => { if (scrubbing) { scrubbing = false; updatePlayback(); } };
  scrubber.addEventListener('input', onSeek);
  scrubber.addEventListener('pointerdown', onScrubStart);
  document.addEventListener('pointerup', onScrubEnd);
  document.addEventListener('pointercancel', onScrubEnd);
  scrubber.disabled = false;
  const onPause = () => {
    paused = !paused;
    pauseButton.textContent = paused ? 'Resume' : 'Pause';
    updatePlayback();
  };
  pauseButton.addEventListener('click', onPause);
  pauseButton.disabled = false;
  const onSpeed = () => { speedValue.value = `${Number(speedSlider.value).toFixed(2).replace(/\.?0+$/, '')}×`; };
  speedSlider.addEventListener('input', onSpeed);
  speedSlider.disabled = false;
  document.addEventListener('visibilitychange', updatePlayback);
  const visibility = new IntersectionObserver(entries => { visible = entries.some(entry => entry.isIntersecting); updatePlayback(); });
  visibility.observe(host);
  const resize = new ResizeObserver(() => {
    const width = surface.clientWidth;
    if (!width) return;
    surface.style.setProperty('--circuit-scale', String(width / sceneWidth));
    app.renderer.resize(width, width * sceneHeight / sceneWidth); scene.scale.set(width / sceneWidth); app.render();
  });
  resize.observe(surface);
  draw(); app.render(); updatePlayback();
  return () => { resize.disconnect(); visibility.disconnect();
    document.removeEventListener('visibilitychange', updatePlayback);
    pauseButton.removeEventListener('click', onPause);
    speedSlider.removeEventListener('input', onSpeed);
    scrubber.removeEventListener('input', onSeek);
    scrubber.removeEventListener('pointerdown', onScrubStart);
    document.removeEventListener('pointerup', onScrubEnd);
    document.removeEventListener('pointercancel', onScrubEnd);
    labels.remove();
    app.destroy(true, { children: true, texture: true, textureSource: true }); };
}
