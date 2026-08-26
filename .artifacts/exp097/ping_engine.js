(() => {
  const data = window.EXP097_MEASURED_STATE;
  if (!data) return;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const duration = 3200;
  const indexAt = phase => Math.min(data.ge.length - 1, Math.floor(phase * data.ge.length));
  const range = values => [Math.min(...values), Math.max(...values)];
  const geRange = range(data.ge), giRange = range(data.gi);
  const scale = (value, limits) => limits[1] > limits[0] ? (value - limits[0]) / (limits[1] - limits[0]) : 0;
  const setCylinder = (fill, plunger, value) => {
    const height = 12 + 96 * value;
    fill.setAttribute('y', String(252 - height));
    fill.setAttribute('height', String(height));
    plunger.setAttribute('transform', `translate(0 ${-96 * value})`);
  };
  const bind = (root, render) => {
    const button = root.querySelector('.exp097-animation-toggle');
    const state = { playing: !reduced, start: performance.now(), held: 0 };
    const frame = now => {
      if (!state.playing) return;
      state.held = now - state.start;
      render((state.held % duration) / duration);
      requestAnimationFrame(frame);
    };
    button.addEventListener('click', () => {
      state.playing = !state.playing;
      button.textContent = state.playing ? 'Pause' : 'Play';
      button.setAttribute('aria-pressed', String(state.playing));
      if (state.playing) { state.start = performance.now() - state.held; requestAnimationFrame(frame); }
    });
    render(0);
    if (state.playing) requestAnimationFrame(frame);
  };

  const engine = document.getElementById('exp097-engine');
  if (engine) {
    engine.querySelector('.exp097-animation-kicker').textContent = `Simulation result · input seed ${data.seed}`;
    const geFill = engine.querySelector('#exp097-ge-fill'), giFill = engine.querySelector('#exp097-gi-fill');
    const gePlunger = engine.querySelector('#exp097-ge-plunger'), giPlunger = engine.querySelector('#exp097-gi-plunger');
    const ePulse = engine.querySelector('#exp097-e-pulse'), iPulse = engine.querySelector('#exp097-i-pulse');
    const dot = engine.querySelector('#exp097-phase-dot'), trail = engine.querySelector('#exp097-phase-trail');
    const cursor = engine.querySelector('#exp097-engine-cursor'), label = engine.querySelector('#exp097-engine-state');
    const points = [];
    bind(engine, phase => {
      const index = indexAt(phase), ge = scale(data.ge[index], geRange), gi = scale(data.gi[index], giRange);
      setCylinder(geFill, gePlunger, ge); setCylinder(giFill, giPlunger, gi);
      ePulse.style.opacity = data.e_spikes[index] ? '1' : '0';
      iPulse.style.opacity = data.i_spikes[index] ? '1' : '0';
      const x = 710 + 220 * ge, y = 278 - 170 * gi;
      dot.setAttribute('cx', x); dot.setAttribute('cy', y);
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`); if (points.length > 70) points.shift();
      trail.setAttribute('d', `M${points.join(' L')}`);
      const cursorX = 18 + phase * 964; cursor.setAttribute('x1', cursorX); cursor.setAttribute('x2', cursorX);
      label.textContent = `t = ${data.time_ms[index].toFixed(1)} ms`;
    });
  }

  const cells = document.getElementById('exp097-stochasticity');
  if (cells) {
    cells.querySelector('.exp097-animation-kicker').textContent = `Simulation result · input seed ${data.seed}`;
    cells.querySelector('.exp097-title').textContent = 'Small target-specific kicks ride on a shared recurrent rhythm';
    cells.querySelector('.exp097-note').textContent = 'Bars are simulated target cells; their common cycle dominates their residual differences';
    const bars = [...cells.querySelectorAll('.exp097-cell-bar')];
    const trace = cells.querySelector('#exp097-mean-trace'), dot = cells.querySelector('#exp097-mean-dot');
    const label = cells.querySelector('#exp097-mean-label'), points = [];
    bind(cells, phase => {
      const index = indexAt(phase);
      const values = bars.map((bar, cell) => {
        const value = scale(data.ge_cells[cell][index], geRange), height = 18 + 130 * value;
        bar.setAttribute('y', 252 - height); bar.setAttribute('height', height); return value;
      });
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const x = 78 + phase * 812, y = 240 - mean * 120;
      if (points.length && x < Number(points.at(-1).split(',')[0])) points.length = 0;
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`); if (points.length > 90) points.shift();
      trace.setAttribute('d', `M${points.join(' L')}`); dot.setAttribute('cx', x); dot.setAttribute('cy', y);
      label.textContent = `mean ${data.ge[index].toFixed(3)} µS`;
    });
  }
})();
