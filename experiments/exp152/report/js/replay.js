// Select completed samples only; never interpolate across callback delays.
export function sampleIndex(times, time) {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (times[mid] <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

// Independent recorded events keep travelling when later spikes arrive.
export function pulseSamples(times, time, travelMs = 30, arrivalMs = 8) {
  const samples = [];
  for (let i = sampleIndex(times, time - travelMs - arrivalMs) + 1;
       i < times.length && times[i] <= time; i++) {
    const age = time - times[i];
    samples.push({ progress: Math.min(1, age / travelMs),
      opacity: age <= travelMs ? 1 : 1 - (age - travelMs) / arrivalMs });
  }
  return samples;
}

export function routePoint(route, progress) {
  const t = Math.max(0, Math.min(1, progress)), u = 1 - t;
  if (route.length === 2) return route[0].map((v, axis) => v * u + route[1][axis] * t);
  return route[0].map((v, axis) => u ** 3 * v + 3 * u * u * t * route[1][axis]
    + 3 * u * t * t * route[2][axis] + t ** 3 * route[3][axis]);
}

export function loopTime(time, elapsed, speed, duration) {
  return (time + Math.max(0, elapsed) * speed) % duration;
}

export const toDegrees = radians => radians * 180 / Math.PI;
export const voltageFraction = (voltage, minimum = -80, maximum = 40) => Number.isFinite(voltage)
  ? Math.max(0, Math.min(1, (voltage - minimum) / (maximum - minimum))) : 0;

export function peerInhibitionRoutes() {
  return [
    { name: "flexor", pop: 0, inhibitory: true, peer: true,
      points: [[236, 210], [420, 35], [820, 35], [1004, 210]] },
    { name: "extensor", pop: 0, inhibitory: true, peer: true,
      points: [[1004, 220], [820, 124], [420, 124], [236, 220]] },
  ];
}

export function controllerPopulations(data) {
  if (data.schema !== "pinglab.exp152.replay/v2" || data.neuron_model !== "elevate"
      || data.controller_populations?.length !== 1 || data.controller_populations[0] !== "motor"
      || data.neurons_per_population !== 1
      || data.mechanics?.angle_convention !== "zero_downward_positive_right")
    throw new Error("Replay requires the single-cell Elevate loaded-joint recording.");
  return data.controller_populations;
}

export function sensoryExcitationRoutes() {
  return [
    {name: "flexor", pop: 0, sensory: true, inhibitory: false,
      points: [[465, 412], [335, 490], [265, 485], [225, 270]]},
    {name: "extensor", pop: 0, sensory: true, inhibitory: false,
      points: [[775, 412], [905, 490], [975, 485], [1015, 270]]},
  ];
}

export function arrowHead(points) {
  const tip = routePoint(points, 1), before = routePoint(points, .99);
  const angle = Math.atan2(tip[1] - before[1], tip[0] - before[0]);
  const base = [tip[0] - 11 * Math.cos(angle), tip[1] - 11 * Math.sin(angle)];
  const across = [-5 * Math.sin(angle), 5 * Math.cos(angle)];
  return [tip[0], tip[1], base[0] + across[0], base[1] + across[1],
    base[0] - across[0], base[1] - across[1]];
}

// Replay the report-selected immutable Pingstore payload.
export async function mountReplay(host) {
  const $ = id => host.querySelector(`[data-id="${id}"]`);
  const apps = [], observers = [], removers = [];
  let frameId;
  function listen(target, type, listener) {
    target.addEventListener(type, listener);
    removers.push(() => target.removeEventListener(type, listener));
  }
  const names = ["flexor", "extensor", "plant"];
  const colors = { flexor: 0x176d91, extensor: 0xbc3866, plant: 0x348577, ink: 0x263343, muted: 0x687787, grid: 0xdce3e8 };
  const hex = (color) => `#${color.toString(16).padStart(6, "0")}`;
  const title = (name) => name === "plant" ? "Joint Dynamics" : name[0].toUpperCase() + name.slice(1);
  const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "—";
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const upper = (array, value) => {
    let lo = 0, hi = array.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (array[mid] <= value) lo = mid + 1; else hi = mid; }
    return lo;
  };
  const last = (array) => array.length ? array[array.length - 1] : 0;
  let time = 0, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  try {
    const [{ Application, Graphics }, response] = await Promise.all([
      import('pixi.js'), fetch(host.dataset.source),
    ]);
    if (!response.ok) throw new Error(`Replay HTTP ${response.status}`);
    const data = await response.json();
    const populations = controllerPopulations(data);
    // Use one fixed scale for both muscles throughout the complete recording.
    const peakActivation = data.nodes.plant.activation.reduce(
      (peak, row) => Math.max(peak, ...row), 0);
    const activationScale = Math.max(.01, Math.ceil(peakActivation * 100) / 100);
    const period = data.period_ms;
    const endTime = Math.max(data.duration_ms, ...names.map((name) => last(data.nodes[name].time_ms)));
    for (const name of ["flexor", "extensor"]) {
      const node = data.nodes[name];
      node.spikeTimes = populations.map(() => []);
      node.spikes.forEach((row, index) => row.forEach((count, pop) => {
        if (count > 0) node.spikeTimes[pop].push(node.time_ms[index]);
      }));
    }
    $("scrub").max = String(endTime);

    async function surface(id, height, originY = 0) {
      const host = $(id), app = new Application();
      apps.push(app);
      await app.init({ width: 1240, height, backgroundAlpha: 0, antialias: true, resolution: Math.min(devicePixelRatio || 1, 2), autoDensity: true, autoStart: false, preference: "webgl" });
      app.stage.position.y = -originY;
      app.canvas.style.width = "100%";
      app.canvas.style.height = "100%";
      host.appendChild(app.canvas);
      const labels = document.createElement("div"); labels.className = "labels"; host.appendChild(labels);
      const observer = new ResizeObserver(() => host.style.setProperty("--scale", host.clientWidth / 1240)); observer.observe(host); observers.push(observer);
      function label(text, x, y, size = 14, color = colors.ink, anchor = "left") {
        const el = document.createElement("span"); el.className = "scene-label"; el.textContent = text;
        Object.assign(el.style, { left: `${x / 1240 * 100}%`, top: `${(y - originY) / height * 100}%`, fontSize: `clamp(8px, calc(var(--scale,1)*${size}px), ${size}px)`, color: hex(color), transform: anchor === "center" ? "translateX(-50%)" : anchor === "right" ? "translateX(-100%)" : "" });
        labels.appendChild(el); return el;
      }
      return { host, app, label };
    }
    const circuitHeight = 560;
    const circuit = await surface("circuit", circuitHeight, 0);
    circuit.host.style.aspectRatio = `1240 / ${circuitHeight}`;
    const graphic = new Graphics(); circuit.app.stage.addChild(graphic);
    circuit.label("FLEXOR", 54, 113, 14, colors.flexor);
    circuit.label("EXTENSOR", 1186, 113, 14, colors.extensor, "right");
    circuit.label("Flexor → extensor inhibition", 620, 43, 12, colors.muted, "center");
    circuit.label("Extensor → flexor inhibition", 620, 128, 12, colors.muted, "center");
    const sides = [
      { name: "flexor", motor: [205, 240], anchor: [465, 300], frame: 24 },
      { name: "extensor", motor: [1035, 240], anchor: [775, 300], frame: 840 },
    ];
    const routes = [...peerInhibitionRoutes(), ...sensoryExcitationRoutes()];
    for (const side of sides) {
      const color = colors[side.name], x = side.motor[0];
      circuit.label("startup E", x + (side.name === "flexor" ? -100 : 100), 218, 11, colors.muted, "center");
      circuit.label("Elevate", x, 195, 14, color, "center");
      circuit.label("one neuron", x + (side.name === "flexor" ? -70 : 70), 282, 12, colors.muted, "center");
      circuit.label("E · sensory feedback", side.name === "flexor" ? 330 : 910, 495, 12, color, "center");
      circuit.label("E · muscle activation", side.name === "flexor" ? 320 : 920, 235, 11, color, "center");
      circuit.label(side.name === "flexor" ? "+ torque" : "− torque", side.anchor[0], 272, 12, color, "center");
      const scaleX = side.anchor[0] + (side.name === "flexor" ? 22 : -22);
      const scaleAnchor = side.name === "flexor" ? "left" : "right";
      circuit.label(`${(100 * activationScale).toFixed(0)}%`, scaleX, 317, 11, color, scaleAnchor);
      circuit.label("0%", scaleX, 393, 11, color, scaleAnchor);
      const exit = [x + (side.name === "flexor" ? 35 : -35), 245];
      const distance = Math.hypot(exit[0] - side.anchor[0], exit[1] - side.anchor[1]);
      const approach = side.anchor.map((value, axis) => value + (exit[axis] - value) * 14 / distance);
      routes.push({name: side.name, pop: 0, points: [exit, approach], inhibitory: false});
    }
    const angleLabel = circuit.label("Awaiting joint", 620, 446, 22, colors.ink, "center");
    circuit.label(`JOINT · ${fmt(data.mechanics.tip_mass_kg * 1000, 0)} g at ${fmt(data.mechanics.arm_length_m * 100, 0)} cm`, 620, 196, 13, colors.muted, "center");
    circuit.label("0°", 550, 493, 11, colors.muted, "center");
    const velocityLabel = circuit.label("", 620, 478, 13, colors.muted, "center");
    const line = (g, a, b, color, width = 1.5, alpha = 1) =>
      g.moveTo(...a).lineTo(...b).stroke({ color, width, alpha });
    function terminal(g, route, color) {
      g.poly(arrowHead(route.points)).fill(color);
    }

    function cell(point, voltage, color) {
      const [x, y] = point, radius = 27;
      const fraction = voltageFraction(voltage, ...data.voltage_range_mv);
      graphic.circle(x, y, radius).fill(0xffffff);
      if (fraction > 0) {
        // Fill the circular segment below the voltage level, rather than
        // changing opacity or adding decorative intracellular structures.
        const angle = Math.asin(1 - 2 * fraction);
        const halfWidth = radius * Math.cos(angle), level = y + radius * Math.sin(angle);
        graphic.moveTo(x - halfWidth, level).lineTo(x + halfWidth, level)
          .arc(x, y, radius, angle, Math.PI - angle).closePath()
          .fill({color, alpha: .5});
      }
      graphic.circle(x, y, radius).stroke({color, width: 1.5});
    }
    function drawCircuit() {
      graphic.clear();
      for (const side of sides) {
        const color = colors[side.name];
        graphic.roundRect(side.frame, 98, 376, 430, 12)
          .fill({ color, alpha: .018 }).stroke({ color: colors.grid, width: 1 });
        const inputSign = side.name === "flexor" ? -1 : 1;
        const tonic = {points: [[side.motor[0] + inputSign * 145, 240], [side.motor[0] + inputSign * 35, 240]], inhibitory: false};
        line(graphic, ...tonic.points, color, 1.3, .7); terminal(graphic, tonic, color);
      }
      graphic.roundRect(440, 180, 360, 340, 12)
        .fill({color: 0xffffff, alpha: 1}).stroke({color: colors.grid, width: 1});
      // Paths and particles share exactly the same coordinates and endpoints.
      for (const route of routes) {
        const color = route.peer ? colors.muted : colors[route.name];
        const points = route.points;
        graphic.moveTo(...points[0]);
        if (points.length === 4) graphic.bezierCurveTo(...points[1], ...points[2], ...points[3]);
        else graphic.lineTo(...points[1]);
        graphic.stroke({color, width: route.peer ? 1.2 : 1.5, alpha: .55});
      }
      for (const side of sides) {
        const node = data.nodes[side.name], index = sampleIndex(node.time_ms, time);
        cell(side.motor, index >= 0 ? node.voltage[index][0] : null, colors[side.name]);
      }
      // Render ALL in-flight events. New spikes never reset earlier particles.
      // Opacity remains full until arrival, followed by an 8 ms endpoint fade.
      for (const route of routes) {
        const color = colors[route.name];
        for (const pulse of pulseSamples(route.sensory ? data.sensory_time_ms[route.name] : data.nodes[route.name].spikeTimes[route.pop], time)) {
          const point = routePoint(route.points, pulse.progress);
          graphic.circle(...point, 7).fill({color, alpha: pulse.opacity * .14});
          graphic.circle(...point, 3).fill({color, alpha: pulse.opacity});
          graphic.circle(...point, 1).fill({color: 0xffffff, alpha: pulse.opacity});
        }
      }
      const plant = data.nodes.plant, index = sampleIndex(plant.time_ms, time);
      const angle = index >= 0 ? plant.angle[index] : data.mechanics.initial_angle;
      const activation = index >= 0 ? plant.activation[index] : [0, 0];
      const pivot = [550, 340], length = 130;
      const tip = [pivot[0] + Math.sin(angle) * length,
        pivot[1] + Math.cos(angle) * length];
      for (const [sideIndex, side] of sides.entries()) {
        const width = 24, height = 76, x = side.anchor[0] - width / 2, y = 322;
        graphic.roundRect(x, y, width, height, 6).fill({color: colors[side.name], alpha: .12});
        const filled = height * clamp(activation[sideIndex] / activationScale, 0, 1);
        if (filled > 0) graphic.rect(x, y + height - filled, width, filled).fill(colors[side.name]);
        for (const tickY of [y, y + height])
          line(graphic, [x - 3, tickY], [x + width + 3, tickY], colors[side.name], 1, .6);
        graphic.circle(...side.anchor, 6).fill(colors[side.name]);
      }
      // Keep the fixed downward reference clear of the moving arm.
      for (let y = pivot[1] + 23; y < 484; y += 12)
        line(graphic, [pivot[0], y], [pivot[0], y + 5], colors.muted, 1, .45);
      line(graphic, pivot, tip, colors.ink, 10);
      graphic.circle(...tip, data.mechanics.tip_mass_kg > 0 ? 13 : 7).fill(colors.ink);
      graphic.circle(...pivot, 16).fill(0xffffff).stroke({color: colors.ink, width: 2});
      graphic.circle(...pivot, 6).fill(colors.ink);
      angleLabel.textContent = `${fmt(toDegrees(angle), 1)}°${index < 0 ? " · initial" : ""}`;
      velocityLabel.textContent = index < 0 ? "" : `${fmt(toDegrees(plant.velocity[index]), 1)}°/s` +
        (time > last(plant.time_ms) + period ? " · recording ended" : "");
      // Draw terminals last so neuron fills, muscle strokes and arriving
      // particles cannot obscure the filled arrowheads.
      for (const route of routes) terminal(graphic, route,
        route.peer ? colors.muted : colors[route.name]);
      circuit.app.render();
    }

    const yMax = period * 1.2;
    const chartStart = 310, chartWidth = 894, laneY = [48, 118, 188], laneHeight = 45;
    async function chart(id) {
      const s = await surface(id, 240, 24), graph = new Graphics(), overlay = new Graphics(); s.app.stage.addChild(graph);
      const labels = [];
      for (let n = 0; n < 3; n++) {
        s.label(title(names[n]), 26, laneY[n] + 15, 13, colors[names[n]]);
        s.label(`${Number(period.toFixed(2))} ms`, chartStart - 10, laneY[n] + laneHeight * (1 - period / yMax) - 5, 10, colors.muted, "right");
        s.label("0", chartStart - 10, laneY[n] + laneHeight - 7, 10, colors.muted, "right");
      }
      for (let i = 0; i <= 4; i++) labels.push(s.label("", chartStart + chartWidth * i / 4, 232, 11, colors.muted, i === 4 ? "right" : i === 0 ? "left" : "center"));
      const traces = names.map((_, index) => {
        const trace = new Graphics(), mask = new Graphics();
        mask.rect(chartStart, laneY[index], chartWidth, laneHeight).fill(0xffffff);
        s.app.stage.addChild(trace, mask);
        trace.mask = mask;
        return trace;
      });
      s.app.stage.addChild(overlay);
      return { ...s, graph, overlay, traces, tickLabels: labels };
    }
    const overview = await chart("overview");
    function drawChart(chart, start, end) {
      const g = chart.graph; g.clear();
      const x = (t) => chartStart + (t - start) / (end - start) * chartWidth;
      for (let n = 0; n < 3; n++) {
        const node = data.nodes[names[n]], y = laneY[n];
        const trace = chart.traces[n]; trace.clear();
        const stopped = last(node.time_ms);
        if (stopped < end) g.rect(x(Math.max(start, stopped)), y, x(end) - x(Math.max(start, stopped)), laneHeight).fill({ color: colors.grid, alpha: .27 });
        for (let i = 0; i <= 4; i++) line(g, [chartStart + i * chartWidth / 4, y], [chartStart + i * chartWidth / 4, y + laneHeight], colors.grid, 1, .6);
        line(g, [chartStart, y + laneHeight], [chartStart + chartWidth, y + laneHeight], colors.grid);
        const budgetY = y + laneHeight * (1 - period / yMax);
        for (let xx = chartStart; xx < chartStart + chartWidth; xx += 10) line(g, [xx, budgetY], [Math.min(xx + 5, chartStart + chartWidth), budgetY], colors.muted, 1, .6);
        const a = Math.max(0, upper(node.time_ms, start) - 1), b = Math.min(node.time_ms.length, upper(node.time_ms, end) + 1);
        let begun = false;
        for (let i = a; i < b; i++) {
          if (node.time_ms[i] < start || node.time_ms[i] > end) continue;
          const xx = x(node.time_ms[i]), yy = y + laneHeight * (1 - node.callback_ms[i] / yMax);
          if (!begun) { trace.moveTo(xx, yy); begun = true; } else trace.lineTo(xx, yy);
        }
        if (begun) trace.stroke({ color: colors[names[n]], width: 1.2, alpha: .8 });

      }
      chart.tickLabels.forEach((el, i) => el.textContent = `${fmt((start + (end - start) * i / 4) / 1000)} s`);
    }
    drawChart(overview, 0, endTime);
    const voltageRows = ["flexor", "extensor"].map(side => ({side, pop: 0, name: "Elevate"}));
    const dynamicsHeight = 142 + voltageRows.length * 78 + 42;
    const dynamics = await surface("dynamics", dynamicsHeight);
    dynamics.host.style.aspectRatio = `1240 / ${dynamicsHeight}`;
    const angleGraph = new Graphics(), angleOverlay = new Graphics();
    dynamics.app.stage.addChild(angleGraph, angleOverlay);
    dynamics.overlay = angleOverlay;
    const plant = data.nodes.plant;
    const angles = plant.angle.map(toDegrees);
    const angleMin = Math.floor(angles.reduce((low, value) => Math.min(low, value), toDegrees(data.mechanics.initial_angle)) / 10) * 10;
    const angleMax = Math.max(angleMin + 10, Math.ceil(angles.reduce((high, value) => Math.max(high, value), toDegrees(data.mechanics.initial_angle)) / 10) * 10);
    const angleTop = 24, angleHeight = 86;
    const angleX = value => chartStart + value / endTime * chartWidth;
    const angleY = value => angleTop + (angleMax - value) / (angleMax - angleMin) * angleHeight;
    dynamics.label("Angle", 26, 60, 13, colors.plant);
    for (const value of [angleMax, (angleMin + angleMax) / 2, angleMin]) {
      dynamics.label(`${value}°`, chartStart - 10, angleY(value) - 5, 10, colors.muted, "right");
      line(angleGraph, [chartStart, angleY(value)], [chartStart + chartWidth, angleY(value)], colors.grid, 1, .6);
    }
    for (let i = 0; i <= 4; i++) {
      const x = chartStart + i * chartWidth / 4;
      line(angleGraph, [x, angleTop], [x, angleTop + angleHeight], colors.grid, 1, .6);
      dynamics.label(`${fmt(endTime * i / 4 / 1000)} s`, x, dynamicsHeight - 40, 11, colors.muted,
        i === 4 ? "right" : i === 0 ? "left" : "center");
    }
    // Hold each recorded angle until the next plant callback completes.
    const angleTrace = new Graphics();
    dynamics.app.stage.addChildAt(angleTrace, 1);
    angles.forEach((value, i) => {
      const x = angleX(plant.time_ms[i]), y = angleY(value);
      if (!i) angleTrace.moveTo(x, y);
      else angleTrace.lineTo(x, angleY(angles[i - 1])).lineTo(x, y);
    });
    if (angles.length) angleTrace.stroke({color: colors.plant, width: 1.2, alpha: .8});

    // One shared voltage scale preserves amplitude comparisons between cells.
    let voltageMin = -80, voltageMax = 40;
    for (const side of ["flexor", "extensor"]) {
      for (const row of data.nodes[side].voltage) {
        for (const value of row) {
          voltageMin = Math.min(voltageMin, value);
          voltageMax = Math.max(voltageMax, value);
        }
      }
    }
    voltageMin = Math.floor(voltageMin / 20) * 20;
    voltageMax = Math.ceil(voltageMax / 20) * 20;
    const voltageTop = 142, voltageStride = 78, voltageHeight = 58;
    const dynamicsBottom = voltageTop + (voltageRows.length - 1) * voltageStride + voltageHeight;
    voltageRows.forEach(({side, pop, name}, index) => {
      const node = data.nodes[side], top = voltageTop + index * voltageStride;
      const voltageY = value => top + (voltageMax - value) / (voltageMax - voltageMin) * voltageHeight;
      dynamics.label(`${title(side)} ${name} (mV)`, 26, top + 24, 13, colors[side]);
      for (const value of [voltageMax, 0, voltageMin]) {
        line(angleGraph, [chartStart, voltageY(value)],
          [chartStart + chartWidth, voltageY(value)], colors.grid, 1, .6);
      }
      for (let i = 0; i <= 4; i++) {
        const x = chartStart + i * chartWidth / 4;
        line(angleGraph, [x, top], [x, top + voltageHeight], colors.grid, 1, .6);
      }
      const trace = new Graphics();
      dynamics.app.stage.addChildAt(trace, dynamics.app.stage.children.length - 1);
      // Preserve each callback's timestamp and hold voltage until the next sample.
      node.voltage.forEach((row, i) => {
        const x = angleX(node.time_ms[i]), y = voltageY(row[pop]);
        if (!i) trace.moveTo(x, y);
        else trace.lineTo(x, voltageY(node.voltage[i - 1][pop])).lineTo(x, y);
      });
      if (node.voltage.length) trace.stroke({color: colors[side], width: 1.2, alpha: .8});
    });

    function cursor(chart, start, end, top = 33, bottom = 235) {
      const g = chart.overlay; g.clear();
      const x = (t) => chartStart + (t - start) / (end - start) * chartWidth;
      line(g, [x(time), top], [x(time), bottom], colors.ink, 1.5, .85);
      chart.app.render();
    }

    function render() {
      drawCircuit(); cursor(overview, 0, endTime);
      cursor(dynamics, 0, endTime, angleTop, dynamicsBottom);
      $("scrub").value = String(time);
      $("scrub").setAttribute("aria-valuetext", `${fmt(time / 1000)} seconds`);
      $("play").textContent = playing ? "Pause" : time >= endTime ? "Replay" : "Play";
    }
    function seek(value) { time = clamp(value, 0, endTime); playing = false; render(); }
    listen($("play"), "click", () => { playing = !playing; render(); });
    listen($("restart"), "click", () => seek(0));
    listen($("scrub"), "input", event => seek(Number(event.target.value)));
    function chartClick(event) {
      const rect = event.currentTarget.getBoundingClientRect(), xx = (event.clientX - rect.left) / rect.width * 1240;
      if (xx < chartStart || xx > chartStart + chartWidth) return;
      const t = (xx - chartStart) / chartWidth * endTime;
      seek(t);
    }
    listen(overview.host, "click", chartClick);
    listen(dynamics.host, "click", chartClick);
    for (const id of ["play", "restart", "scrub", "speed"]) $(id).disabled = false;
    render();
    let previousFrame = performance.now();
    function frame(now) {
      const elapsed = Math.min(now - previousFrame, 100); previousFrame = now;
      if (playing && !document.hidden) { time = loopTime(time, elapsed, Number($("speed").value), endTime); render(); }
      frameId = requestAnimationFrame(frame);
    }
    frameId = requestAnimationFrame(frame);
    listen(document, "visibilitychange", () => { previousFrame = performance.now(); });
  } catch (error) {
    $("error").hidden = false; $("error").textContent = error.message;
    console.error(error);
  }

  return () => {
    cancelAnimationFrame(frameId);
    for (const remove of removers) remove();
    for (const observer of observers) observer.disconnect();
    for (const app of apps) app.destroy(true, { children: true });
    host.querySelectorAll('.labels').forEach(node => node.remove());
  };
}
