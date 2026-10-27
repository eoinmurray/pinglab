/** Replay native rate recordings. Moving strokes are continuous flow, not spikes. */
export async function mountLifecycle(host) {
  const [{ Application, Graphics }, response] = await Promise.all([import('pixi.js'), fetch(host.dataset.source)]);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  validateRecording(data);
  const get = id => host.querySelector(`[data-id="${id}"]`);
  const scene = get('scene'), app = new Application();
  await app.init({ width:760, height:320, backgroundAlpha:0, antialias:true,
    resolution:Math.min(devicePixelRatio || 1, 2), autoDensity:true, autoStart:false });
  scene.appendChild(app.canvas);
  const background = new Graphics(), g = new Graphics(), foreground = new Graphics();
  app.stage.addChild(background, g, foreground);
  const labels = document.createElement('div'); labels.className = 'scene-labels'; scene.appendChild(labels);
  const abort = new AbortController(), signal = abort.signal;
  let disposed = false, visible = true, position = 0, previousTime = null, frame = 0;
  let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let width = 760, height = 320, text = {}, points = [], geometry = {};
  const ink = 0x263343, muted = 0x687787, line = 0xdce3e8, blue = 0x176d91;
  const purple = 0x94658a, green = 0x27866a, red = 0xbc3b43, amber = 0xc58a24;
  const arrivalPhase = .30, processingEnd = .62, decisionPhase = .82;
  const readoutStart = arrivalPhase + .75*(processingEnd-arrivalPhase);
  let maxEnergy = data.initial_energy, energyRanges = [];
  const condition = data.conditions[0], network = data.network;
  const outputRange = Math.max(1,Math.ceil(Math.max(...condition.events.filter(e=>e.phase!=='prenatal').map(e=>Math.abs(e.prediction)))));
  const pathways = new Map(), flowSelections = new WeakMap();
  for (const edge of network.edges) {
    if (!pathways.has(edge.pathway)) pathways.set(edge.pathway, []);
    pathways.get(edge.pathway).push(edge);
  }
  // Reconstruct the same scale after any seek, without using future reserves.
  const birthRange = 2 ** Math.ceil(Math.log2(data.initial_energy));
  let range = birthRange;
  energyRanges = condition.events.map(e => {
    if (e.birth || e.conception) range = birthRange;
    if (e.phase === 'prenatal') return {before:range, after:range};
    while (e.energy_before > range) range *= 2;
    const before = range;
    const incoming = e.energy_before + (e.accepted ? Math.max(0, e.delta) : 0);
    while (incoming > range) range *= 2;
    return {before, after:range};
  });
  get('scrub').max = String(condition.events.length);

  function label(name, x, y, size = 18, color = muted, centered = false) {
    const node = document.createElement('span'); node.className = 'scene-label';
    Object.assign(node.style, {left:`${x / width * 100}%`, top:`${y / height * 100}%`,
      fontSize:`${size * scene.clientWidth / width}px`, color:`#${color.toString(16).padStart(6,'0')}`,
      transform:centered ? 'translateX(-50%)' : ''});
    labels.append(node); text[name] = node; return node;
  }
  function layout() {
    width = scene.clientWidth < 700 ? 700 : 760;
    scene.style.aspectRatio = `${width} / ${height}`;
    app.renderer.resize(width, height);
    // Pixi's autoDensity resize writes fixed inline CSS dimensions. Override
    // them after each resize so the full scene fits the article container.
    app.canvas.style.width = '100%';
    app.canvas.style.height = '100%';
    labels.replaceChildren(); text = {};
    const ax = width/2, cy = 134, spacing = 42, offset = 113, horizontalOffset = 133;
    const inputX = ax-horizontalOffset, foodStop = inputX-42, foodStart = 20;
    const outputX = width*.8, energyX = width-48;
    geometry = {ax, cy, spacing, inputX, foodStart, foodStop, outputX, energyX, collectorX:ax+horizontalOffset, barTop:cy-90, barBottom:cy+90, barHeight:180};
    points = [];
    for (let i = 0; i < 16; i++) points.push({x:ax+(i%4-1.5)*spacing, y:cy+(Math.floor(i/4)-1.5)*spacing});
    for (let i = 0; i < 4; i++) points.push({x:ax+(i%2-.5)*spacing, y:cy+offset+Math.floor(i/2)*spacing});
    for (let i = 0; i < 6; i++) points.push({x:inputX, y:cy+(i%3-1)*spacing});
    label('agent',ax,16,22,ink,true);
    label('inputs',inputX,cy-spacing-38,17,muted,true).textContent = 'Inputs';
    label('energy',energyX,12,17,muted,true);
    label('energyMax',energyX+22,geometry.barTop-6,13,muted,true);
    label('energyZero',energyX+22,geometry.barBottom-6,13,muted,true).textContent = '0';
    label('energyValue',energyX,geometry.barBottom+14,17,ink,true);
    label('outputTitle',outputX,12,17,muted,true).textContent = 'Predicted ΔE';
    label('outputLow',outputX+25,geometry.barBottom-6,13,muted,true).textContent = `−${outputRange}`;
    label('outputHigh',outputX+25,geometry.barTop-6,13,muted,true).textContent = `+${outputRange}`;
    label('outputZero',outputX+25,cy-6,13,muted,true).textContent = '0';
    label('outputValue',outputX,geometry.barBottom+14,17,ink,true);
    label('outputChoice',outputX,geometry.barBottom+38,16,ink,true);
    label('exploration',outputX,geometry.barBottom+62,13,muted,true);
    draw();
  }
  function drawEnergyBand(x, low, high, color) {
    const bottom = Math.max(0, Math.min(maxEnergy,low)), top = Math.max(0, Math.min(maxEnergy,high));
    const {barBottom,barHeight} = geometry;
    if (top > bottom) g.rect(x-7,barBottom-top/maxEnergy*barHeight,14,(top-bottom)/maxEnergy*barHeight).fill(color);
  }
  function drawEnergy(x,e,energy,revealed) {
    const {barTop,barHeight} = geometry;
    g.roundRect(x-7,barTop,14,barHeight,4).fill(0xedf0f3);
    if (e.phase === 'prenatal') {
      g.roundRect(x-7,barTop,14,barHeight,4).stroke({color:line,width:1});
      return;
    }
    drawEnergyBand(x,0,energy,blue);
    if (revealed) {
      const gain = e.accepted ? Math.max(0,e.delta) : 0, poison = e.accepted ? Math.max(0,-e.delta) : 0;
      drawEnergyBand(x,energy-gain,energy,green);
      drawEnergyBand(x,energy,energy+poison,red);
      drawEnergyBand(x,energy+poison,energy+poison+data.metabolic_cost,amber);
    }
    g.roundRect(x-7,barTop,14,barHeight,4).stroke({color:line,width:1});
  }
  const edgeColor = edge => edge.source >= 20 ? muted : edge.source < 16 ? blue : purple;
  function movingStroke(a,b,progress,color,width,alpha) {
    // Let the head reach the target before the tail follows it into the same
    // centre. Clipping both ends makes the stroke grow out of the source and
    // shrink into the target, with no mid-path wrap or unfinished tail.
    const length = .2, head = progress*(1+length);
    const start = Math.max(0,Math.min(1,head-length)), end = Math.max(0,Math.min(1,head));
    if (end <= start) return;
    foreground.moveTo(a.x+(b.x-a.x)*start,a.y+(b.y-a.y)*start)
      .lineTo(a.x+(b.x-a.x)*end,a.y+(b.y-a.y)*end)
      .stroke({color,width,alpha});
  }
  function selectedFlows(e) {
    if (flowSelections.has(e)) return flowSelections.get(e);
    const peaks = Array.from({length:20},(_,i)=>Math.max(...e.rates.slice(0,-1).map(row=>row[i])));
    const inputs = Array.from({length:6},(_,i)=>Math.max(...e.drives.map(row=>row[i]))), selected = [];
    for (const edges of pathways.values()) {
      const byTarget = new Map();
      for (const edge of edges) {
        const source = edge.source >= 20 ? inputs[edge.source-20] : peaks[edge.source];
        const weight = edge.plastic_path < 0 ? edge.weight : e.weights[edge.plastic_path][edge.parameter_index];
        if (!byTarget.has(edge.target)) byTarget.set(edge.target,[]);
        byTarget.get(edge.target).push({edge,weight,peak:source*weight});
      }
      for (const values of byTarget.values()) selected.push(...values.sort((a,b)=>b.peak-a.peak).slice(0,2));
    }
    flowSelections.set(e,selected);
    return selected;
  }
  function drawConnections(e,step,dead,progress) {
    for (const edges of pathways.values()) {
      const inputSource = edges[0].source;
      if (inputSource>=20 && (inputSource>=23)!==(e.phase==='prenatal')) continue;
      for (const edge of edges) {
        const a = points[edge.source], b = points[edge.target];
        background.moveTo(a.x,a.y).lineTo(b.x,b.y);
      }
      background.stroke({color:edgeColor(edges[0]),width:.65,alpha:dead ? .025 : .065});
    }
    if (step === 0 || dead) return;
    const previous = e.rates[step-1], inputs = e.drives[step];
    // Keep selected paths fixed for this encounter so changing rate rankings
    // cannot remove a travelling stroke before it reaches its destination.
    for (const {edge,weight,peak} of selectedFlows(e)) {
      const scale = Math.max(network.flow_scales[edge.pathway],1e-12);
      if (peak/scale < .002) continue;
      const source = edge.source >= 20 ? inputs[edge.source-20] : previous[edge.source];
      const strength = Math.min(1,source*weight/scale);
      // Travel is a display convention for continuous flow, not a spike or
      // physical delay. Every selected stroke completes before readout begins.
      movingStroke(points[edge.source],points[edge.target],progress,edgeColor(edge),1.4,.15+.65*strength);
    }
  }
  function drawNetwork(e,step,dead,born,phase,arrived) {
    const {ax,cy} = geometry, rates = e.rates[step];
    if (born) g.roundRect(ax-80,cy-87,160,265,14).stroke({color:blue,width:2,alpha:Math.max(0,1-phase*4)});
    for (let i = 0; i < 20; i++) {
      const p = points[i], rate = rates[i], color = dead ? 0xabb3bb : i < 16 ? blue : purple;
      g.circle(p.x,p.y,14).fill(0xffffff);
      g.circle(p.x,p.y,14).fill({color,alpha:dead ? .12 : .06+.9*rate}).stroke({color,width:1.3});
    }
    for (let i = 0; i < 3; i++) {
      const offset = e.phase==='prenatal' ? 23 : 20;
      const p = points[offset+i], value = arrived ? e.drives[step][offset-20+i] : 0;
      g.circle(p.x,p.y,8).fill(0xffffff);
      g.circle(p.x,p.y,8).fill({color:muted,alpha:.08+.55*value}).stroke({color:muted,width:1.2});
    }
    if (dead) g.moveTo(ax-65,cy-63).lineTo(ax+65,cy+169).moveTo(ax+65,cy-63).lineTo(ax-65,cy+169).stroke({color:red,width:3});
  }
  function drawObject(x,y,radius,revealed,delta) {
    const color = revealed ? (delta > 0 ? green : red) : blue;
    g.circle(x,y,radius).fill(color);
  }
  function readoutStroke(a,b,progress,color) {
    background.moveTo(a.x,a.y).lineTo(b.x,b.y).stroke({color,width:1,alpha:.12});
    movingStroke(a,b,progress,color,2.5,.9);
  }
  function drawReadout(phase) {
    const {collectorX,outputX,cy,spacing} = geometry;
    const active = phase >= readoutStart;
    const progress = Math.max(0,Math.min(1,(phase-processingEnd)/(decisionPhase-processingEnd)));
    const gather = Math.min(1,progress*2), project = Math.max(0,progress*2-1);
    for (let group = 0; group < 2; group++) {
      const color = group === 0 ? green : red, collector = {x:collectorX,y:cy+(group-.5)*spacing};
      if (active) {
        // These routes illustrate the fixed signed averaging operation, rather
        // than adding synapses or changing the native controller.
        for (let i = group*8; i < group*8+8; i++) readoutStroke(points[i],collector,gather,color);
        readoutStroke(collector,{x:outputX,y:cy},project,color);
      }
      g.circle(collector.x,collector.y,14).fill(0xffffff);
      g.circle(collector.x,collector.y,14).fill({color,alpha:active ? .2 : .05})
        .stroke({color,width:1.5,alpha:active ? 1 : .25});
    }
  }
  function drawOutput(e,revealed,phase,step) {
    const {outputX:x, cy:center, barTop, barBottom, barHeight} = geometry, half = barHeight/2;
    const prenatal = e.phase==='prenatal';
    text.outputTitle.textContent = prenatal ? 'Activity' : 'Predicted ΔE';
    text.outputHigh.textContent = prenatal ? '1' : `+${outputRange}`;
    text.outputLow.textContent = prenatal ? '0' : `−${outputRange}`;
    text.outputZero.style.visibility = prenatal ? 'hidden' : 'visible';
    g.roundRect(x-7,barTop,14,barHeight,4).fill(0xedf0f3);
    if (prenatal) {
      const rates = e.rates[step];
      const eMean = rates.slice(0,16).reduce((a,b)=>a+b,0)/16;
      const iMean = rates.slice(16).reduce((a,b)=>a+b,0)/4;
      g.rect(x-7,barBottom-eMean*barHeight,7,eMean*barHeight).fill(blue);
      g.rect(x,barBottom-iMean*barHeight,7,iMean*barHeight).fill(purple);
      const target = data.genome.homeostatic_target_rate;
      const y = barBottom-target*barHeight;
      g.moveTo(x-12,y).lineTo(x+12,y).stroke({color:ink,width:1});
      text.outputValue.textContent = phase>=processingEnd ? `L ${e.loss.toFixed(3)}` : '…';
      text.outputChoice.textContent = phase>=processingEnd ? 'Homeostatic update' : `Target ${target}`;
      text.exploration.textContent = `η ${e.learning_rate}`;
    } else {
      g.rect(x-7,center-half,14,half).fill({color:green,alpha:.08});
      g.rect(x-7,center,14,half).fill({color:red,alpha:.08});
      if (revealed) {
        const y = center-half*e.prediction/outputRange;
        if (Math.abs(y-center)>1e-9) g.rect(x-7,Math.min(y,center),14,Math.abs(y-center)).fill(e.prediction>0 ? green : red);
        g.moveTo(x-11,y).lineTo(x+11,y).stroke({color:ink,width:2});
      }
      g.moveTo(x-12,center).lineTo(x+12,center).stroke({color:muted,width:1});
      text.outputValue.textContent = revealed ? signed(e.prediction) : '…';
      text.outputChoice.textContent = revealed ? (e.accepted ? 'Consume' : 'Reject') : phase>=processingEnd ? 'Readout' : phase>=arrivalPhase ? 'Processing' : 'Awaiting';
      text.exploration.textContent = revealed && e.explored ? 'Exploration' : `η ${e.learning_rate}`;
    }
    g.roundRect(x-7,barTop,14,barHeight,4).stroke({color:line,width:1});
  }
  function draw() {
    if (!points.length || disposed) return;
    const index = Math.min(condition.events.length-1,Math.floor(position));
    const phase = position >= condition.events.length ? 1 : position-index;
    const e = condition.events[index], prenatal = e.phase==='prenatal';
    const arrived = prenatal || phase >= arrivalPhase;
    const revealed = !prenatal && phase >= decisionPhase, dead = revealed && e.death;
    const energy = prenatal ? 0 : revealed ? e.energy_after : e.energy_before;
    // Hold the native initial state during approach; replay sensory processing
    // only once the object reaches the inputs, then reveal the recorded choice.
    const neuralProgress = Math.max(0,Math.min(1,prenatal ? phase/processingEnd : (phase-arrivalPhase)/(processingEnd-arrivalPhase)));
    const step = Math.floor(neuralProgress*(e.rates.length-1));
    maxEnergy = revealed ? energyRanges[index].after : energyRanges[index].before;
    const completed = condition.events.slice(0,index+(revealed ? 1 : 0));
    get('encounter').textContent = `${e.trial} / ${data.food_encounters}`;
    get('phase').textContent = e.phase[0].toUpperCase()+e.phase.slice(1);
    const phases = ['prenatal','juvenile','adult'];
    for (const node of host.querySelectorAll('[data-stage]')) {
      const current = phases.indexOf(e.phase), stage = phases.indexOf(node.dataset.stage);
      node.dataset.state = stage===current ? 'active' : stage<current ? 'done' : 'upcoming';
      node.setAttribute('aria-current',String(stage===current));
    }
    get('life').textContent = String(e.life); get('age').textContent = prenatal ? `${e.phase_age}/${data.genome.prenatal_windows}` : String(e.age);
    get('energy').textContent = prenatal ? '—' : energy.toFixed(2); get('energy').style.color = dead ? '#bc3b43' : '#263343';
    get('deaths').textContent = String(completed.filter(v=>v.death).length);
    get('scrub').value = String(position); get('play').textContent = playing ? 'Pause' : 'Play';
    get('play').setAttribute('aria-pressed',String(playing));
    text.agent.textContent = dead ? 'Death' : e.birth && phase < .25 ? 'Birth' : prenatal ? 'Prenatal circuit' : 'E/I controller';
    text.inputs.textContent = prenatal ? 'Internal drive' : 'Inputs';
    text.energy.textContent = revealed && energyRanges[index].after > energyRanges[index].before ? 'Energy · ×2' : 'Energy';
    text.energyMax.textContent = prenatal ? '—' : String(maxEnergy); text.energyValue.textContent = prenatal ? '—' : energy.toFixed(2);
    text.energyZero.style.visibility = prenatal ? 'hidden' : 'visible';
    background.clear(); g.clear(); foreground.clear();
    const {cy,foodStart,foodStop,energyX} = geometry;
    if (!prenatal) {
      background.moveTo(foodStart,cy).lineTo(foodStop,cy).stroke({color:line,width:1.5});
      for (let x=foodStart+40;x<foodStop-10;x+=55) background.poly([x-7,cy-4,x,cy,x-7,cy+4]).stroke({color:line,width:1.3});
    }
    drawConnections(e,step,dead,neuralProgress);
    if (!prenatal) drawReadout(phase);
    drawNetwork(e,step,dead,e.birth||e.conception,phase,arrived); drawEnergy(energyX,e,energy,revealed);
    drawOutput(e,revealed,phase,step);
    const progress = Math.min(1,phase/arrivalPhase);
    const ox = foodStart+(foodStop-foodStart)*progress;
    // All incoming objects are identical. Reveal the sensory cue (size) only
    // at the stop; reveal the actual energy/colour only after the choice.
    const radius = arrived ? 10+10*e.cue : 14;
    if (!prenatal) drawObject(ox,cy,radius,revealed,e.delta);
    app.render();
  }
  const signed = value => { const magnitude=Math.abs(value).toFixed(2); return `${value < 0 && magnitude !== '0.00' ? '−' : '+'}${magnitude}`; };
  for (const control of host.querySelectorAll('button,input,select')) control.disabled=false;
  get('play').addEventListener('click',()=>{ if (position>=condition.events.length) position=0; playing=!playing; previousTime=null; draw(); },{signal});
  get('restart').addEventListener('click',()=>{ position=0; previousTime=null; draw(); },{signal});
  get('scrub').addEventListener('input',()=>{ position=Number(get('scrub').value); previousTime=null; draw(); },{signal});
  const observer = new IntersectionObserver(entries=>{ visible=entries[0].isIntersecting; previousTime=null; },{threshold:.01}); observer.observe(host);
  const resize = new ResizeObserver(layout); resize.observe(scene);
  function tick(now) {
    if (disposed) return;
    if (playing && visible && !document.hidden && previousTime !== null) {
      position=Math.min(condition.events.length,position+Math.min(now-previousTime,100)/1000*Number(get('speed').value));
      if (position>=condition.events.length) playing=false;
      draw();
    }
    previousTime=now; frame=requestAnimationFrame(tick);
  }
  layout(); frame=requestAnimationFrame(tick);
  return ()=>{ disposed=true; cancelAnimationFrame(frame); abort.abort(); observer.disconnect(); resize.disconnect(); labels.remove(); app.destroy(true,{children:true}); };
}

export function validateRecording(data) {
  if (data.schema !== 'exp154.lifecycle/v7' || data.conditions.length !== 1 || data.network.e_cells !== 16 || data.network.i_cells !== 4) throw new Error('Invalid lifecycle recording');
  const network = data.network, steps=Math.round(network.duration_ms/network.dt_ms);
  if (network.edges.some(e=>!Number.isInteger(e.source) || e.source<0 || e.source>25 || e.target<0 || e.target>19 || !Number.isFinite(e.weight))) throw new Error('Invalid native topology');
  for (const c of data.conditions) {
    if (c.sign !== 1) throw new Error('Invalid cue mapping');
    let foodTrial = 0;
    for (const [i,e] of c.events.entries()) {
      if (e.step!==i+1 || !['prenatal','juvenile','adult'].includes(e.phase) || !Number.isFinite(e.learning_rate)) throw new Error('Invalid phase timeline');
      if (e.phase==='prenatal') {
        if (e.trial!==foodTrial || e.cue!==null || e.delta!==null || e.birth || e.death || !Number.isFinite(e.loss)) throw new Error('Invalid prenatal window');
      } else {
        foodTrial++;
        if (e.updated !== e.accepted || e.trial!==foodTrial || ![e.energy_before,e.energy_after,e.cue,e.delta,e.prediction].every(Number.isFinite)) throw new Error('Invalid encounter values');
      }
      if (e.rates.length !== steps+1 || e.rates.some(row=>row.length !== 20 || row.some(v=>!Number.isFinite(v) || v<0 || v>1)) || [e.weights,e.weights_after].some(weights=>weights.length !== 2 || weights.some(row=>row.length !== 64 || row.some(v=>!Number.isFinite(v) || v<0 || v>data.genome.weight_upper)))) throw new Error('Invalid native neural recording');
      if (e.drives.length!==steps+1 || e.drives.some(row=>row.length!==6 || row.some(v=>!Number.isFinite(v)||v<0||v>1))) throw new Error('Invalid drive recording');
      const inactive = e.phase==='prenatal' ? [0,3] : [3,6];
      if (e.drives.some(row=>row.slice(...inactive).some(v=>v!==0))) throw new Error('Inputs active in the wrong developmental phase');
    }
    if (foodTrial!==data.food_encounters) throw new Error('Incomplete encounter stream');
  }
}
