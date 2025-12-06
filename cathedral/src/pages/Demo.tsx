import { useEffect, useRef, useState, useCallback } from 'react';

// Neural network simulation types
interface Neuron {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  activation: number;
  lastFired: number;
  type: 'excitatory' | 'inhibitory' | 'input' | 'output';
  layer: number;
}

interface Synapse {
  from: number;
  to: number;
  weight: number;
  signal: number;
  signalPos: number;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
}

// Color palette - cyberpunk neuroscience
const COLORS = {
  bg: '#0a0a0f',
  grid: 'rgba(0, 255, 200, 0.03)',
  neuronExcitatory: '#00ffc8',
  neuronInhibitory: '#ff0066',
  neuronInput: '#00aaff',
  neuronOutput: '#ffaa00',
  synapse: 'rgba(0, 255, 200, 0.15)',
  signal: '#ffffff',
  text: '#00ffc8',
  textDim: 'rgba(0, 255, 200, 0.5)',
  scanline: 'rgba(0, 0, 0, 0.1)',
  glow: '#00ffc8',
};

export default function Demo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>(0);
  const neuronsRef = useRef<Neuron[]>([]);
  const synapsesRef = useRef<Synapse[]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const mouseRef = useRef({ x: 0, y: 0, active: false });
  const timeRef = useRef(0);
  const statsRef = useRef({
    firings: 0,
    avgActivation: 0,
    signalsInFlight: 0,
    fps: 0,
    lastFrameTime: 0,
    frameCount: 0,
  });

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [glitchActive, setGlitchActive] = useState(false);

  // Initialize neural network
  const initNetwork = useCallback((width: number, height: number) => {
    const neurons: Neuron[] = [];
    const synapses: Synapse[] = [];

    const numNeurons = Math.floor((width * height) / 15000);
    const layers = 5;
    const neuronsPerLayer = Math.floor(numNeurons / layers);

    // Create neurons in layers
    for (let layer = 0; layer < layers; layer++) {
      const layerX = (width * 0.1) + (width * 0.8 * layer / (layers - 1));

      for (let i = 0; i < neuronsPerLayer; i++) {
        const spread = height * 0.7;
        const baseY = height * 0.15 + (spread * i / neuronsPerLayer);

        let type: Neuron['type'] = 'excitatory';
        if (layer === 0) type = 'input';
        else if (layer === layers - 1) type = 'output';
        else if (Math.random() < 0.2) type = 'inhibitory';

        neurons.push({
          id: neurons.length,
          x: layerX + (Math.random() - 0.5) * 60,
          y: baseY + (Math.random() - 0.5) * 40,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          radius: 3 + Math.random() * 3,
          activation: Math.random() * 0.3,
          lastFired: -1000,
          type,
          layer,
        });
      }
    }

    // Create synapses between adjacent layers
    for (const neuron of neurons) {
      const nextLayerNeurons = neurons.filter(n => n.layer === neuron.layer + 1);
      const connectionCount = Math.min(3 + Math.floor(Math.random() * 4), nextLayerNeurons.length);

      // Sort by distance and connect to nearest
      const sorted = nextLayerNeurons
        .map(n => ({ n, dist: Math.hypot(n.x - neuron.x, n.y - neuron.y) }))
        .sort((a, b) => a.dist - b.dist);

      for (let i = 0; i < connectionCount; i++) {
        if (sorted[i]) {
          synapses.push({
            from: neuron.id,
            to: sorted[i].n.id,
            weight: 0.3 + Math.random() * 0.7,
            signal: 0,
            signalPos: 0,
          });
        }
      }
    }

    neuronsRef.current = neurons;
    synapsesRef.current = synapses;
    particlesRef.current = [];
  }, []);

  // Spawn particles at neuron firing
  const spawnParticles = useCallback((x: number, y: number, color: string) => {
    const count = 5 + Math.floor(Math.random() * 5);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.5 + Math.random() * 1.5;
      particlesRef.current.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        maxLife: 30 + Math.random() * 30,
        color,
      });
    }
  }, []);

  // Update simulation
  const update = useCallback((time: number) => {
    const dt = 1;
    const neurons = neuronsRef.current;
    const synapses = synapsesRef.current;
    const particles = particlesRef.current;
    const mouse = mouseRef.current;

    let firings = 0;
    let totalActivation = 0;
    let signalsInFlight = 0;

    // Update neurons
    for (const neuron of neurons) {
      // Gentle floating motion
      neuron.x += neuron.vx * dt;
      neuron.y += neuron.vy * dt;

      // Boundary bounce
      if (neuron.x < 50 || neuron.x > dimensions.width - 50) neuron.vx *= -1;
      if (neuron.y < 50 || neuron.y > dimensions.height - 50) neuron.vy *= -1;

      // Mouse interaction - activate nearby neurons
      if (mouse.active) {
        const dist = Math.hypot(mouse.x - neuron.x, mouse.y - neuron.y);
        if (dist < 100) {
          neuron.activation += (1 - dist / 100) * 0.1;
        }
      }

      // Spontaneous input layer activity
      if (neuron.type === 'input' && Math.random() < 0.02) {
        neuron.activation += 0.3;
      }

      // Decay
      neuron.activation *= 0.98;

      // Fire if threshold reached
      if (neuron.activation > 0.8 && time - neuron.lastFired > 30) {
        neuron.lastFired = time;
        firings++;

        // Propagate to connected neurons
        for (const synapse of synapses) {
          if (synapse.from === neuron.id) {
            synapse.signal = 1;
            synapse.signalPos = 0;
          }
        }

        // Spawn particles
        const color = neuron.type === 'inhibitory' ? COLORS.neuronInhibitory : COLORS.neuronExcitatory;
        spawnParticles(neuron.x, neuron.y, color);
      }

      totalActivation += neuron.activation;
    }

    // Update synapses
    for (const synapse of synapses) {
      if (synapse.signal > 0) {
        signalsInFlight++;
        synapse.signalPos += 0.03;

        if (synapse.signalPos >= 1) {
          // Signal arrived
          const targetNeuron = neurons.find(n => n.id === synapse.to);
          if (targetNeuron) {
            const fromNeuron = neurons.find(n => n.id === synapse.from);
            const modifier = fromNeuron?.type === 'inhibitory' ? -0.5 : 1;
            targetNeuron.activation += synapse.weight * synapse.signal * modifier;
          }
          synapse.signal = 0;
          synapse.signalPos = 0;
        }
      }
    }

    // Update particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.02; // gravity
      p.life -= 1 / p.maxLife;

      if (p.life <= 0) {
        particles.splice(i, 1);
      }
    }

    // Update stats
    statsRef.current.firings = firings;
    statsRef.current.avgActivation = totalActivation / neurons.length;
    statsRef.current.signalsInFlight = signalsInFlight;

    // FPS calculation
    statsRef.current.frameCount++;
    if (time - statsRef.current.lastFrameTime > 1000) {
      statsRef.current.fps = statsRef.current.frameCount;
      statsRef.current.frameCount = 0;
      statsRef.current.lastFrameTime = time;
    }
  }, [dimensions, spawnParticles]);

  // Render frame
  const render = useCallback((ctx: CanvasRenderingContext2D, time: number) => {
    const { width, height } = dimensions;
    const neurons = neuronsRef.current;
    const synapses = synapsesRef.current;
    const particles = particlesRef.current;

    // Clear with dark background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw synapses
    for (const synapse of synapses) {
      const from = neurons.find(n => n.id === synapse.from);
      const to = neurons.find(n => n.id === synapse.to);
      if (!from || !to) continue;

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = synapse.signal > 0
        ? `rgba(0, 255, 200, ${0.3 + synapse.signal * 0.7})`
        : COLORS.synapse;
      ctx.lineWidth = synapse.signal > 0 ? 2 : 1;
      ctx.stroke();

      // Draw traveling signal
      if (synapse.signal > 0) {
        const signalX = from.x + (to.x - from.x) * synapse.signalPos;
        const signalY = from.y + (to.y - from.y) * synapse.signalPos;

        ctx.beginPath();
        ctx.arc(signalX, signalY, 4, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.signal;
        ctx.shadowColor = COLORS.glow;
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Draw particles
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2 * p.life, 0, Math.PI * 2);
      ctx.fillStyle = p.color.replace(')', `, ${p.life})`).replace('rgb', 'rgba');
      ctx.fill();
    }

    // Draw neurons
    for (const neuron of neurons) {
      const isFiring = time - neuron.lastFired < 10;
      const glowIntensity = Math.max(neuron.activation, isFiring ? 1 : 0);

      // Glow effect
      if (glowIntensity > 0.3) {
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, neuron.radius + 10, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(
          neuron.x, neuron.y, neuron.radius,
          neuron.x, neuron.y, neuron.radius + 15
        );

        let color = COLORS.neuronExcitatory;
        if (neuron.type === 'inhibitory') color = COLORS.neuronInhibitory;
        if (neuron.type === 'input') color = COLORS.neuronInput;
        if (neuron.type === 'output') color = COLORS.neuronOutput;

        // Parse hex to rgba
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);

        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${glowIntensity * 0.5})`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.fill();
      }

      // Neuron body
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);

      let baseColor = COLORS.neuronExcitatory;
      if (neuron.type === 'inhibitory') baseColor = COLORS.neuronInhibitory;
      if (neuron.type === 'input') baseColor = COLORS.neuronInput;
      if (neuron.type === 'output') baseColor = COLORS.neuronOutput;

      ctx.fillStyle = baseColor;
      ctx.shadowColor = baseColor;
      ctx.shadowBlur = isFiring ? 20 : 5;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Inner highlight
      ctx.beginPath();
      ctx.arc(neuron.x - neuron.radius * 0.3, neuron.y - neuron.radius * 0.3, neuron.radius * 0.3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.fill();
    }

    // Scanline effect
    ctx.fillStyle = COLORS.scanline;
    for (let y = 0; y < height; y += 3) {
      ctx.fillRect(0, y, width, 1);
    }

    // Vignette
    const vignette = ctx.createRadialGradient(
      width / 2, height / 2, height * 0.3,
      width / 2, height / 2, height * 0.8
    );
    vignette.addColorStop(0, 'rgba(0, 0, 0, 0)');
    vignette.addColorStop(1, 'rgba(0, 0, 0, 0.6)');
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, width, height);

    // Glitch effect
    if (glitchActive) {
      const sliceHeight = 5 + Math.random() * 20;
      const sliceY = Math.random() * height;
      const sliceOffset = (Math.random() - 0.5) * 30;

      const imageData = ctx.getImageData(0, sliceY, width, sliceHeight);
      ctx.putImageData(imageData, sliceOffset, sliceY);

      // Color channel split
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = 'rgba(255, 0, 0, 0.03)';
      ctx.fillRect(2, 0, width, height);
      ctx.fillStyle = 'rgba(0, 255, 255, 0.03)';
      ctx.fillRect(-2, 0, width, height);
      ctx.globalCompositeOperation = 'source-over';
    }
  }, [dimensions, glitchActive]);

  // Draw HUD overlay
  const renderHUD = useCallback((ctx: CanvasRenderingContext2D, time: number) => {
    const { width, height } = dimensions;
    const stats = statsRef.current;

    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = COLORS.text;
    ctx.shadowColor = COLORS.glow;
    ctx.shadowBlur = 10;

    // Top left - System info
    const lines = [
      `NEURAL_SIM v2.847.1`,
      `────────────────────`,
      `NEURONS: ${neuronsRef.current.length}`,
      `SYNAPSES: ${synapsesRef.current.length}`,
      `PARTICLES: ${particlesRef.current.length}`,
      ``,
      `FPS: ${stats.fps}`,
      `TIME: ${(time / 1000).toFixed(2)}s`,
    ];

    lines.forEach((line, i) => {
      ctx.fillText(line, 20, 30 + i * 16);
    });

    // Top right - Activity monitor
    const rightLines = [
      `ACTIVITY MONITOR`,
      `────────────────────`,
      `FIRINGS/FRAME: ${stats.firings}`,
      `AVG_ACTIVATION: ${stats.avgActivation.toFixed(3)}`,
      `SIGNALS_ACTIVE: ${stats.signalsInFlight}`,
      ``,
      `[INPUT] ████████`,
      `[HIDDEN] ██████`,
      `[OUTPUT] ████`,
    ];

    ctx.textAlign = 'right';
    rightLines.forEach((line, i) => {
      ctx.fillText(line, width - 20, 30 + i * 16);
    });
    ctx.textAlign = 'left';

    // Bottom left - Legend
    ctx.fillStyle = COLORS.neuronInput;
    ctx.fillText('● INPUT', 20, height - 60);
    ctx.fillStyle = COLORS.neuronExcitatory;
    ctx.fillText('● EXCITATORY', 20, height - 44);
    ctx.fillStyle = COLORS.neuronInhibitory;
    ctx.fillText('● INHIBITORY', 20, height - 28);
    ctx.fillStyle = COLORS.neuronOutput;
    ctx.fillText('● OUTPUT', 120, height - 60);

    // Bottom center - Instructions
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = 'center';
    ctx.fillText('[ MOVE MOUSE TO STIMULATE NEURAL ACTIVITY ]', width / 2, height - 20);
    ctx.textAlign = 'left';

    // Decorative corners
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;

    // Top left corner
    ctx.beginPath();
    ctx.moveTo(10, 10);
    ctx.lineTo(10, 50);
    ctx.moveTo(10, 10);
    ctx.lineTo(50, 10);
    ctx.stroke();

    // Top right corner
    ctx.beginPath();
    ctx.moveTo(width - 10, 10);
    ctx.lineTo(width - 10, 50);
    ctx.moveTo(width - 10, 10);
    ctx.lineTo(width - 50, 10);
    ctx.stroke();

    // Bottom left corner
    ctx.beginPath();
    ctx.moveTo(10, height - 10);
    ctx.lineTo(10, height - 50);
    ctx.moveTo(10, height - 10);
    ctx.lineTo(50, height - 10);
    ctx.stroke();

    // Bottom right corner
    ctx.beginPath();
    ctx.moveTo(width - 10, height - 10);
    ctx.lineTo(width - 10, height - 50);
    ctx.moveTo(width - 10, height - 10);
    ctx.lineTo(width - 50, height - 10);
    ctx.stroke();

    ctx.shadowBlur = 0;
  }, [dimensions]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let running = true;

    const loop = (timestamp: number) => {
      if (!running) return;

      timeRef.current = timestamp;
      update(timestamp);
      render(ctx, timestamp);
      renderHUD(ctx, timestamp);

      animationRef.current = requestAnimationFrame(loop);
    };

    animationRef.current = requestAnimationFrame(loop);

    return () => {
      running = false;
      cancelAnimationFrame(animationRef.current);
    };
  }, [dimensions, update, render, renderHUD]);

  // Handle resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleResize = () => {
      const { width, height } = container.getBoundingClientRect();
      setDimensions({ width, height });
      initNetwork(width, height);
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => window.removeEventListener('resize', handleResize);
  }, [initNetwork]);

  // Handle mouse
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = e.clientX - rect.left;
      mouseRef.current.y = e.clientY - rect.top;
      mouseRef.current.active = true;
    };

    const handleMouseLeave = () => {
      mouseRef.current.active = false;
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  // Random glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() < 0.1) {
        setGlitchActive(true);
        setTimeout(() => setGlitchActive(false), 50 + Math.random() * 100);
      }
    }, 2000);

    return () => clearInterval(glitchInterval);
  }, []);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 bg-[#0a0a0f] overflow-hidden cursor-crosshair"
      style={{ fontFamily: '"JetBrains Mono", monospace' }}
    >
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className="block w-full h-full"
      />

      {/* Chromatic aberration overlay */}
      <div
        className="pointer-events-none fixed inset-0 mix-blend-screen opacity-[0.02]"
        style={{
          background: 'linear-gradient(90deg, #ff0000 0%, transparent 33%, transparent 66%, #00ffff 100%)',
        }}
      />

      {/* Noise texture overlay */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Corner decorations */}
      <div className="pointer-events-none fixed top-4 left-4 w-16 h-16 border-l-2 border-t-2 border-[#00ffc8]/30" />
      <div className="pointer-events-none fixed top-4 right-4 w-16 h-16 border-r-2 border-t-2 border-[#00ffc8]/30" />
      <div className="pointer-events-none fixed bottom-4 left-4 w-16 h-16 border-l-2 border-b-2 border-[#00ffc8]/30" />
      <div className="pointer-events-none fixed bottom-4 right-4 w-16 h-16 border-r-2 border-b-2 border-[#00ffc8]/30" />

      {/* Title */}
      <div className="pointer-events-none fixed top-6 left-1/2 -translate-x-1/2 text-center">
        <h1
          className="text-[#00ffc8] text-xl tracking-[0.3em] font-light"
          style={{ textShadow: '0 0 20px rgba(0, 255, 200, 0.5)' }}
        >
          NEURAL_CORTEX_SIMULATION
        </h1>
        <div className="text-[#00ffc8]/40 text-[10px] tracking-[0.5em] mt-1">
          PINGLAB RESEARCH DIVISION
        </div>
      </div>
    </div>
  );
}
