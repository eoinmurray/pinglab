import { useEffect, useRef } from 'react';

export default function Demo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      ctx.scale(dpr, dpr);
    };

    resize();
    window.addEventListener('resize', resize);

    // Orb configuration
    const orbs = [
      { x: 0.2, y: 0.3, radius: 400, color: [124, 58, 237], speed: 0.0003, phase: 0 },
      { x: 0.8, y: 0.2, radius: 350, color: [236, 72, 153], speed: 0.0004, phase: 2 },
      { x: 0.5, y: 0.7, radius: 450, color: [6, 182, 212], speed: 0.0002, phase: 4 },
      { x: 0.3, y: 0.8, radius: 300, color: [249, 115, 22], speed: 0.0005, phase: 1 },
      { x: 0.7, y: 0.5, radius: 380, color: [34, 197, 94], speed: 0.00035, phase: 3 },
    ];

    // Flowing particles
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      life: number;
      maxLife: number;
      size: number;
    }> = [];

    const spawnParticle = (width: number, height: number) => {
      particles.push({
        x: Math.random() * width,
        y: height + 10,
        vx: (Math.random() - 0.5) * 0.5,
        vy: -0.5 - Math.random() * 1.5,
        life: 1,
        maxLife: 200 + Math.random() * 300,
        size: 1 + Math.random() * 2,
      });
    };

    const animate = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      time += 16;

      // Dark base
      ctx.fillStyle = '#030014';
      ctx.fillRect(0, 0, width, height);

      // Animated gradient orbs with blur effect
      ctx.globalCompositeOperation = 'lighter';

      for (const orb of orbs) {
        const wobbleX = Math.sin(time * orb.speed + orb.phase) * 100;
        const wobbleY = Math.cos(time * orb.speed * 0.7 + orb.phase) * 80;
        const pulseRadius = orb.radius + Math.sin(time * orb.speed * 2) * 50;

        const x = orb.x * width + wobbleX;
        const y = orb.y * height + wobbleY;

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, pulseRadius);
        gradient.addColorStop(0, `rgba(${orb.color.join(',')}, 0.4)`);
        gradient.addColorStop(0.4, `rgba(${orb.color.join(',')}, 0.1)`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
      }

      // Mesh grid effect
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;

      const gridSize = 60;
      for (let x = 0; x < width; x += gridSize) {
        const wave = Math.sin(time * 0.001 + x * 0.01) * 5;
        ctx.beginPath();
        ctx.moveTo(x + wave, 0);
        ctx.lineTo(x + wave, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        const wave = Math.cos(time * 0.001 + y * 0.01) * 5;
        ctx.beginPath();
        ctx.moveTo(0, y + wave);
        ctx.lineTo(width, y + wave);
        ctx.stroke();
      }

      // Spawn and update particles
      if (Math.random() < 0.3) spawnParticle(width, height);

      ctx.globalCompositeOperation = 'lighter';
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx + Math.sin(time * 0.002 + p.y * 0.01) * 0.3;
        p.y += p.vy;
        p.life -= 1 / p.maxLife;

        if (p.life <= 0 || p.y < -10) {
          particles.splice(i, 1);
          continue;
        }

        const alpha = p.life * 0.6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.fill();
      }

      // Horizontal light streaks
      ctx.globalCompositeOperation = 'lighter';
      for (let i = 0; i < 3; i++) {
        const streakY = (height * 0.3) + (i * height * 0.2);
        const streakPhase = time * 0.0005 + i;
        const streakX = ((Math.sin(streakPhase) + 1) / 2) * width * 1.5 - width * 0.25;

        const streakGradient = ctx.createLinearGradient(streakX - 200, 0, streakX + 200, 0);
        streakGradient.addColorStop(0, 'rgba(255, 255, 255, 0)');
        streakGradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.03)');
        streakGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

        ctx.fillStyle = streakGradient;
        ctx.fillRect(streakX - 200, streakY - 1, 400, 2);
      }

      // Aurora wave effect at top
      ctx.globalCompositeOperation = 'lighter';
      for (let i = 0; i < 5; i++) {
        ctx.beginPath();
        ctx.moveTo(0, 0);

        for (let x = 0; x <= width; x += 10) {
          const waveHeight = 100 + i * 30;
          const y = Math.sin(x * 0.003 + time * 0.001 + i) * waveHeight * 0.5 +
                    Math.sin(x * 0.007 - time * 0.0015) * waveHeight * 0.3;
          ctx.lineTo(x, y + 50);
        }

        ctx.lineTo(width, 0);
        ctx.closePath();

        const auroraGradient = ctx.createLinearGradient(0, 0, 0, 200);
        const hue = (i * 40 + time * 0.02) % 360;
        auroraGradient.addColorStop(0, `hsla(${hue}, 80%, 60%, 0.05)`);
        auroraGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = auroraGradient;
        ctx.fill();
      }

      // Vignette
      ctx.globalCompositeOperation = 'source-over';
      const vignette = ctx.createRadialGradient(
        width / 2, height / 2, height * 0.2,
        width / 2, height / 2, height
      );
      vignette.addColorStop(0, 'rgba(0, 0, 0, 0)');
      vignette.addColorStop(1, 'rgba(0, 0, 0, 0.5)');
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, width, height);

      // Noise texture simulation
      ctx.globalCompositeOperation = 'overlay';
      ctx.fillStyle = 'rgba(128, 128, 128, 0.02)';
      for (let i = 0; i < 50; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        ctx.fillRect(x, y, 2, 2);
      }

      ctx.globalCompositeOperation = 'source-over';

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#030014]">
      <canvas ref={canvasRef} className="fixed inset-0 w-full h-full" />

      {/* Content */}
      <div className="relative z-10">
        {/* Nav */}
        <nav className="flex items-center justify-between px-8 py-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500" />
            <span className="text-white font-semibold text-lg">pinglab</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm text-white/60">
            <a href="#" className="hover:text-white transition-colors">Platform</a>
            <a href="#" className="hover:text-white transition-colors">Research</a>
            <a href="#" className="hover:text-white transition-colors">Docs</a>
            <a href="#" className="hover:text-white transition-colors">Pricing</a>
          </div>
          <div className="flex items-center gap-4">
            <button className="text-sm text-white/60 hover:text-white transition-colors">Sign in</button>
            <button className="text-sm px-4 py-2 rounded-full bg-white text-black font-medium hover:bg-white/90 transition-colors">
              Get Started
            </button>
          </div>
        </nav>

        {/* Hero */}
        <div className="max-w-6xl mx-auto px-8 pt-32 pb-20">
          {/* Badge */}
          <div className="flex justify-center mb-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm">
              <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-sm text-white/70">Series B — $120M raised from Sequoia & a]6z</span>
            </div>
          </div>

          {/* Heading */}
          <h1 className="text-center text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight mb-8">
            <span className="bg-gradient-to-b from-white via-white/90 to-white/50 bg-clip-text text-transparent">
              The infrastructure for
            </span>
            <br />
            <span className="bg-gradient-to-r from-violet-400 via-fuchsia-400 to-cyan-400 bg-clip-text text-transparent">
              biological intelligence
            </span>
          </h1>

          {/* Subheading */}
          <p className="text-center text-lg md:text-xl text-white/50 max-w-2xl mx-auto mb-12 leading-relaxed">
            Accelerate discovery with our unified platform for computational biology.
            From protein design to drug discovery — 10x faster, infinitely scalable.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20">
            <button className="group relative px-8 py-4 rounded-full bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-medium text-lg overflow-hidden transition-all hover:shadow-[0_0_40px_rgba(167,139,250,0.4)]">
              <span className="relative z-10">Start Building Free</span>
              <div className="absolute inset-0 bg-gradient-to-r from-violet-500 to-fuchsia-500 opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
            <button className="px-8 py-4 rounded-full border border-white/20 text-white font-medium text-lg hover:bg-white/5 transition-all flex items-center gap-2">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" />
              </svg>
              Watch Demo
            </button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            {[
              { value: '50B+', label: 'Proteins analyzed' },
              { value: '10x', label: 'Faster discovery' },
              { value: '99.9%', label: 'Uptime SLA' },
              { value: '300+', label: 'Research teams' },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-3xl md:text-4xl font-bold bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-white/40">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Logos */}
        <div className="border-t border-white/5 py-12">
          <p className="text-center text-sm text-white/30 mb-8">Trusted by leading research institutions</p>
          <div className="flex flex-wrap justify-center items-center gap-12 px-8 opacity-40">
            {['Stanford', 'MIT', 'Genentech', 'Moderna', 'DeepMind'].map((name) => (
              <div key={name} className="text-white/60 text-xl font-semibold tracking-tight">
                {name}
              </div>
            ))}
          </div>
        </div>

        {/* Feature cards */}
        <div className="max-w-6xl mx-auto px-8 py-24">
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: '⚡',
                title: 'GPU-Native Compute',
                description: 'Run molecular simulations on our distributed GPU cluster. Scale from 1 to 10,000 nodes instantly.'
              },
              {
                icon: '🧬',
                title: 'Foundation Models',
                description: 'Access state-of-the-art protein language models trained on 200M+ sequences with zero setup.'
              },
              {
                icon: '🔬',
                title: 'Lab Integration',
                description: 'Connect your wet lab with our SDK. Automated experiment tracking and reproducibility.'
              },
            ].map((feature, i) => (
              <div
                key={i}
                className="group p-8 rounded-2xl bg-gradient-to-b from-white/[0.05] to-transparent border border-white/10 hover:border-white/20 transition-all hover:bg-white/[0.02]"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                <p className="text-white/50 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
