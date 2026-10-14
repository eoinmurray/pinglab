// The full prescribed input-rate protocol recorded by the analysis stage.
export function createAfferentRate({ data, scene, text, Graphics, colors }) {
  const samples = Math.round(data.duration_ms / data.dt_ms);
  const rates = data.afferent_rates_hz;
  for (const pop of ['e', 'i']) {
    if (!Array.isArray(rates?.[pop]) || rates[pop].length !== samples ||
      !rates[pop].every(value => Number.isFinite(value) && value >= 0)) {
      throw new Error(`Invalid afferent-rate trace: ${pop}`);
    }
  }
  const maxRate = Math.ceil(Math.max(...rates.e, ...rates.i) * 12) / 10;
  const left = 867, top = 397, width = 345, height = 99, bottom = top + height;
  const x = time => left + time / data.duration_ms * width;
  const y = rate => bottom - rate / maxRate * height;
  const axes = new Graphics(), traces = new Graphics(), cursor = new Graphics();
  scene.addChild(axes, traces, cursor);
  text('C', 824, 370, 14);
  text('Afferent rate (Hz/source)', 846, 369, 15);
  text('E', 1156, 370, 13, colors.e);
  text('I', 1202, 370, 13, colors.i);
  axes.moveTo(1138, 378).lineTo(1151, 378).stroke({ color: colors.e, width: 2 });
  axes.moveTo(1181, 378).lineTo(1188, 378).moveTo(1192, 378).lineTo(1198, 378)
    .stroke({ color: colors.i, width: 2 });
  axes.moveTo(left, top).lineTo(left, bottom).lineTo(left + width, bottom)
    .stroke({ color: 0x98a5b2, width: 1 });
  for (const rate of [0, maxRate / 2, maxRate]) {
    axes.moveTo(left, y(rate)).lineTo(left + width, y(rate))
      .stroke({ color: 0xe6ebef, width: 0.7 });
    const label = text(rate.toFixed(2).replace(/\.?0+$/, ''), left - 10, y(rate) - 7, 12, colors.muted);
    label.style.transform = 'translateX(-100%)';
  }
  for (let tick = 0; tick <= 3; tick++) {
    const time = data.duration_ms * tick / 3;
    axes.moveTo(x(time), bottom).lineTo(x(time), bottom + 4)
      .stroke({ color: 0x98a5b2, width: 1 });
    const label = text(String(Math.round(data.start_ms + time)), x(time), bottom + 9, 12, colors.muted);
    label.style.transform = 'translateX(-50%)';
  }
  const unit = text('ms', left + width / 2, 524, 12, colors.muted);
  unit.style.transform = 'translateX(-50%)';
  for (const pop of ['e', 'i']) {
    let distance = 0;
    traces.moveTo(x(0), y(rates[pop][0]));
    for (let index = 1; index <= samples; index++) {
      const previousTime = (index - 1) * data.dt_ms;
      const time = Math.min(index * data.dt_ms, data.duration_ms);
      const previousRate = rates[pop][index - 1];
      const rate = rates[pop][Math.min(index, samples - 1)];
      distance += Math.hypot(x(time) - x(previousTime), y(rate) - y(previousRate));
      if (pop === 'e' || distance % 13 < 8) traces.lineTo(x(time), y(rate));
      else traces.moveTo(x(time), y(rate));
    }
    traces.stroke({ color: colors[pop], width: 2 });
  }
  return time => {
    cursor.clear().rect(x(time) - 2, top, 4, height).fill({ color: 0x626b74, alpha: 0.22 });
    cursor.moveTo(x(time), top).lineTo(x(time), bottom)
      .stroke({ color: 0x626b74, width: 1.3, alpha: 0.8 });
  };
}
