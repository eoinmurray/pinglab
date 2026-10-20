// Recorded population means, sampled on the same clock as the circuit replay.
export function createPistons({ data, scene, text, Graphics, colors }) {
  const samples = Math.round(data.duration_ms / data.dt_ms);
  const definitions = [
    { key: 'mean_g_e', label: 'g<sub>E</sub>', x: 842, color: colors.e, scale: 'conductance_us', unit: 'µS' },
    { key: 'mean_g_i', label: 'g<sub>I</sub>', x: 936, color: colors.i, scale: 'conductance_us', unit: 'µS' },
    { key: 'mean_v_e', label: 'V<sub>E</sub>', x: 1052, color: colors.e, scale: 'voltage_mv', unit: 'mV' },
    { key: 'mean_v_i', label: 'V<sub>I</sub>', x: 1146, color: colors.i, scale: 'voltage_mv', unit: 'mV' },
  ];
  const centered = (value, x, y, size = 14, color = colors.ink) => {
    const label = text(value, x, y, size, color);
    label.style.transform = 'translateX(-50%)';
    return label;
  };
  text('B', 824, 8, 14);
  centered('Conductances onto E', 921, 32, 15);
  centered('Population voltages', 1125, 32, 15);
  const tracks = new Graphics(), fill = new Graphics(), heads = new Graphics();
  tracks.moveTo(814, 8).lineTo(814, 525).stroke({ color: 0xe1e6eb, width: 1 });
  scene.addChild(tracks, fill, heads);
  const top = 112, bottom = 272, trackWidth = 64;
  const pistons = definitions.map(definition => {
    const series = data.means?.[definition.key];
    if (!Array.isArray(series) || series.length !== samples || !series.every(Number.isFinite)) {
      throw new Error(`Invalid recorded series: ${definition.key}`);
    }
    const [min, max] = data.piston_scales[definition.scale];
    const center = definition.x + trackWidth / 2;
    const label = centered('', center, 60, 25, definition.color);
    label.innerHTML = definition.label;
    centered(definition.unit === 'µS' ? max.toFixed(3) : String(max), center, 91, 13, colors.muted);
    centered(String(min), center, 283, 13, colors.muted);
    const value = centered('', center, 308, 18, definition.color);
    centered(definition.unit, center, 334, 13, colors.muted);
    tracks.roundRect(definition.x, top, trackWidth, bottom - top, 6)
      .fill({ color: 0xf7f9fa }).stroke({ color: 0xcbd5df, width: 1 });
    for (let tick = 1; tick <= 3; tick++) {
      const y = bottom - (bottom - top) * tick / 4;
      tracks.moveTo(definition.x, y).lineTo(definition.x + 7, y)
        .moveTo(definition.x + trackWidth - 7, y).lineTo(definition.x + trackWidth, y)
        .stroke({ color: 0xcbd5df, width: 1 });
    }
    return { ...definition, series, min, max, value };
  });
  return time => {
    const index = Math.min(samples - 1, Math.floor(time / data.dt_ms));
    fill.clear(); heads.clear();
    for (const piston of pistons) {
      const value = piston.series[index];
      const fraction = Math.max(0, Math.min(1, (value - piston.min) / (piston.max - piston.min)));
      const y = bottom - 4 - fraction * (bottom - top - 8);
      if (fraction > 0) fill.rect(piston.x + 4, y, trackWidth - 8, bottom - 4 - y)
        .fill({ color: piston.color, alpha: 0.2 });
      heads.roundRect(piston.x - 5, y - 3, trackWidth + 10, 6, 2)
        .fill({ color: piston.color, alpha: 0.9 });
      const formatted = piston.unit === 'µS' ? value.toFixed(4) : value.toFixed(1);
      if (piston.value.textContent !== formatted) piston.value.textContent = formatted;
    }
  };
}
