const formatter = new Intl.NumberFormat('en-GB', {
  minimumSignificantDigits: 2,
  maximumSignificantDigits: 2,
});

export function formatTableNumber(value: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error('Table numbers must be finite');
  }
  return formatter.format(value);
}
