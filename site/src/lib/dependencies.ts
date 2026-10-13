// Scientific dependencies declared by the existing articles.
export const dependencies: Record<string, string[]> = {
  exp022: [], exp023: [], exp024: ['exp022'], exp025: ['exp022'],
  exp037: ['exp022'], exp038: ['exp022'], exp041: ['exp022'],
  exp042: ['exp022'], exp044: ['exp022'], exp046: ['exp041', 'exp022'],
  exp047: [], exp049: ['exp022'], exp054: [], exp080: [], exp081: [],
  exp082: ['exp022'], exp099: [],
  exp110: ['exp025', 'exp117', 'exp037', 'exp038', 'exp041', 'exp044', 'exp046', 'exp054'],
  exp117: [], exp123: ['exp022'],
};
export function downstream(experiment: string): string[] {
  return Object.keys(dependencies).filter(id => dependencies[id].includes(experiment)).sort();
}
