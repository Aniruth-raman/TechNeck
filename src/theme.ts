export const colors = {
  bg: '#070B10',
  card: '#0E1621',
  raised: '#131F2C',
  border: '#1D2A38',
  text: '#EDF4FA',
  muted: '#8CA1B4',
  faint: '#5F7488',
  mint: '#3BE8B0',
  amber: '#FFC24B',
  coral: '#FF6B6B',
  mintSoft: 'rgba(59, 232, 176, 0.12)',
  amberSoft: 'rgba(255, 194, 75, 0.12)',
  coralSoft: 'rgba(255, 107, 107, 0.14)',
} as const;

export const radius = {
  sm: 12,
  md: 20,
  lg: 28,
} as const;

export const severityColor = {
  good: colors.mint,
  warn: colors.amber,
  bad: colors.coral,
} as const;
