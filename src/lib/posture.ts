export type Severity = 'good' | 'warn' | 'bad';

/** Degrees past the limit that still count as "almost". */
export const WARN_BAND = 8;

/** How far off neutral the gauge ring can go before it is full. */
export const GAUGE_MAX = 45;

export const CALIBRATION_MS = 1500;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Median of the samples, ignoring empty input. */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

/** Exponential moving average. */
export function smooth(previous: number | null, next: number, alpha = 0.3): number {
  return previous === null ? next : previous + alpha * (next - previous);
}

export function bendFrom(neutral: number | null, pitch: number): number {
  if (neutral === null) return 0;
  return Math.abs(pitch - neutral);
}

export function severityFor(bend: number, threshold: number): Severity {
  if (bend <= threshold) return 'good';
  if (bend <= threshold + WARN_BAND) return 'warn';
  return 'bad';
}

export function uprightPercent(uprightMs: number, trackedMs: number): number {
  if (trackedMs <= 0) return 0;
  return clamp(Math.round((uprightMs / trackedMs) * 100), 0, 100);
}

/** 0:07, 4:31, 1:02:09 */
export function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(seconds)}` : `${minutes}:${pad(seconds)}`;
}
