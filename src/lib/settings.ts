import AsyncStorage from '@react-native-async-storage/async-storage';
import { clamp } from './posture';

export type Prefs = {
  /** Neck bend (degrees from neutral) before the posture counts as bent. */
  threshold: number;
  /** How long the bend must last before an alert fires. */
  holdMs: number;
  /** Play a sound/notification when posture is bad for too long. */
  notifications: boolean;
  /** Pitch angle of a level head, measured once during calibration. */
  neutral: number | null;
};

export const DEFAULT_PREFS: Prefs = {
  threshold: 20,
  holdMs: 6000,
  notifications: true,
  neutral: null,
};

export const THRESHOLD_OPTIONS = [15, 20, 25] as const;
export const HOLD_OPTIONS = [4000, 8000, 15000] as const;

const KEY = 'techneck.prefs.v1';

/** Saved data is untrusted: it may come from an older build or a tampered store. */
export function sanitizePrefs(raw: unknown): Prefs {
  const value = (raw ?? {}) as Partial<Prefs>;
  const threshold = Number(value.threshold);
  const holdMs = Number(value.holdMs);
  const neutral = value.neutral === null || value.neutral === undefined ? null : Number(value.neutral);
  const notifications =
    typeof value.notifications === 'boolean'
      ? value.notifications
      : typeof (value as { haptics?: boolean }).haptics === 'boolean'
        ? (value as { haptics: boolean }).haptics
        : DEFAULT_PREFS.notifications;
  return {
    threshold: Number.isFinite(threshold) ? clamp(threshold, 10, 40) : DEFAULT_PREFS.threshold,
    holdMs: Number.isFinite(holdMs) ? clamp(holdMs, 2000, 30000) : DEFAULT_PREFS.holdMs,
    notifications,
    neutral: neutral !== null && Number.isFinite(neutral) && Math.abs(neutral) <= 90 ? neutral : null,
  };
}

export async function loadPrefs(): Promise<Prefs> {
  try {
    const stored = await AsyncStorage.getItem(KEY);
    return stored === null ? DEFAULT_PREFS : sanitizePrefs(JSON.parse(stored));
  } catch {
    return DEFAULT_PREFS;
  }
}

export async function savePrefs(prefs: Prefs): Promise<void> {
  try {
    await AsyncStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    // Settings are a convenience; losing them must never break monitoring.
  }
}
