import { useSyncExternalStore } from 'react';
import { schedulePostureAlert } from './notifications';
import {
  bendFrom,
  CALIBRATION_MS,
  clamp,
  median,
  severityFor,
  smooth,
  type Severity,
} from './posture';
import { DEFAULT_PREFS, type Prefs } from './settings';

export type Status = Severity | 'no-face' | 'calibrating';

export type SessionSummary = {
  durationMs: number;
  trackedMs: number;
  uprightMs: number;
  alerts: number;
  endedAt: number;
};

export type Snapshot = {
  status: Status;
  bend: number;
  pitch: number | null;
  face: boolean;
  alerting: boolean;
  calibrating: boolean;
  calibrationProgress: number;
  durationMs: number;
  trackedMs: number;
  uprightMs: number;
  alerts: number;
  active: boolean;
  neutral: number | null;
  threshold: number;
  lastSession: SessionSummary | null;
};

const FACE_TIMEOUT_MS = 800;
const ALERT_COOLDOWN_MS = 20_000;
const PUBLISH_INTERVAL_MS = 200;
const MAX_TICK_MS = 1000;
const MIN_CALIBRATION_SAMPLES = 5;

class PostureEngine {
  private neutral: number | null = DEFAULT_PREFS.neutral;
  private threshold = DEFAULT_PREFS.threshold;
  private holdMs = DEFAULT_PREFS.holdMs;
  private notifications = DEFAULT_PREFS.notifications;

  private face = false;
  private lastFaceAt = 0;
  private smoothed: number | null = null;
  private bend = 0;
  private status: Status = 'no-face';
  private alerting = false;
  private badSince: number | null = null;
  private lastAlertAt = 0;

  private active = false;
  private startedAt = 0;
  private trackedMs = 0;
  private uprightMs = 0;
  private alerts = 0;
  private lastTickAt = 0;
  private lastSession: SessionSummary | null = null;

  private calibrating = false;
  private calibrationStart = 0;
  private calibrationSamples: number[] = [];
  private onCalibrated: ((neutral: number) => void) | null = null;

  private snapshot: Snapshot;
  private listeners = new Set<() => void>();
  private lastPublish = 0;

  constructor() {
    this.snapshot = this.build(Date.now());
  }

  applyPrefs(prefs: Prefs) {
    this.threshold = prefs.threshold;
    this.holdMs = prefs.holdMs;
    this.notifications = prefs.notifications;
    this.neutral = prefs.neutral;
    this.publish(Date.now(), true);
  }

  /** A face pitch in degrees, or null when no face is in frame. */
  reportFace(pitch: number | null) {
    const now = Date.now();

    if (pitch === null) {
      this.clearFace(now);
      return;
    }

    this.face = true;
    this.lastFaceAt = now;

    if (this.calibrating) {
      if (this.calibrationStart === 0) this.calibrationStart = now;
      this.calibrationSamples.push(pitch);
      if (now - this.calibrationStart >= CALIBRATION_MS && this.calibrationSamples.length >= MIN_CALIBRATION_SAMPLES) {
        this.finishCalibration(now);
      } else {
        this.status = 'calibrating';
        this.publish(now);
      }
      return;
    }

    this.smoothed = smooth(this.smoothed, pitch);
    this.bend = this.neutral === null ? 0 : bendFrom(this.neutral, this.smoothed);
    this.status = this.neutral === null ? 'warn' : severityFor(this.bend, this.threshold);
    this.updateAlert(now);
    this.publish(now);
  }

  /** Call on a timer so timers and the face watchdog keep running without new frames. */
  tick() {
    const now = Date.now();

    if (this.face && now - this.lastFaceAt > FACE_TIMEOUT_MS) this.clearFace(now);

    if (this.active) {
      const elapsed = this.lastTickAt === 0 ? 0 : clamp(now - this.lastTickAt, 0, MAX_TICK_MS);
      this.lastTickAt = now;
      if (this.face) {
        this.trackedMs += elapsed;
        if (this.status === 'good') this.uprightMs += elapsed;
      }
    }

    this.publish(now);
  }

  startSession() {
    const now = Date.now();
    this.active = true;
    this.startedAt = now;
    this.lastTickAt = now;
    this.trackedMs = 0;
    this.uprightMs = 0;
    this.alerts = 0;
    this.alerting = false;
    this.badSince = null;
    this.lastAlertAt = 0;
    this.face = false;
    this.smoothed = null;
    this.bend = 0;
    this.status = 'no-face';
    this.publish(now, true);
  }

  endSession() {
    if (!this.active) return;
    const now = Date.now();
    this.lastSession = {
      durationMs: Math.max(0, now - this.startedAt),
      trackedMs: this.trackedMs,
      uprightMs: this.uprightMs,
      alerts: this.alerts,
      endedAt: now,
    };
    this.active = false;
    this.publish(now, true);
  }

  startCalibration(onCalibrated: (neutral: number) => void) {
    this.calibrating = true;
    this.onCalibrated = onCalibrated;
    this.calibrationSamples = [];
    this.calibrationStart = 0;
    this.face = false;
    this.lastFaceAt = 0;
    this.smoothed = null;
    this.alerting = false;
    this.badSince = null;
    this.status = 'calibrating';
    this.publish(Date.now(), true);
  }

  cancelCalibration() {
    if (!this.calibrating) return;
    this.calibrating = false;
    this.onCalibrated = null;
    this.calibrationSamples = [];
    this.calibrationStart = 0;
    this.status = this.neutral === null ? 'no-face' : 'good';
    this.publish(Date.now(), true);
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = () => this.snapshot;

  private finishCalibration(now: number) {
    const neutral = Math.round(median(this.calibrationSamples) * 10) / 10;
    this.calibrating = false;
    this.calibrationSamples = [];
    this.calibrationStart = 0;
    this.neutral = neutral;
    this.smoothed = neutral;
    this.bend = 0;
    this.status = 'good';
    const done = this.onCalibrated;
    this.onCalibrated = null;
    this.publish(now, true);
    done?.(neutral);
  }

  private clearFace(now: number) {
    if (!this.face && now - this.lastFaceAt <= FACE_TIMEOUT_MS) return;
    this.face = false;
    this.smoothed = null;
    this.bend = 0;
    if (!this.calibrating) this.status = 'no-face';
    this.badSince = null;
    this.alerting = false;
    this.publish(now);
  }

  private updateAlert(now: number) {
    if (
      !this.active ||
      this.neutral === null ||
      this.status === 'good' ||
      this.status === 'no-face' ||
      this.status === 'calibrating'
    ) {
      this.badSince = null;
      this.alerting = false;
      return;
    }
    if (this.badSince === null) this.badSince = now;
    if (now - this.badSince >= this.holdMs && now - this.lastAlertAt >= ALERT_COOLDOWN_MS) {
      this.lastAlertAt = now;
      this.alerts += 1;
      this.alerting = true;
      if (this.notifications) this.notify();
    }
  }

  private notify() {
    schedulePostureAlert().catch(() => {
      // Notifications are optional; permission denials must not crash a session.
    });
  }

  private publish(now: number, force = false) {
    if (!force && now - this.lastPublish < PUBLISH_INTERVAL_MS) return;
    this.lastPublish = now;
    this.snapshot = this.build(now);
    this.listeners.forEach((listener) => listener());
  }

  private build(now: number): Snapshot {
    return {
      status: this.status,
      bend: this.bend,
      pitch: this.face ? this.smoothed : null,
      face: this.face,
      alerting: this.alerting,
      calibrating: this.calibrating,
      calibrationProgress:
        this.calibrating && this.calibrationStart > 0
          ? clamp((now - this.calibrationStart) / CALIBRATION_MS, 0, 1)
          : 0,
      durationMs: this.active ? Math.max(0, now - this.startedAt) : (this.lastSession?.durationMs ?? 0),
      trackedMs: this.trackedMs,
      uprightMs: this.uprightMs,
      alerts: this.alerts,
      active: this.active,
      neutral: this.neutral,
      threshold: this.threshold,
      lastSession: this.lastSession,
    };
  }
}

export const engine = new PostureEngine();

export function usePosture(): Snapshot {
  return useSyncExternalStore(engine.subscribe, engine.getSnapshot, engine.getSnapshot);
}
