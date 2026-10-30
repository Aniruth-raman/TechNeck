import Ionicons from '@expo/vector-icons/Ionicons';
import { LinearGradient } from 'expo-linear-gradient';
import { useKeepAwake } from 'expo-keep-awake';
import { memo, useEffect } from 'react';
import { Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { Camera as FaceCamera, type Face } from 'react-native-vision-camera-face-detector';
import { Gauge } from '../components/Gauge';
import { Button, Pill } from '../components/ui';
import { engine, usePosture, type Status } from '../lib/engine';
import { formatClock, GAUGE_MAX, uprightPercent } from '../lib/posture';
import { colors, radius, severityColor } from '../theme';

type Props = {
  mode: 'monitor' | 'calibrate';
  onExit: () => void;
  onCalibrated: (neutral: number) => void;
};

const STATUS: Record<Status, { label: string; caption: string; color: string; field: string }> = {
  good: { label: 'Upright', caption: 'Neck is upright', color: colors.mint, field: colors.mintSoft },
  warn: { label: 'Past limit', caption: 'Ease back', color: colors.amber, field: colors.amberSoft },
  bad: { label: 'Neck bent', caption: 'Lift your head', color: colors.coral, field: colors.coralSoft },
  'no-face': { label: 'No face', caption: 'Align your face', color: colors.faint, field: 'rgba(255,255,255,0.06)' },
  calibrating: { label: 'Calibrating', caption: 'Hold still', color: colors.mint, field: colors.mintSoft },
};

function handleFaces(faces: Face[]) {
  engine.reportFace(faces.length > 0 ? faces[0].pitchAngle : null);
}

function handleCameraError(error: Error) {
  console.warn('[TechNeck] face detection failed', error);
}

export function MonitorScreen({ mode, onExit, onCalibrated }: Props) {
  useKeepAwake();

  useEffect(() => {
    if (mode === 'calibrate') {
      engine.startCalibration(onCalibrated);
      return () => engine.cancelCalibration();
    }
    engine.startSession();
    return () => engine.endSession();
  }, [mode, onCalibrated]);

  useEffect(() => {
    const id = setInterval(() => engine.tick(), 250);
    return () => clearInterval(id);
  }, []);

  return (
    <View style={styles.root}>
      <CameraPane />
      {mode === 'calibrate' ? <CalibrationHud onExit={onExit} /> : <MonitoringHud onExit={onExit} />}
    </View>
  );
}

const CameraPane = memo(function CameraPane() {
  const { hasPermission, canRequestPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');

  useEffect(() => {
    if (!hasPermission && canRequestPermission) void requestPermission();
  }, [hasPermission, canRequestPermission, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.fallback}>
        <Ionicons name="videocam-off-outline" size={30} color={colors.muted} />
        <Text style={styles.fallbackTitle}>Camera access is off</Text>
        <Text style={styles.fallbackCopy}>TechNeck needs the front camera to measure your neck angle.</Text>
        <Button
          label="Open system settings"
          variant="ghost"
          onPress={() => Linking.openSettings().catch(() => {})}
          style={styles.fallbackButton}
        />
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.fallback}>
        <Ionicons name="camera-outline" size={30} color={colors.muted} />
        <Text style={styles.fallbackTitle}>No front camera found</Text>
        <Text style={styles.fallbackCopy}>This device cannot run the posture check.</Text>
      </View>
    );
  }

  return (
    <FaceCamera
      style={StyleSheet.absoluteFill}
      device={device}
      isActive
      resizeMode="cover"
      cameraFacing="front"
      performanceMode="fast"
      runLandmarks={false}
      runContours={false}
      runClassifications={false}
      minFaceSize={0.2}
      onFacesDetected={handleFaces}
      onError={handleCameraError}
    />
  );
});

function MonitoringHud({ onExit }: { onExit: () => void }) {
  const state = usePosture();
  const insets = useSafeAreaInsets();
  const tone = STATUS[state.status];
  const hasFace = state.face;
  const upright = uprightPercent(state.uprightMs, state.trackedMs);

  return (
    <View style={styles.overlay} pointerEvents="box-none">
      <LinearGradient
        pointerEvents="none"
        colors={['rgba(7,11,16,0.94)', 'rgba(7,11,16,0)']}
        style={styles.topScrim}
      />
      <LinearGradient
        pointerEvents="none"
        colors={['rgba(7,11,16,0)', 'rgba(7,11,16,0.96)']}
        style={styles.bottomScrim}
      />

      <View style={[styles.topBar, { paddingTop: insets.top + 10 }]}>
        <Pressable
          onPress={onExit}
          style={({ pressed }) => [styles.circleButton, pressed && styles.pressed]}
          accessibilityRole="button"
          accessibilityLabel="Stop monitoring"
        >
          <Ionicons name="chevron-down" size={22} color={colors.text} />
        </Pressable>
        <View style={styles.timerPill}>
          <View style={[styles.liveDot, { backgroundColor: tone.color }]} />
          <Text style={styles.timerText}>{formatClock(state.durationMs)}</Text>
        </View>
        <View style={styles.circleButtonSpacer} />
      </View>

      <View style={styles.gaugeArea} pointerEvents="none">
        <View style={styles.gaugeWrap}>
          <Gauge
            progress={hasFace ? state.bend / GAUGE_MAX : 0}
            color={tone.color}
            value={hasFace ? String(Math.round(state.bend)) : '--'}
            caption={tone.caption}
          />
        </View>
        <Pill label={tone.label} color={tone.color} background={tone.field} />
        <Text style={styles.limitText}>
          {hasFace ? `Limit ${state.threshold}° · smoothed live` : 'Center your face in the frame'}
        </Text>
      </View>

      <View style={[styles.bottomArea, { paddingBottom: insets.bottom + 16 }]}>
        {state.alerting && (
          <View style={styles.alert}>
            <Ionicons name="arrow-up-circle" size={22} color={colors.coral} />
            <View style={styles.alertText}>
              <Text style={styles.alertTitle}>Head up</Text>
              <Text style={styles.alertCopy}>You've been bent forward. Roll your shoulders back.</Text>
            </View>
          </View>
        )}

        <View style={styles.metrics}>
          <Metric label="Upright" value={hasFace ? `${upright}%` : '—'} />
          <View style={styles.divider} />
          <Metric label="Tracking" value={hasFace ? 'Locked' : 'Searching'} />
          <View style={styles.divider} />
          <Metric label="Nudges" value={String(state.alerts)} />
        </View>

        <Button label="End session" variant="danger" onPress={onExit} />
      </View>
    </View>
  );
}

function CalibrationHud({ onExit }: { onExit: () => void }) {
  const state = usePosture();
  const insets = useSafeAreaInsets();
  const percent = Math.round(state.calibrationProgress * 100);

  return (
    <View style={[styles.calibrationRoot, { paddingTop: insets.top + 10, paddingBottom: insets.bottom + 20 }]}>
      <View style={styles.topBar}>
        <Pressable
          onPress={onExit}
          style={({ pressed }) => [styles.circleButton, pressed && styles.pressed]}
          accessibilityRole="button"
          accessibilityLabel="Cancel calibration"
        >
          <Ionicons name="close" size={22} color={colors.text} />
        </Pressable>
        <Text style={styles.calibrationStep}>CALIBRATION</Text>
        <View style={styles.circleButtonSpacer} />
      </View>

      <View style={styles.calibrationBody}>
        <View style={styles.calibrationTarget}>
          <Ionicons name="scan-outline" size={54} color={colors.mint} />
        </View>
        <Text style={styles.calibrationTitle}>Set your neutral</Text>
        <Text style={styles.calibrationCopy}>
          Hold the phone at eye level, sit tall, and look straight ahead. This becomes your zero.
        </Text>

        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${Math.max(4, percent)}%` }]} />
        </View>

        <Text style={[styles.calibrationState, state.face && styles.calibrationStateActive]}>
          {state.face ? (percent > 0 ? `Hold still… ${percent}%` : 'Hold still…') : 'Waiting for your face…'}
        </Text>
      </View>

      <Button label="Cancel" variant="ghost" onPress={onExit} />
    </View>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricValue}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#000',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  topScrim: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 190,
  },
  bottomScrim: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: 340,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 18,
  },
  circleButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(15,22,33,0.72)',
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  circleButtonSpacer: {
    width: 44,
  },
  pressed: {
    opacity: 0.7,
  },
  timerPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: 'rgba(15,22,33,0.72)',
    borderWidth: 1,
    borderColor: colors.border,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  timerText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  gaugeArea: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  gaugeWrap: {
    borderRadius: 999,
    backgroundColor: 'rgba(7,11,16,0.62)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
    padding: 10,
  },
  limitText: {
    color: colors.faint,
    fontSize: 12,
    fontWeight: '600',
  },
  bottomArea: {
    paddingHorizontal: 18,
    gap: 12,
  },
  alert: {
    flexDirection: 'row',
    gap: 12,
    alignItems: 'center',
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: 'rgba(255,107,107,0.35)',
    backgroundColor: 'rgba(48,17,21,0.92)',
    padding: 14,
  },
  alertText: {
    flex: 1,
    gap: 2,
  },
  alertTitle: {
    color: colors.coral,
    fontSize: 14,
    fontWeight: '700',
  },
  alertCopy: {
    color: 'rgba(255,209,209,0.85)',
    fontSize: 12.5,
    lineHeight: 17,
  },
  metrics: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: 'rgba(15,22,33,0.82)',
    paddingVertical: 14,
  },
  metric: {
    flex: 1,
    alignItems: 'center',
    gap: 3,
  },
  metricValue: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  metricLabel: {
    color: colors.faint,
    fontSize: 10.5,
    fontWeight: '600',
    letterSpacing: 0.7,
    textTransform: 'uppercase',
  },
  divider: {
    width: 1,
    height: 26,
    backgroundColor: colors.border,
  },
  fallback: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    padding: 32,
    backgroundColor: colors.bg,
  },
  fallbackTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '700',
    marginTop: 4,
  },
  fallbackCopy: {
    color: colors.muted,
    fontSize: 13.5,
    lineHeight: 20,
    textAlign: 'center',
  },
  fallbackButton: {
    marginTop: 12,
    minWidth: 220,
  },
  calibrationRoot: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(7,11,16,0.94)',
    paddingHorizontal: 18,
    justifyContent: 'space-between',
  },
  calibrationStep: {
    color: colors.faint,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.4,
  },
  calibrationBody: {
    alignItems: 'center',
    gap: 14,
    paddingHorizontal: 8,
  },
  calibrationTarget: {
    width: 108,
    height: 108,
    borderRadius: 54,
    borderWidth: 1,
    borderColor: 'rgba(59,232,176,0.35)',
    backgroundColor: colors.mintSoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  calibrationTitle: {
    color: colors.text,
    fontSize: 25,
    fontWeight: '700',
    letterSpacing: -0.5,
  },
  calibrationCopy: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
    textAlign: 'center',
  },
  progressTrack: {
    width: '100%',
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.border,
    overflow: 'hidden',
    marginTop: 10,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
    backgroundColor: colors.mint,
  },
  calibrationState: {
    color: colors.faint,
    fontSize: 13,
    fontWeight: '600',
  },
  calibrationStateActive: {
    color: colors.mint,
  },
});
