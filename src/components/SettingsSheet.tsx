import { Modal, Pressable, StyleSheet, Switch, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { HOLD_OPTIONS, THRESHOLD_OPTIONS, type Prefs } from '../lib/settings';
import { colors, radius } from '../theme';

type Props = {
  visible: boolean;
  prefs: Prefs;
  onChange: (patch: Partial<Prefs>) => void;
  onClose: () => void;
};

const SENSITIVITY = [
  { value: THRESHOLD_OPTIONS[2], label: 'Strict', hint: '15°' },
  { value: THRESHOLD_OPTIONS[1], label: 'Standard', hint: '20°' },
  { value: THRESHOLD_OPTIONS[0], label: 'Relaxed', hint: '25°' },
];

const HOLDS = [
  { value: HOLD_OPTIONS[0], label: '4s' },
  { value: HOLD_OPTIONS[1], label: '8s' },
  { value: HOLD_OPTIONS[2], label: '15s' },
];

export function SettingsSheet({ visible, prefs, onChange, onClose }: Props) {
  const insets = useSafeAreaInsets();

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.root}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} accessibilityLabel="Close settings" />
        <View style={[styles.sheet, { paddingBottom: insets.bottom + 22 }]}>
          <View style={styles.handle} />
          <View style={styles.header}>
            <View>
              <Text style={styles.title}>Settings</Text>
              <Text style={styles.subtitle}>Tune how strictly TechNeck watches you.</Text>
            </View>
            <Pressable onPress={onClose} style={styles.close} accessibilityRole="button" accessibilityLabel="Close">
              <Text style={styles.closeLabel}>Done</Text>
            </Pressable>
          </View>

          <Text style={styles.label}>Neck bend limit</Text>
          <View style={styles.segmented}>
            {SENSITIVITY.map((option) => {
              const selected = prefs.threshold === option.value;
              return (
                <Pressable
                  key={option.value}
                  onPress={() => onChange({ threshold: option.value })}
                  accessibilityRole="button"
                  accessibilityState={{ selected }}
                  style={[styles.segment, selected && styles.segmentSelected]}
                >
                  <Text style={[styles.segmentLabel, selected && styles.segmentLabelSelected]}>{option.label}</Text>
                  <Text style={[styles.segmentHint, selected && styles.segmentHintSelected]}>{option.hint}</Text>
                </Pressable>
              );
            })}
          </View>

          <Text style={styles.label}>Alert after</Text>
          <View style={styles.segmented}>
            {HOLDS.map((option) => {
              const selected = prefs.holdMs === option.value;
              return (
                <Pressable
                  key={option.value}
                  onPress={() => onChange({ holdMs: option.value })}
                  accessibilityRole="button"
                  accessibilityState={{ selected }}
                  style={[styles.segment, selected && styles.segmentSelected]}
                >
                  <Text style={[styles.segmentLabel, selected && styles.segmentLabelSelected]}>{option.label}</Text>
                </Pressable>
              );
            })}
          </View>

          <View style={styles.toggleRow}>
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>Haptic nudge</Text>
              <Text style={styles.toggleHint}>A short buzz when you need to lift your head.</Text>
            </View>
            <Switch
              value={prefs.haptics}
              onValueChange={(haptics) => onChange({ haptics })}
              trackColor={{ false: colors.border, true: 'rgba(59,232,176,0.45)' }}
              thumbColor={prefs.haptics ? colors.mint : colors.muted}
            />
          </View>

          <Text style={styles.note}>
            Everything runs on-device with ML Kit. No camera frames are recorded or uploaded.
          </Text>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  sheet: {
    backgroundColor: colors.card,
    borderTopLeftRadius: radius.lg,
    borderTopRightRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 22,
    paddingTop: 10,
    gap: 12,
  },
  handle: {
    width: 42,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.border,
    alignSelf: 'center',
    marginBottom: 8,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  title: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '700',
  },
  subtitle: {
    color: colors.muted,
    fontSize: 13,
    marginTop: 4,
  },
  close: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: colors.mintSoft,
  },
  closeLabel: {
    color: colors.mint,
    fontSize: 13,
    fontWeight: '700',
  },
  label: {
    color: colors.faint,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
    marginTop: 10,
  },
  segmented: {
    flexDirection: 'row',
    gap: 8,
  },
  segment: {
    flex: 1,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: 'rgba(255,255,255,0.02)',
    paddingVertical: 12,
    alignItems: 'center',
    gap: 2,
  },
  segmentSelected: {
    borderColor: colors.mint,
    backgroundColor: colors.mintSoft,
  },
  segmentLabel: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '600',
  },
  segmentLabelSelected: {
    color: colors.mint,
    fontWeight: '700',
  },
  segmentHint: {
    color: colors.faint,
    fontSize: 11,
  },
  segmentHintSelected: {
    color: 'rgba(59,232,176,0.7)',
  },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
    marginTop: 10,
    padding: 16,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: 'rgba(255,255,255,0.02)',
  },
  toggleText: {
    flex: 1,
    gap: 3,
  },
  toggleTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '600',
  },
  toggleHint: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 17,
  },
  note: {
    color: colors.faint,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 8,
  },
});
