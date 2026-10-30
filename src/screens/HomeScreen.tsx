import Ionicons from '@expo/vector-icons/Ionicons';
import { LinearGradient } from 'expo-linear-gradient';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { formatClock, uprightPercent } from '../lib/posture';
import { usePosture } from '../lib/engine';
import type { Prefs } from '../lib/settings';
import { colors, radius } from '../theme';
import { Button } from '../components/ui';

type Props = {
  prefs: Prefs;
  onStart: () => void;
  onCalibrate: () => void;
  onOpenSettings: () => void;
};

const TIPS = [
  'Prop your phone at eye level, and steady it against something.',
  'Sit so your face fills the top half of the preview.',
  'Recalibrate whenever you move the phone to a new height.',
];

export function HomeScreen({ prefs, onStart, onCalibrate, onOpenSettings }: Props) {
  const { lastSession } = usePosture();
  const calibrated = prefs.neutral !== null;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={styles.headerText}>
            <Text style={styles.brand}>TechNeck</Text>
            <Text style={styles.tagline}>Keep your head above your shoulders</Text>
          </View>
          <Pressable
            onPress={onOpenSettings}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
            accessibilityRole="button"
            accessibilityLabel="Open settings"
          >
            <Ionicons name="options-outline" size={20} color={colors.text} />
          </Pressable>
        </View>

        <LinearGradient
          colors={['#143F35', '#0C1A24', '#0B1219']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.hero}
        >
          <View style={styles.heroBadge}>
            <View style={styles.heroDot} />
            <Text style={styles.heroBadgeLabel}>ON-DEVICE POSTURE GUARD</Text>
          </View>
          <Text style={styles.heroTitle}>Your neck deserves{'\n'}better than a screen angle.</Text>
          <Text style={styles.heroCopy}>
            The front camera measures how far your neck drifts forward and nudges you before the ache sets in.
          </Text>
          <Button
            label={calibrated ? 'Start monitoring' : 'Calibrate & start'}
            onPress={onStart}
            style={styles.heroButton}
          />
          <Text style={styles.heroFoot}>
            {calibrated ? 'About 30 seconds to set up. Your neutral is saved.' : 'Takes 2 seconds. Point the camera at your face.'}
          </Text>
        </LinearGradient>

        <View style={styles.stats}>
          <Stat label="Last session" value={lastSession ? formatClock(lastSession.durationMs) : '—'} />
          <Stat
            label="Upright"
            value={lastSession ? `${uprightPercent(lastSession.uprightMs, lastSession.trackedMs)}%` : '—'}
          />
          <Stat label="Nudges" value={lastSession ? String(lastSession.alerts) : '—'} />
        </View>

        <Pressable
          onPress={onCalibrate}
          style={({ pressed }) => [styles.rowCard, pressed && styles.pressed]}
          accessibilityRole="button"
          accessibilityLabel="Recalibrate neutral angle"
        >
          <View style={styles.rowIcon}>
            <Ionicons name="scan-outline" size={18} color={colors.mint} />
          </View>
          <View style={styles.rowText}>
            <Text style={styles.rowTitle}>Neutral angle</Text>
            <Text style={styles.rowHint}>
              {calibrated ? `Calibrated at ${prefs.neutral?.toFixed(1)}° of head pitch` : 'Not set yet — takes one look ahead'}
            </Text>
          </View>
          <Text style={styles.rowAction}>{calibrated ? 'Recalibrate' : 'Set up'}</Text>
        </Pressable>

        <View style={styles.tips}>
          <Text style={styles.sectionLabel}>Before you start</Text>
          {TIPS.map((tip, index) => (
            <View key={tip} style={styles.tip}>
              <Text style={styles.tipNumber}>{index + 1}</Text>
              <Text style={styles.tipText}>{tip}</Text>
            </View>
          ))}
        </View>

        <Text style={styles.privacy}>
          Detection runs fully on-device with ML Kit. Footage is never recorded, stored or uploaded.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  content: {
    padding: 20,
    paddingBottom: 36,
    gap: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 2,
  },
  headerText: {
    gap: 2,
  },
  brand: {
    color: colors.text,
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: -0.6,
  },
  tagline: {
    color: colors.muted,
    fontSize: 12.5,
  },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pressed: {
    opacity: 0.7,
  },
  hero: {
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(59,232,176,0.18)',
    padding: 22,
    gap: 12,
  },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  heroDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.mint,
  },
  heroBadgeLabel: {
    color: colors.mint,
    fontSize: 10.5,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
  heroTitle: {
    color: colors.text,
    fontSize: 27,
    lineHeight: 33,
    fontWeight: '700',
    letterSpacing: -0.7,
  },
  heroCopy: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  heroButton: {
    marginTop: 8,
  },
  heroFoot: {
    color: colors.faint,
    fontSize: 11.5,
    textAlign: 'center',
  },
  stats: {
    flexDirection: 'row',
    gap: 10,
  },
  stat: {
    flex: 1,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    paddingVertical: 16,
    paddingHorizontal: 12,
    gap: 4,
  },
  statValue: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  statLabel: {
    color: colors.faint,
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  rowCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    padding: 16,
  },
  rowIcon: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: colors.mintSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rowText: {
    flex: 1,
    gap: 3,
  },
  rowTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '600',
  },
  rowHint: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 17,
  },
  rowAction: {
    color: colors.mint,
    fontSize: 12.5,
    fontWeight: '700',
  },
  tips: {
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    padding: 18,
    gap: 14,
  },
  sectionLabel: {
    color: colors.faint,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  tip: {
    flexDirection: 'row',
    gap: 12,
    alignItems: 'flex-start',
  },
  tipNumber: {
    color: colors.mint,
    fontSize: 12,
    fontWeight: '800',
    width: 18,
    marginTop: 1,
  },
  tipText: {
    flex: 1,
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  privacy: {
    color: colors.faint,
    fontSize: 11.5,
    lineHeight: 17,
    textAlign: 'center',
    paddingHorizontal: 8,
  },
});
