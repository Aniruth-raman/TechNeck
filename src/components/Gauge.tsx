import { StyleSheet, Text, View } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import { clamp } from '../lib/posture';
import { colors } from '../theme';

type Props = {
  progress: number;
  color: string;
  value: string;
  unit?: string;
  caption: string;
  size?: number;
};

export function Gauge({ progress, color, value, unit = '°', caption, size = 250 }: Props) {
  const stroke = 14;
  const radius = (size - stroke) / 2 - 8;
  const center = size / 2;
  const circumference = 2 * Math.PI * radius;
  const filled = clamp(progress, 0, 1) * circumference;
  const dash = filled <= 1 ? 0 : filled;

  return (
    <View style={{ width: size, height: size }}>
      <Svg width={size} height={size}>
        <Circle cx={center} cy={center} r={radius - stroke} fill="rgba(255,255,255,0.03)" />
        <Circle cx={center} cy={center} r={radius} stroke={colors.border} strokeWidth={stroke} fill="none" />
        <Circle
          cx={center}
          cy={center}
          r={radius}
          stroke={color}
          strokeOpacity={0.22}
          strokeWidth={stroke + 10}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          rotation={-90}
          origin={`${center}, ${center}`}
          fill="none"
        />
        <Circle
          cx={center}
          cy={center}
          r={radius}
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          rotation={-90}
          origin={`${center}, ${center}`}
          fill="none"
        />
      </Svg>
      <View style={styles.center}>
        <View style={styles.readout}>
          <Text style={[styles.value, { color }]}>{value}</Text>
          <Text style={styles.unit}>{unit}</Text>
        </View>
        <Text style={styles.caption}>{caption}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  center: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  readout: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  value: {
    fontSize: 62,
    fontWeight: '700',
    letterSpacing: -2,
    fontVariant: ['tabular-nums'],
  },
  unit: {
    color: colors.muted,
    fontSize: 22,
    fontWeight: '600',
    marginTop: 12,
    marginLeft: 2,
  },
  caption: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },
});
