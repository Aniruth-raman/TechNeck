import type { ReactNode } from 'react';
import { Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from 'react-native';
import { colors, radius } from '../theme';

type PillProps = {
  label: string;
  color: string;
  background: string;
};

export function Pill({ label, color, background }: PillProps) {
  return (
    <View style={[styles.pill, { backgroundColor: background, borderColor: color }]}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={[styles.pillLabel, { color }]}>{label}</Text>
    </View>
  );
}

type ButtonProps = {
  label: string;
  onPress: () => void;
  variant?: 'solid' | 'ghost' | 'danger';
  style?: StyleProp<ViewStyle>;
  children?: ReactNode;
};

export function Button({ label, onPress, variant = 'solid', style }: ButtonProps) {
  const palette =
    variant === 'solid'
      ? { backgroundColor: colors.mint, borderColor: colors.mint }
      : variant === 'danger'
        ? { backgroundColor: colors.coralSoft, borderColor: 'rgba(255,107,107,0.4)' }
        : { backgroundColor: 'rgba(255,255,255,0.04)', borderColor: colors.border };
  const textColor = variant === 'solid' ? colors.bg : variant === 'danger' ? colors.coral : colors.text;

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      onPress={onPress}
      style={({ pressed }) => [styles.button, palette, pressed && styles.pressed, style]}
    >
      <Text style={[styles.buttonLabel, { color: textColor }]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 999,
    borderWidth: 1,
    alignSelf: 'center',
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  pillLabel: {
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  button: {
    minHeight: 54,
    borderRadius: radius.sm,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  pressed: {
    opacity: 0.75,
  },
  buttonLabel: {
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
