import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import * as SystemUI from 'expo-system-ui';
import { SettingsSheet } from './src/components/SettingsSheet';
import { engine } from './src/lib/engine';
import { loadPrefs, savePrefs, type Prefs } from './src/lib/settings';
import { HomeScreen } from './src/screens/HomeScreen';
import { MonitorScreen } from './src/screens/MonitorScreen';
import { colors } from './src/theme';

type Screen = 'home' | 'calibrate' | 'monitor';

export default function App() {
  const [prefs, setPrefs] = useState<Prefs | null>(null);
  const [screen, setScreen] = useState<Screen>('home');
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    let mounted = true;
    void loadPrefs().then((loaded) => {
      if (mounted) setPrefs(loaded);
    });
    SystemUI.setBackgroundColorAsync(colors.bg).catch(() => {});
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (prefs) engine.applyPrefs(prefs);
  }, [prefs]);

  const updatePrefs = useCallback((patch: Partial<Prefs>) => {
    setPrefs((current) => {
      if (!current) return current;
      const next = { ...current, ...patch };
      void savePrefs(next);
      return next;
    });
  }, []);

  const handleCalibrated = useCallback(
    (neutral: number) => {
      updatePrefs({ neutral });
      setScreen('monitor');
    },
    [updatePrefs],
  );

  const goHome = useCallback(() => setScreen('home'), []);
  const start = useCallback(
    () => setScreen(prefs?.neutral === null ? 'calibrate' : 'monitor'),
    [prefs?.neutral],
  );
  const calibrate = useCallback(() => setScreen('calibrate'), []);

  if (!prefs) {
    return (
      <View style={styles.loading}>
        <StatusBar style="light" />
        <ActivityIndicator color={colors.mint} />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      {screen === 'home' ? (
        <HomeScreen
          prefs={prefs}
          onStart={start}
          onCalibrate={calibrate}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      ) : (
        <MonitorScreen
          mode={screen === 'calibrate' ? 'calibrate' : 'monitor'}
          onExit={goHome}
          onCalibrated={handleCalibrated}
        />
      )}
      <SettingsSheet
        visible={settingsOpen}
        prefs={prefs}
        onChange={updatePrefs}
        onClose={() => setSettingsOpen(false)}
      />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  loading: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
