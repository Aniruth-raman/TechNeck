import { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission]);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // intentionally empty for now — just confirming frame processors run
  }, []);

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text>Waiting for camera permission…</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text>No camera device found.</Text>
      </View>
    );
  }

  return (
    <Camera
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={true}
      frameProcessor={frameProcessor}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});