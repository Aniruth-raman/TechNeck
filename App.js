import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, SafeAreaView, Button, Image } from 'react-native';
import { useEffect, useRef, useState } from 'react';
import { Camera } from 'expo-camera';
import { shareAsync } from 'expo-sharing';
import * as MediaLibrary from 'expo-media-library';
import { PoseDetector } from '@mediapipe/pose';

export default function App() {
  let cameraRef = useRef();
  const [hasCameraPermission, setHasCameraPermission] = useState();
  const [hasAudioPermission, setHasAudioPermission] = useState();
  const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = useState();
  const [photo, setPhoto] = useState();
  const [video, setVideo] = useState();

  useEffect(() => {
    (async () => {
      const cameraStatus = await Camera.requestCameraPermissionsAsync();
      const micstatus = await Camera.requestMicrophonePermissionsAsync();
      setHasCameraPermission(cameraStatus.status === 'granted');
      setHasAudioPermission(micstatus.status === 'granted');
      const mediaStatus = await MediaLibrary.requestPermissionsAsync();
      setHasMediaLibraryPermission(mediaStatus.status === 'granted');
    })();
  }, []);

  if (hasCameraPermission === undefined) {
    return <Text>Requesting permissions...</Text>;
  }

  if (!hasCameraPermission) {
    return <Text>Permission for camera not granted. Please allow the app to access the camera.</Text>;
  }

  const startRecording = async () => {
    if (cameraRef.current) {
      try {
        const options = { quality: '720p' };
        const newVideo = await cameraRef.current.recordAsync(options);
        setVideo(newVideo);
      } catch (err) {
        console.warn(err);
      }
    }
  };

  const stopRecording = async () => {
    if (cameraRef.current) {
      cameraRef.current.stopRecording();
    }
  };

  const shareMedia = async (media) => {
    if (media) {
      try {
        await shareAsync(media.uri);
      } catch (err) {
        console.warn(err);
      }
    }
  };

  const saveMediaToLibrary = async (media) => {
    if (media && hasMediaLibraryPermission) {
      try {
        await MediaLibrary.saveToLibraryAsync(media.uri);
      } catch (err) {
        console.warn(err);
      }
    }
  };

  const discardMedia = () => {
    setPhoto(undefined);
    setVideo(undefined);
  };

  if (video) {
    return (
      <SafeAreaView style={styles.container}>
        <Image style={styles.preview} source={{ uri: video.uri }} />
        <Button title="Share" onPress={() => shareMedia(video)} />
        {hasMediaLibraryPermission ? (
          <Button title="Save" onPress={() => saveMediaToLibrary(video)} />
        ) : undefined}
        <Button title="Discard" onPress={discardMedia} />
      </SafeAreaView>
    );
  }

  return (
    <Camera style={styles.container} ref={cameraRef}>
      <View style={styles.buttonContainer}>
        <Button title="Record" onPress={startRecording} />
        <Button title="Stop Recording" onPress={stopRecording} />
      </View>
      <StatusBar style="auto" />
    </Camera>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonContainer: {
    backgroundColor: '#fff',
    alignSelf: 'flex-end'
  },
  preview: {
    alignSelf: 'stretch',
    flex: 1
  }
});
