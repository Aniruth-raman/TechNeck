import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, Text, View, Dimensions, Platform, ToastAndroid, Alert } from 'react-native';

import { Camera } from 'expo-camera';

import * as tf from '@tensorflow/tfjs';
import * as posedetection from '@tensorflow-models/pose-detection';
import * as ScreenOrientation from 'expo-screen-orientation';
import {
  bundleResourceIO,
  cameraWithTensors,
} from '@tensorflow/tfjs-react-native';
import Svg, { Circle } from 'react-native-svg';
import { ExpoWebGLRenderingContext } from 'expo-gl';
import { CameraType } from 'expo-camera/build/Camera.types';

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === 'android';
const IS_IOS = Platform.OS === 'ios';

// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get('window').width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.3;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc). For best result, use the size that
// doesn't distort the image.
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// Whether to auto-render TensorCamera preview.
const AUTO_RENDER = false;

// Whether to load model from app bundle (true) or through network (false).
const LOAD_MODEL_FROM_BUNDLE = true;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<posedetection.PoseDetector>();
  const [poses, setPoses] = useState<posedetection.Pose[]>();
  const [fps, setFps] = useState(0);
  const [aligned, setAligned] = useState(false);
  const [orientation, setOrientation] =
    useState<ScreenOrientation.Orientation>();
  const [cameraType, setCameraType] = useState<CameraType>(
    CameraType.front
  );
  const [color, setColor] = useState('green');
  // Use `useRef` so that changing it won't trigger a re-render.
  //
  // - null: unset (initial value).
  // - 0: animation frame/loop has been canceled.
  // - >0: animation frame has been scheduled.
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Set initial orientation.
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);

      // Listens to orientation change.
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });

      // Camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
      const movenetModelConfig: posedetection.MoveNetModelConfig = {
        modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
      };
      if (LOAD_MODEL_FROM_BUNDLE) {
        const modelJson = require('./offline_model/model.json');
        const modelWeights1 = require('./offline_model/group1-shard1of2.bin');
        const modelWeights2 = require('./offline_model/group1-shard2of2.bin');
        movenetModelConfig.modelUrl = bundleResourceIO(modelJson, [
          modelWeights1,
          modelWeights2,
        ]);
      }
      const model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        movenetModelConfig
      );
      setModel(model);

      // Ready!
      setTfReady(true);
    }

    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value as tf.Tensor3D;
      if (model && tfReady && imageTensor != null) {
        const startTime = performance.now();
        const poses = await model.estimatePoses(imageTensor,
          undefined,
          Date.now()
          //   {
          //   flipHorizontal: Platform.OS === 'ios' ? false : true,
          //   // decodingMethod: 'single-person',
          // }
        );
        setPoses(poses);

        // Compute fps.
        const currFps = 1000 / (Date.now() - startTime);
        setFps(Math.floor(currFps));

        // Detect poor body posture.
        if (poses && poses.length > 0) {
          const pose = poses[0];
          const keypoints = pose.keypoints;
          const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
          const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');
          if (leftShoulder && rightShoulder) {
            if (findDistance(leftShoulder.x, leftShoulder.y, rightShoulder.x, rightShoulder.y) < 100) {
              // notifyMessage("Shoulders not aligned");
              setAligned(true);
            } else {
              // notifyMessage("Shoulders not aligned");
              setAligned(false);
            }
          }
          const leftEar = keypoints.find((kp) => kp.name === 'left_ear');
          const rightEar = keypoints.find((kp) => kp.name === 'right_ear');
          const leftHip = keypoints.find((kp) => kp.name === 'left_hip');
          const rightHip = keypoints.find((kp) => kp.name === 'right_hip');


          // const leftEye = keypoints.find((kp) => kp.name === 'left_eye');
          // const rightEye = keypoints.find((kp) => kp.name === 'right_eye');
          const nose = keypoints.find((kp) => kp.name === 'nose');

          if (cameraType === CameraType.front && leftEar && rightEar && nose) {
            const earsMidpoint = {
              x: (leftEar.x + rightEar.x) / 2,
              y: (leftEar.y + rightEar.y) / 2,
            };

            const dist = findDistance(earsMidpoint.x, earsMidpoint.y, nose.x, nose.y);
            // console.log("Distance:"+dist);
            if (dist < 10) {
              setColor('green');
              setAligned(true);
            } else {
              setColor('red');
              setAligned(false);
            }
          }

          if (cameraType === CameraType.back && leftShoulder && rightShoulder && leftEar && rightEar && leftHip && rightHip) {
            const neckAngle = findAngle(
              leftShoulder.x,
              leftShoulder.y,
              // rightShoulder.x,
              // rightShoulder.y,
              leftEar.x,
              leftEar.y,
              // rightEar.x,
              // rightEar.y
            );
            const torsoAngle = findAngle(leftHip.x, leftHip.y, leftShoulder.x, leftShoulder.y);
            // console.log("Angle:"+angle)
            // console.log("Neck Angle:" + neckAngle + ",Torso Angle:" + torsoAngle);
            if (neckAngle < 40 && torsoAngle < 10) {
              // Set the SVG color to green.
              setColor('green');
              setAligned(true);
            } else {
              // Set the SVG color to red.
              setColor('red');
              setAligned(false);
            }
          }
        }
      }
      tf.dispose([imageTensor]);

      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false.
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      rafId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  const renderPose = () => {
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((k) => {
          // Flip horizontally on android or when using back camera on iOS.
          const flipX = IS_ANDROID || cameraType === CameraType.back;
          const x = flipX ? getOutputTensorWidth() - k.x : k.x;
          const y = k.y;
          const cx =
            (x / getOutputTensorWidth()) *
            (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
          const cy =
            (y / getOutputTensorHeight()) *
            (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={cx}
              cy={cy}
              r='4'
              strokeWidth='2'
              // fill='#00AA00'
              fill={color}
              stroke='white'
            />
          );
        });

      return <Svg style={styles.svg}>{keypoints}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {fps}</Text>
      </View>
    );
  };
  const renderAligned = () => {
    return (
      <View style={styles.textContainer}>
        <Text>{aligned ? "" : "Not "}Aligned</Text>
      </View>
    );
  };

  const renderCameraTypeSwitcher = () => {
    return (
      <View
        style={styles.cameraTypeSwitcher}
        onTouchEnd={handleSwitchCameraType}
      >
        <Text>
          Switch to{' '}
          {cameraType === CameraType.front ? 'back' : 'front'} camera
        </Text>
      </View>
    );
  };

  const handleSwitchCameraType = () => {
    if (cameraType === CameraType.front) {
      setCameraType(CameraType.back);
    } else {
      setCameraType(CameraType.front);
    }
  };

  const isPortrait = () => {
    return (
      orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
      orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
    );
  };

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor to
    // get better result. Without this, the image stored in the output tensor
    // would be stretched too much.
    //
    // Same for getOutputTensorHeight below.
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };

  const getTextureRotationAngleInDegrees = () => {
    // On Android, the camera texture will rotate behind the scene as the phone
    // changes orientation, so we don't need to rotate it in TensorCamera.
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS, the camera texture won't rotate automatically. Calculate the
    // rotation angles here which will be passed to TensorCamera to rotate it
    // internally.
    switch (orientation) {
      // Not supported on iOS as of 11/2021, but add it here just in case.
      case ScreenOrientation.Orientation.PORTRAIT_DOWN:
        return 180;
      case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
        return cameraType === CameraType.front ? 270 : 90;
      case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
        return cameraType === CameraType.front ? 90 : 270;
      default:
        return 0;
    }
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      // Note that you don't need to specify `cameraTextureWidth` and
      // `cameraTextureHeight` prop in `TensorCamera` below.
      <View
        style={
          isPortrait() ? styles.containerPortrait : styles.containerLandscape
        }
      >
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          autorender={AUTO_RENDER}
          type={cameraType}
          // tensor related props
          resizeWidth={getOutputTensorWidth()}
          resizeHeight={getOutputTensorHeight()}
          resizeDepth={3}
          rotation={getTextureRotationAngleInDegrees()}
          onReady={handleCameraStream} useCustomShadersToResize={false} cameraTextureWidth={0} cameraTextureHeight={0} />
        {renderPose()}
        {renderFps()}
        {renderAligned()}
        {renderCameraTypeSwitcher()}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  containerPortrait: {
    position: 'relative',
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: 'relative',
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    width: '100%',
    height: '100%',
    zIndex: 1,
  },
  svg: {
    width: '100%',
    height: '100%',
    position: 'absolute',
    zIndex: 30,
  },
  fpsContainer: {
    position: 'absolute',
    top: 10,
    left: 10,
    width: 80,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 180,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  textContainer: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    // alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    // zIndex: 20,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: (Platform.OS === 'ios') ? 20 : 0
  },
});
// function findAngle(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, x4: number, y4: number) {
//   const vector1 = { x: x2 - x1, y: y2 - y1 };
//   const vector2 = { x: x4 - x3, y: y4 - y3 };
//   const dotProduct = vector1.x * vector2.x + vector1.y * vector2.y;
//   const magnitude1 = Math.sqrt(vector1.x ** 2 + vector1.y ** 2);
//   const magnitude2 = Math.sqrt(vector2.x ** 2 + vector2.y ** 2);
//   const angle = Math.acos(dotProduct / (magnitude1 * magnitude2));
//   return angle * (180 / Math.PI);
// }
function findAngle(x1: number, y1: number, x2: number, y2: number) {
  // const vector1 = { x: x2 - x1, y: y2 - y1 };
  // const vector2 = { x: x4 - x3, y: y4 - y3 };
  // const dotProduct = vector1.x * vector2.x + vector1.y * vector2.y;
  // const magnitude1 = Math.sqrt(vector1.x ** 2 + vector1.y ** 2);
  // const magnitude2 = Math.sqrt(vector2.x ** 2 + vector2.y ** 2);
  const angle = Math.acos((y2 - y1) * (-y1) / (Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1));
  return Math.floor(angle * (180 / Math.PI));
}

function findDistance(x1: number, y1: number, x2: number, y2: number) {
  const dist = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return Math.floor(dist);
}

function notifyMessage(msg: string) {
  if (Platform.OS === 'android') {
    ToastAndroid.show(msg, ToastAndroid.SHORT)
  } else {
    Alert.alert(msg);
  }
}

