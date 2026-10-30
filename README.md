# TechNeck

On-device posture monitor for phone-heavy days. The front camera measures how far your head drifts forward from the neutral position you calibrate once, then nudges you when the bend is sustained.

## How it works

1. **Calibrate** — hold the phone at eye level, look straight ahead for ~1.5s. That head pitch becomes your zero.
2. **Monitor** — ML Kit reports face pitch on every camera frame. TechNeck smooths it, removes the neutral offset, and compares the result to your limit.
3. **Nudge** — once the bend lasts past your limit for the configured delay, you get a visual alert and an optional haptic. A 20s cooldown keeps it from nagging.

There is no bundled `.tflite` model anymore. Detection runs through Google ML Kit's native face detector via `react-native-vision-camera-face-detector`, which is maintained for VisionCamera 5 and the new architecture. Frames are processed in memory only; nothing is recorded or uploaded. The app has no network permissions or code.

## Run it

```sh
npm install
npx expo prebuild
npm run android   # or: npm run ios
```

A native dev build is required (Expo Go cannot load VisionCamera, Nitro or ML Kit).

```sh
npm run check     # typecheck + posture math self-test
npx expo-doctor   # dependency checks
```

## Security

`npm audit` is clean. Build-time advisories (shell-quote, brace-expansion, node-forge, js-yaml, source-map-js, compression, zod) are pinned to patched versions through `overrides`.

`deepmerge` has no upstream fix for CVE-2026-93753 / GHSA-gcqx-w48m-w2h4, so `vendor/deepmerge` is a patched copy of 4.3.1 (key denylist, `Object.defineProperty` writes, fail-closed `in` check) installed as a `file:` dependency. It is only reached by `@expo/cli` when merging app config; nothing in the app bundle imports it. `npm run check` exercises the guard, and `vendor/deepmerge/README.md` says when to drop the local copy.
