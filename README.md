# Roots and Weed

Cross-platform (Android + Windows) song **key finder**, **BPM finder**, and
**stem separator**, built with Kivy + KivyMD. Ported and rewritten from the
BPM/key algorithm in [Octave Live Wire](../Octave_live_wire) (Django app),
generalized into a standalone mobile/desktop app.

## What it does

- **Convert tab**: turns an MP3 into a WAV first (the analyzer only reads
  WAV — see below). Desktop shells out to a system `ffmpeg`; Android decodes
  MP3 on-device via `MediaExtractor`/`MediaCodec` (the codec every phone
  already ships with — no ffmpeg binary needed in the APK).
- Detects a song's **BPM** (tempo) and **musical key** from a WAV file, with
  confidence scores and the runner-up key shown when the pick is ambiguous
  (see `okf/ALGORITHM_NOTES.md` for the full reasoning).
- Draws the waveform natively via Kivy `Canvas` (no matplotlib dependency).
- Separates a track into stems:
  - **Desktop**: real 4-stem separation via [Demucs](https://github.com/facebookresearch/demucs) (vocals/drums/bass/other) — best quality, requires `torch`.
  - **Android**: pure-NumPy DSP fallback (harmonic/percussive split +
    center-channel vocal isolation) — a different, lighter quality tier,
    labeled as such in the UI. Real Demucs cannot run on Android: there is
    no PyTorch build for Android and no python-for-android recipe for it.

## Project layout

```
main.py                  # Kivy/KivyMD app entrypoint + UI logic
rootsandweed.kv          # Material Design layout
okf/
  audio_analysis.py       # BPM + key detection (pure NumPy, no librosa)
  separation.py            # Demucs (desktop) + DSP fallback (Android-safe)
  mp3_convert.py            # MP3 -> WAV (ffmpeg on desktop, MediaCodec on Android)
  ALGORITHM_NOTES.md        # Why it's built this way, and what changed vs. the original
buildozer.spec            # Android build config (python-for-android)
requirements-desktop.txt  # Windows/Linux/macOS deps (adds torch + demucs)
.github/workflows/
  build-apk.yml            # Builds the Android APK on every push to main
  build-windows.yml         # Builds a Windows executable on every push to main
```

## Running locally (desktop)

```bash
pip install -r requirements-desktop.txt
python main.py
```

## Building the APK (no Linux machine required)

Buildozer/python-for-android only build on Linux. Since development here is
on Windows, the APK is built by **GitHub Actions** instead:

1. Push to `main` (or run the workflow manually from the Actions tab).
2. The `Build Android APK` workflow runs `buildozer android debug` on an
   Ubuntu runner and uploads the resulting `.apk` as a workflow artifact.
3. Download it from the workflow run's **Artifacts** section.

The first build is slow (~20-40 min) because it bootstraps the whole
Android NDK/SDK toolchain; later builds are faster thanks to caching inside
the `buildozer-action`.

## Building for Windows

The `Build Windows executable` workflow packages the app with PyInstaller
on a `windows-latest` runner and uploads `dist/OKF/` as an artifact. That CI
build intentionally skips `torch`/`demucs` to keep CI fast and reliable — it
will use the DSP separation fallback. For a release build with real Demucs
separation on Windows, run PyInstaller locally after
`pip install -r requirements-desktop.txt`.

## Known limitations

- Only WAV input is decoded by the analyzer itself (stdlib `wave`, zero
  extra dependencies — the safest choice for Android). Use the in-app
  **Convert** tab (or `ffmpeg` directly on desktop) to turn MP3 into WAV
  first.
- Android stem separation is DSP-based, not neural — see
  `okf/ALGORITHM_NOTES.md` for why, and what it actually does.
- Tempo/key detection is a best-effort estimate on any audio-only algorithm;
  confidence scores are surfaced so the UI never asserts false certainty.
