# Algorithm notes: porting `audio_analyser.py` → `audio_analysis.py`

Thought process behind the rewrite, kept separate from the code so it doesn't
rot into stale comments.

## Why not just reuse the original file as-is

The original (`Octave_live_wire/backing_track_creator/audio_analyser.py`) is a
clean, correct Django-side implementation, but it leans on `librosa`, which
transitively pulls in `numba` (JIT compiler), `scipy`, `soundfile`,
`audioread`, and `matplotlib`. None of these have dependable
python-for-android recipes:

- `numba` needs LLVM at build time — no Android recipe exists, and building
  one is a multi-week project on its own.
- `scipy` *has* a p4a recipe but it's fragile and frequently breaks across
  NDK/API-level combinations.
- `soundfile`/`audioread` expect `libsndfile`/`ffmpeg`/`gstreamer` to be
  present on the system — none of that exists inside an Android app sandbox.
- `matplotlib` can theoretically be recipe-built for Android but it's heavy
  (fonts, backends) for a feature that's just "draw a waveform," which Kivy's
  own `Canvas`/`Line` can do natively and far more cheaply.

Given the goal is one codebase that builds for both Android (buildozer) and
Windows, keeping librosa as a hard dependency means the Android build either
fails outright or becomes a fragile, unmaintainable pile of custom p4a
recipes. So the analyzer was rewritten on NumPy alone — the one heavy
dependency that *does* have a stable, official p4a recipe.

## What changed algorithmically (not just "removed a library")

**1. BPM — octave-error correction.**
The original calls `librosa.beat.beat_track` directly and trusts its output.
Beat trackers (including librosa's) are well known to occasionally lock onto
half or double the perceptual tempo — e.g. a song that's "really" 140 BPM
autocorrelates just as strongly (sometimes more strongly) at 70 BPM, because
every other beat is also a local onset peak. The rewrite computes the
autocorrelation of the onset-strength envelope, picks the strongest peak in a
wide 50–220 BPM window, then explicitly re-scores the `1x`, `2x`, and `0.5x`
candidates against the same autocorrelation curve and keeps whichever the
data actually supports best. This is a standard mitigation in tempo-tracking
literature (Percival & Tzanetakis-style octave correction), and it's cheap:
three lookups against an array we already computed.

**2. Key detection — correlation instead of dot product.**
The original scores the Krumhansl-Schmuckler profiles with a raw dot product:
`np.dot(chroma_vector, profile)`. That's a bug-adjacent shortcut: the
Krumhansl-Schmuckler method is defined over the *correlation* between the
(mean-centered) chroma distribution and each key profile, not the raw inner
product. A dot product is biased toward whichever profile has the larger
overall magnitude/energy — for our profiles that's a mild but real bias
towards the tonic pitch class carrying more weight than it should relative to
the *shape* of the profile. Switching to Pearson correlation
(mean-center both vectors, normalize by their norms) is the textbook
approach and empirically more accurate on ambiguous material.

**3. Key confidence + alternate key (new).**
The original returns only the winning key with no notion of how confident
that pick was. Major/minor relative keys (e.g. C Major / A Minor) share 6 of
7 scale tones, so chroma-only detection is *structurally* unable to
distinguish them from pitch content alone — the correlation gap between the
top two candidates for such pieces is small by construction, not because the
algorithm is failing. Surfacing `key_confidence` (score gap between 1st and
2nd place) and `alternate_key` (the runner-up) turns a silent coin-flip into
visible information the UI can show the user ("F# Minor, or possibly A
Major") instead of asserting false certainty.

**4. WAV-only input, on purpose.**
Dropping `audioread`/`soundfile` means MP3/FLAC/etc. aren't decoded
in-process anymore. Rather than pull in a heavier decode path (which reopens
the Android build-fragility problem), the analyzer expects WAV in, and the
app's **Convert** tab (`okf/mp3_convert.py`) handles turning MP3 into WAV
before analysis: a system `ffmpeg` subprocess on desktop, and Android's own
`MediaExtractor`/`MediaCodec` (via pyjnius) on-device — the same hardware/
software MP3 decoder every phone already ships with, so no ffmpeg binary or
extra native dependency has to travel inside the APK.

## What's intentionally *not* claimed to be equivalent

This is not a bit-for-bit reimplementation of librosa's `chroma_stft` or
`beat_track` — it's a from-scratch STFT → onset-envelope → autocorrelation →
chroma-folding pipeline built to match the same conceptual pipeline with
NumPy-only primitives. On clean, rhythmically steady material the two should
agree closely; on ambiguous or highly syncopated material they may disagree
in ways neither is definitively "more correct" about — tempo/key estimation
from audio alone is inherently a best-effort signal, not ground truth.

## Stem separation follow-on (see `separation.py`)

The same constraint (no numba/scipy/torch on Android) is why stem separation
is split into two tiers rather than one implementation:

- **Desktop**: real Demucs (PyTorch) — best quality, four true stems
  (vocals/drums/bass/other), because torch installs fine via pip on
  Windows/Linux/macOS.
- **Android**: a DSP fallback (median-filter harmonic/percussive split +
  center-channel phase-cancellation vocal isolation) — genuinely functional,
  built from NumPy only, but a different quality tier and explicitly labeled
  as such in the UI rather than pretending it's Demucs-equivalent.

There is no way to get real Demucs *inside* the APK today: PyTorch has no
official Android build and no python-for-android recipe exists for it. Any
tool claiming otherwise for a buildozer/p4a project would either be lying
about what ships in the APK or shipping a multi-hundred-MB experimental
build unlikely to survive a GitHub Actions CI run.
