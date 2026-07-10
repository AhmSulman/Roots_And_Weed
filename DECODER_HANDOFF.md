# Decoder Handoff — pick up in VS Code

_Last updated 2026-07-10. Context for continuing the MP3-support work._

## The decision (why this differs from `okf/mp3_convert.py`)

`mp3_convert.py` uses the **MP3 → WAV file → reload** model: convert to a temp WAV,
then `load_wav()` reads it back. It works, but it round-trips through disk and keeps
"MP3" as a special second-class path.

New model — the one to implement — matches how the pipeline actually thinks:

> **read file → decode to one common in-memory PCM stream → operate.**

A single loader always hands the analyzer the same thing: a `float32` mono array in
`[-1, 1]` plus a sample rate. Format (WAV / MP3) is decided _inside* the decoder, in
memory, no temp file. numpy never sees a file. Two decoders, one per machine, same
output contract.

Key correction from the discussion: _pydub is not a fix__ — for MP3 it shells out to
the ffmpeg binary, so it inherits the exact "ffmpeg missing → crash" problem. That's
why desktop uses ffmpeg _directly and honestly_ (with a clear error if absent), and
Android avoids ffmpeg entirely via the OS codec.

## Output contract (both files must obey)

```python
def decode(path, target_sr=22050, mono=True) -> tuple[np.ndarray, int]:
    # returns (y, sr):  y = float32 in [-1, 1], shape (n,) if mono; sr = int
```

- `.wav` → decode with stdlib `wave` (zero dependencies, always works).
- anything else (`.mp3`, `.m4a`, …) → platform decoder, in memory.

---

## File 1 — `okf/decode_desktop.py`  (Windows / Linux / macOS)

WAV via stdlib; everything else piped straight from ffmpeg as raw PCM on stdout
(`pipe:1`, `f32le`) — no temp WAV, ffmpeg also does the resample/downmix.

```python
"""Desktop audio decoder -> common in-memory PCM stream (y float32, sr).

WAV: stdlib `wave` (no dependency). MP3/other: pipe raw PCM out of the system
ffmpeg binary -- no temp file, ffmpeg handles resample + mono downmix.
pydub is deliberately NOT used: it just wraps this same ffmpeg call.
"""

import os
import shutil
import subprocess

import numpy as np

from okf.audio_analysis import load_wav


class DecodeError(Exception):
    pass


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def decode(path, target_sr=22050, mono=True):
    if path.lower().endswith(".wav"):
        return load_wav(path, target_sr=target_sr, mono=mono)
    return _decode_ffmpeg(path, target_sr, mono)


def _decode_ffmpeg(path, target_sr, mono):
    if not ffmpeg_available():
        raise DecodeError(
            "ffmpeg was not found on PATH. Install it (winget install ffmpeg, "
            "or https://ffmpeg.org/download.html) -- desktop MP3 decoding shells "
            "out to it instead of bundling a decoder."
        )
    if not os.path.exists(path):
        raise DecodeError(f"File not found: {path}")

    cmd = [
        "ffmpeg", "-v", "quiet", "-i", path,
        "-ac", "1" if mono else "2",
        "-ar", str(target_sr),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DecodeError(f"ffmpeg failed: {proc.stderr[-800:].decode(errors='replace')}")

    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()  # already in [-1, 1]
    if not mono:  # interleaved -> (n, 2)
        y = y.reshape(-1, 2)
    return y, target_sr
```

---

## File 2 — `okf/decode_android.py`  (Android APK)

WAV via stdlib; MP3/other via `MediaExtractor` + `MediaCodec` (pyjnius) — the codec
every phone ships. This is the in-memory version of `_convert_android` from
`mp3_convert.py`: it returns the PCM array instead of writing a WAV.

```python
"""Android audio decoder -> common in-memory PCM stream (y float32, sr).

WAV: stdlib `wave`. MP3/other: Android's own MediaExtractor/MediaCodec via
pyjnius -- no ffmpeg, no extra APK size. Returns PCM in memory (no temp WAV).
"""

import numpy as np

from okf.audio_analysis import load_wav, _resample


class DecodeError(Exception):
    pass


def decode(path, target_sr=22050, mono=True):
    if path.lower().endswith(".wav"):
        return load_wav(path, target_sr=target_sr, mono=mono)
    return _decode_android(path, target_sr, mono)


def _decode_android(path, target_sr, mono):
    try:
        from jnius import autoclass
    except ImportError as e:
        raise DecodeError(
            "pyjnius unavailable -- this path only runs inside a built APK, "
            "not a desktop interpreter. Use okf.decode_desktop there."
        ) from e

    MediaExtractor = autoclass("android.media.MediaExtractor")
    MediaCodec = autoclass("android.media.MediaCodec")
    MediaFormat = autoclass("android.media.MediaFormat")
    BufferInfo = autoclass("android.media.MediaCodec$BufferInfo")

    extractor = MediaExtractor()
    extractor.setDataSource(path)

    track_index, audio_format = -1, None
    for i in range(extractor.getTrackCount()):
        fmt = extractor.getTrackFormat(i)
        mime = fmt.getString(MediaFormat.KEY_MIME)
        if mime and mime.startswith("audio/"):
            track_index, audio_format = i, fmt
            break
    if track_index < 0:
        raise DecodeError("No audio track found in file")

    extractor.selectTrack(track_index)
    mime = audio_format.getString(MediaFormat.KEY_MIME)
    src_sr = audio_format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
    src_channels = audio_format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

    codec = MediaCodec.createDecoderByType(mime)
    codec.configure(audio_format, None, None, 0)
    codec.start()

    TIMEOUT_US = 10000
    buf_info = BufferInfo()
    pcm_chunks = []
    input_done = output_done = False

    while not output_done:
        if not input_done:
            in_index = codec.dequeueInputBuffer(TIMEOUT_US)
            if in_index >= 0:
                in_buffer = codec.getInputBuffer(in_index)
                sample_size = extractor.readSampleData(in_buffer, 0)
                if sample_size < 0:
                    codec.queueInputBuffer(in_index, 0, 0, 0,
                                           MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    input_done = True
                else:
                    codec.queueInputBuffer(in_index, 0, sample_size,
                                           extractor.getSampleTime(), 0)
                    extractor.advance()

        out_index = codec.dequeueOutputBuffer(buf_info, TIMEOUT_US)
        if out_index >= 0:
            out_buffer = codec.getOutputBuffer(out_index)
            chunk = bytearray(buf_info.size)
            out_buffer.get(chunk)
            pcm_chunks.append(bytes(chunk))
            codec.releaseOutputBuffer(out_index, False)
            if buf_info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM:
                output_done = True

    codec.stop(); codec.release(); extractor.release()

    y = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16).astype(np.float32) / 32768.0
    if src_channels > 1:
        y = y.reshape(-1, src_channels)
        if mono:
            y = y.mean(axis=1)
    if target_sr and src_sr != target_sr:
        y = _resample(y, src_sr, target_sr)
        src_sr = target_sr
    return y.astype(np.float32), src_sr
```

---

## Wiring — `okf/audio_analysis.py`

Add one dispatcher and route `analyze_file` through it. `load_wav` stays untouched
(the decoders and the tests still call it).

```python
from kivy.utils import platform  # add near the top

def load_audio(path, target_sr=22050, mono=True):
    """Platform-correct decode -> common (y, sr) PCM stream, any supported format."""
    if platform == "android":
        from okf.decode_android import decode
    else:
        from okf.decode_desktop import decode
    return decode(path, target_sr=target_sr, mono=mono)
```

Then in `AudioAnalyzer.analyze_file`, change:

""_python
y, sr = load_wav(filepath, target_sr=target_sr, mono=True)

to:
python
y, sr = load_audio(filepath, target_sr=target_sr, mono=True)
""

## Wiring — `main.py` (UI)

Now MP3 can be analyzed directly; the separate Convert tab becomes optional.

- `open_wav_chooser` / file-manager `ext`: allow `[".wav", ".mp3"]` (and `.m4a` if wanted).
- `status_text` default: change "Pick a WAV file" → "Pick a WAV or MP3 file to begin."
- Keep `mp3_convert.py` and the Convert tab as-is for now (still works); it's just no
  longer the only way in. Remove later if you want.

## Test before pushing

1. `python -m pytest` — existing 49 tests must stay green (WAV path unchanged).
2. Desktop with ffmpeg installed: analyze an `.mp3` → BPM/key/waveform populate.
3. Desktop **without** ffmpeg: analyze `.mp3` → clean `DecodeError` message, no crash;
   `.wav` still works.
4. Android decoder can't be exercised on desktop (needs pyjnius/APK) — verify on device
   or in the built APK.

## Open question to decide in VS Code

Whether to retire `okf/mp3_convert.py` + the Convert tab once `load_audio` lands, or
keep them. The overlap now; the in-memory decoders make the file-conversion path
redundant for _analysis_, but conversion is still handy if you want a saved WAV.
