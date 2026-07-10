"""
MP3 -> WAV conversion, one implementation per platform.

The analyzer (okf/audio_analysis.py) only decodes WAV on purpose -- see
ALGORITHM_NOTES.md for why (no librosa/audioread/soundfile, so Android
actually builds). That means MP3 input needs an explicit conversion step
before analysis, which is what this module -- and the "Convert" tab in the
UI -- is for.

- Desktop (Windows/Linux/macOS): shell out to a system `ffmpeg` binary.
  Simple, reliable, and ffmpeg is a one-time install outside the app.
- Android: there is no bundled ffmpeg and no reliable way to ship one
  through buildozer. Instead this uses the device's own hardware/software
  MP3 decoder via `android.media.MediaExtractor` + `MediaCodec` (through
  pyjnius), which every Android device already has -- no extra native
  dependency, no APK size cost.
"""

import os
import shutil
import subprocess
import wave

from kivy.utils import platform


class ConversionError(Exception):
    pass


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def convert_mp3_to_wav(input_path, output_path, target_sr=22050):
    """Convert an MP3 file to a mono WAV file at output_path."""
    if platform == "android":
        _convert_android(input_path, output_path, target_sr)
    else:
        _convert_desktop(input_path, output_path, target_sr)
    return output_path


def _convert_desktop(input_path, output_path, target_sr):
    if not ffmpeg_available():
        raise ConversionError(
            "ffmpeg was not found on PATH. Install it (e.g. from "
            "https://ffmpeg.org/download.html, or `winget install ffmpeg` "
            "on Windows) and try again -- OKF shells out to it for MP3 "
            "decoding on desktop instead of bundling a decoder."
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", str(target_sr),
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ConversionError(f"ffmpeg failed: {proc.stderr[-800:]}")


def _convert_android(input_path, output_path, target_sr):
    """
    Decode MP3 -> PCM using Android's built-in MediaExtractor/MediaCodec,
    then resample to target_sr and write a WAV. No ffmpeg, no extra APK
    size -- this is the codec every Android phone already ships with.
    """
    try:
        from jnius import autoclass
    except ImportError as e:
        raise ConversionError(
            "pyjnius is unavailable -- this path only runs inside a built "
            "Android APK, not in a desktop Python interpreter."
        ) from e

    import numpy as np

    from okf.audio_analysis import _resample

    MediaExtractor = autoclass("android.media.MediaExtractor")
    MediaCodec = autoclass("android.media.MediaCodec")
    MediaFormat = autoclass("android.media.MediaFormat")
    ByteBuffer = autoclass("java.nio.ByteBuffer")  # noqa: F401 (kept for clarity)

    extractor = MediaExtractor()
    extractor.setDataSource(input_path)

    track_index = -1
    audio_format = None
    for i in range(extractor.getTrackCount()):
        fmt = extractor.getTrackFormat(i)
        mime = fmt.getString(MediaFormat.KEY_MIME)
        if mime and mime.startswith("audio/"):
            track_index = i
            audio_format = fmt
            break

    if track_index < 0:
        raise ConversionError("No audio track found in file")

    extractor.selectTrack(track_index)
    mime = audio_format.getString(MediaFormat.KEY_MIME)
    src_sr = audio_format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
    src_channels = audio_format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

    codec = MediaCodec.createDecoderByType(mime)
    codec.configure(audio_format, None, None, 0)
    codec.start()

    TIMEOUT_US = 10000
    pcm_chunks = []
    input_done = False
    output_done = False

    buf_info = autoclass("android.media.MediaCodec$BufferInfo")()

    while not output_done:
        if not input_done:
            in_index = codec.dequeueInputBuffer(TIMEOUT_US)
            if in_index >= 0:
                in_buffer = codec.getInputBuffer(in_index)
                sample_size = extractor.readSampleData(in_buffer, 0)
                if sample_size < 0:
                    codec.queueInputBuffer(in_index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    input_done = True
                else:
                    codec.queueInputBuffer(in_index, 0, sample_size, extractor.getSampleTime(), 0)
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

    codec.stop()
    codec.release()
    extractor.release()

    pcm_bytes = b"".join(pcm_chunks)
    y = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if src_channels > 1:
        y = y.reshape(-1, src_channels).mean(axis=1)

    if target_sr and src_sr != target_sr:
        y = _resample(y, src_sr, target_sr)
        src_sr = target_sr

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(src_sr)
        wf.writeframes(pcm16.tobytes())
