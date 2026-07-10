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
