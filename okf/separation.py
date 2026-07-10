"""
Stem separation, two tiers:

- `separate_desktop()` — real Demucs (PyTorch), 4 true stems
  (vocals/drums/bass/other). Desktop only (Windows/Linux/macOS): torch has
  no Android build and no python-for-android recipe exists for it, so this
  path is unreachable on Android and is never imported by the Android UI
  branch. See okf/ALGORITHM_NOTES.md for why.
- `separate_dsp()` — pure-NumPy fallback used on Android (and available on
  desktop too, e.g. as a fast preview). Produces:
    - "harmonic" / "percussive" via median-filtering HPSS
    - "instrumental" / "vocals_estimate" via center-channel phase
      cancellation (classic "karaoke" trick: content panned dead-center,
      typically lead vocal, cancels out of L-R; instrumental survives)
  This is NOT Demucs-quality and is labeled as such in the UI -- it's a
  real, working DSP technique, not a placeholder.
"""

import os
import wave

import numpy as np

from okf.audio_analysis import load_wav


def is_torch_available():
    try:
        import torch  # noqa: F401
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def separate_desktop(filepath, out_dir, model_name="htdemucs"):
    """
    Real 4-stem separation via Demucs. Desktop only.

    Requires `torch` + `demucs` (see requirements-desktop.txt). Raises
    RuntimeError with a clear message if they're missing rather than
    silently falling back, so the caller can decide whether to retry with
    separate_dsp() instead.
    """
    if not is_torch_available():
        raise RuntimeError(
            "Demucs/torch not installed. Desktop stem separation requires "
            "`pip install -r requirements-desktop.txt`. On Android, use "
            "separate_dsp() instead -- Demucs cannot run on-device there."
        )

    from demucs.separate import main as demucs_main

    os.makedirs(out_dir, exist_ok=True)
    demucs_main([
        "-n", model_name,
        "-o", out_dir,
        filepath,
    ])

    stem_dir = os.path.join(out_dir, model_name, os.path.splitext(os.path.basename(filepath))[0])
    stems = {}
    for name in ("vocals", "drums", "bass", "other"):
        path = os.path.join(stem_dir, f"{name}.wav")
        if os.path.exists(path):
            stems[name] = path
    return stems


def _median_filter_axis(S, win, axis):
    """Sliding-window median filter along one axis, pure NumPy (no scipy)."""
    if win <= 1:
        return S.copy()
    pad = win // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    padded = np.pad(S, pad_width, mode="reflect")

    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=win, axis=axis)
    return np.median(windows, axis=-1)


def _hpss(S_mag, harmonic_win=17, percussive_win=17):
    """
    Median-filtering harmonic/percussive source separation.

    Harmonic content is smooth across time (long sustained pitches) and
    spiky across frequency; percussive content is the opposite (broadband,
    short in time). Median-filtering the spectrogram along each axis and
    comparing which filtered version dominates at each bin is the standard
    HPSS technique (Fitzgerald 2010 / librosa's `effects.hpss`).
    """
    harmonic_mag = _median_filter_axis(S_mag, harmonic_win, axis=1)
    percussive_mag = _median_filter_axis(S_mag, percussive_win, axis=0)

    total = harmonic_mag + percussive_mag + 1e-8
    harmonic_mask = harmonic_mag / total
    percussive_mask = percussive_mag / total

    return S_mag * harmonic_mask, S_mag * percussive_mask


def _istft(S_complex, hop_length=512, n_fft=2048, length=None):
    n_frames = S_complex.shape[1]
    window = np.hanning(n_fft).astype(np.float32)
    out_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(out_len, dtype=np.float64)
    win_sum = np.zeros(out_len, dtype=np.float64)

    frames_time = np.fft.irfft(S_complex, n=n_fft, axis=0)
    for i in range(n_frames):
        start = i * hop_length
        y[start:start + n_fft] += frames_time[:, i] * window
        win_sum[start:start + n_fft] += window ** 2

    win_sum[win_sum < 1e-8] = 1e-8
    y = y / win_sum
    if length is not None:
        y = y[:length]
    return y.astype(np.float32)


def _stft_complex(y, n_fft=2048, hop_length=512):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_fft, n_frames),
        strides=(y.strides[0], y.strides[0] * hop_length),
    ).copy()
    frames *= window[:, None]
    return np.fft.rfft(frames, axis=0)


def _write_wav(path, y, sr):
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def separate_dsp(filepath, out_dir, n_fft=2048, hop_length=512):
    """
    NumPy-only separation, works on Android and desktop alike.

    Returns paths to: harmonic.wav, percussive.wav, instrumental.wav,
    vocals_estimate.wav
    """
    os.makedirs(out_dir, exist_ok=True)
    stems = {}

    y_stereo, sr = load_wav(filepath, target_sr=None, mono=False)
    y_mono = y_stereo.mean(axis=1) if y_stereo.ndim > 1 else y_stereo

    S_complex = _stft_complex(y_mono, n_fft, hop_length)
    S_mag = np.abs(S_complex)
    phase = S_complex / (S_mag + 1e-8)

    harmonic_mag, percussive_mag = _hpss(S_mag)
    harmonic_y = _istft(harmonic_mag * phase, hop_length, n_fft, length=len(y_mono))
    percussive_y = _istft(percussive_mag * phase, hop_length, n_fft, length=len(y_mono))

    base = os.path.splitext(os.path.basename(filepath))[0]
    harmonic_path = os.path.join(out_dir, f"{base}_harmonic.wav")
    percussive_path = os.path.join(out_dir, f"{base}_percussive.wav")
    _write_wav(harmonic_path, harmonic_y, sr)
    _write_wav(percussive_path, percussive_y, sr)
    stems["harmonic"] = harmonic_path
    stems["percussive"] = percussive_path

    if y_stereo.ndim > 1 and y_stereo.shape[1] >= 2:
        left, right = y_stereo[:, 0], y_stereo[:, 1]
        n = min(len(left), len(right))
        instrumental = left[:n] - right[:n]  # cancels center-panned content
        vocals_estimate = (left[:n] + right[:n]) / 2.0 - instrumental * 0.5

        instrumental_path = os.path.join(out_dir, f"{base}_instrumental.wav")
        vocals_path = os.path.join(out_dir, f"{base}_vocals_estimate.wav")
        _write_wav(instrumental_path, instrumental, sr)
        _write_wav(vocals_path, vocals_estimate, sr)
        stems["instrumental"] = instrumental_path
        stems["vocals_estimate"] = vocals_path

    return stems
