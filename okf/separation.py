"""
Stem separation, two tiers, both returning exactly two stems:
"vocals" and "instrumental".

- `separate_desktop()` — real Demucs (PyTorch) via the model API (vocals +
  everything-else summed). Desktop only (Windows/Linux/macOS): torch has
  no Android build and no python-for-android recipe exists for it, so this
  path is unreachable on Android and is never imported by the Android UI
  branch. See okf/ALGORITHM_NOTES.md for why.
- `separate_dsp()` — pure-NumPy fallback used on Android (and available on
  desktop too, e.g. as a fast preview). Estimates "instrumental" /
  "vocals" via center-channel phase cancellation (classic "karaoke"
  trick: content panned dead-center, typically lead vocal, cancels out of
  L-R; instrumental survives). Requires stereo input for that reason.
  This is NOT Demucs-quality and is labeled as such in the UI -- it's a
  real, working DSP technique, not a placeholder.

The HPSS helpers (`_hpss`, `_median_filter_axis`, STFT/ISTFT) are kept:
they're tested, dependency-free building blocks still useful for previews.
"""

import json
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from okf.audio_analysis import load_audio

STEMS_INDEX_FILENAME = "stems_index.json"

# save_stems_metadata does a read-modify-write of the whole index file;
# without a lock, concurrent separations would silently drop each other's
# entries (last writer wins).
_INDEX_LOCK = threading.Lock()


def is_torch_available():
    try:
        import torch  # noqa: F401
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def separate_desktop(filepath, out_dir, model_name="htdemucs"):
    """
    Real vocals/instrumental separation via Demucs. Desktop only.

    Calls the Demucs model API directly instead of the CLI: audio goes in
    through our own `load_audio` and stems come out through our stdlib
    WAV writer. That deliberately bypasses torchaudio's load/save, which
    in recent versions delegates to the optional `torchcodec` package and
    crashes when it's absent (the CLI hits exactly that after minutes of
    separating). Vocals are one stem; every other model source is summed
    into "instrumental".

    On CPU, segments are processed on `num_workers` parallel threads.
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

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    os.makedirs(out_dir, exist_ok=True)
    model = get_model(model_name)
    model.eval()

    y, sr = load_audio(filepath, target_sr=model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)  # model expects stereo
    wav = torch.from_numpy(np.ascontiguousarray(y.T.astype(np.float32)))  # (2, n)

    # normalize like the demucs CLI does, undo on the way out
    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) + 1e-8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cap CPU workers at 2: each parallel segment holds its own activations
    # (~GBs for htdemucs), and 3+ workers can OOM-crash on a full-length song
    n_jobs = 1 if device == "cuda" else min(2, max(1, (os.cpu_count() or 2) - 1))
    with torch.no_grad():
        sources = apply_model(
            model, ((wav - mean) / std)[None],
            device=device, num_workers=n_jobs,
        )[0]
    sources = sources * std + mean

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]
    instrumental = sources.sum(0) - vocals

    base = os.path.splitext(os.path.basename(filepath))[0]
    stems = {}
    for name, tensor in (("vocals", vocals), ("instrumental", instrumental)):
        path = os.path.join(out_dir, f"{base}_{name}.wav")
        _write_wav(path, tensor.cpu().numpy().T, model.samplerate)
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
    """Write mono (n,) or stereo (n, 2) float [-1, 1] as 16-bit PCM WAV."""
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    n_channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())  # row-major (n, ch) is already interleaved


def separate_dsp(filepath, out_dir, target_sr=44100, progress=None):
    """
    NumPy-only vocals/instrumental separation, works on Android and
    desktop alike. Returns paths to: vocals.wav, instrumental.wav.

    Uses center-channel phase cancellation, which needs a stereo source:
    a mono file has no L/R difference to exploit, so it raises ValueError
    with a clear message instead of fabricating a result.

    `progress`, if given, is called with a fraction in [0, 1] as each
    stage completes (decode dominates the runtime). The two stem files
    are written on parallel threads.
    """
    def report(fraction):
        if progress:
            progress(fraction)

    os.makedirs(out_dir, exist_ok=True)
    report(0.05)

    y_stereo, sr = load_audio(filepath, target_sr=target_sr, mono=False)
    report(0.55)
    if y_stereo.ndim < 2 or y_stereo.shape[1] < 2:
        raise ValueError(
            "Vocal separation (DSP mode) needs a stereo file: it isolates "
            "vocals by cancelling the center channel between left and right, "
            "and a mono file has no left/right difference to work with."
        )

    left, right = y_stereo[:, 0], y_stereo[:, 1]
    n = min(len(left), len(right))
    if np.allclose(left[:n], right[:n], atol=1e-4):
        # mono source upmixed to two identical channels (e.g. mono MP3 via
        # ffmpeg -ac 2): L-R would cancel everything, not just the vocals
        raise ValueError(
            "Vocal separation (DSP mode) needs a true stereo file: this one "
            "has identical left/right channels, so center-channel "
            "cancellation would remove the whole mix, not just the vocals."
        )
    instrumental = left[:n] - right[:n]  # cancels center-panned content
    vocals = (left[:n] + right[:n]) / 2.0 - instrumental * 0.5
    report(0.7)

    base = os.path.splitext(os.path.basename(filepath))[0]
    instrumental_path = os.path.join(out_dir, f"{base}_instrumental.wav")
    vocals_path = os.path.join(out_dir, f"{base}_vocals.wav")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_write_wav, instrumental_path, instrumental, sr),
            pool.submit(_write_wav, vocals_path, vocals, sr),
        ]
        for f in futures:
            f.result()  # re-raise any write error
    report(1.0)
    return {"vocals": vocals_path, "instrumental": instrumental_path}


# -- stems metadata index (JSON) ----------------------------------------------
#
# okf_output/stems_index.json maps each separated source file to its stem
# paths plus how it was made. Lets the app hand back existing stems
# instantly instead of re-separating, and survives restarts. Entries are
# invalidated when the source file changes (size/mtime) or a stem file
# was deleted.

def _index_path(out_dir):
    return os.path.join(out_dir, STEMS_INDEX_FILENAME)


def _load_index(out_dir):
    try:
        with open(_index_path(out_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _fingerprint(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def save_stems_metadata(out_dir, source_path, stems, mode, elapsed_seconds):
    """Record a finished separation in the index; returns the entry."""
    entry = {
        "stems": stems,
        "mode": mode,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": _fingerprint(source_path),
    }
    with _INDEX_LOCK:
        index = _load_index(out_dir)
        index[os.path.abspath(source_path)] = entry
        tmp = _index_path(out_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp, _index_path(out_dir))
    return entry


def load_stems_metadata(out_dir, source_path):
    """Return the index entry for source_path, or None if stale/absent."""
    entry = _load_index(out_dir).get(os.path.abspath(source_path))
    if not entry:
        return None
    try:
        if _fingerprint(source_path) != entry.get("source"):
            return None  # source file changed since separation
    except OSError:
        return None
    if not all(os.path.exists(p) for p in entry.get("stems", {}).values()):
        return None  # a stem file was deleted
    return entry


def separate_with_cache(filepath, out_dir, desktop_mode, progress=None):
    """
    Separation front door: reuse indexed stems when the source is
    unchanged, otherwise separate, time it, and record the metadata.

    Returns (stems, metadata_entry, from_cache).
    """
    cached = load_stems_metadata(out_dir, filepath)
    if cached:
        if progress:
            progress(1.0)
        return cached["stems"], cached, True

    t0 = time.perf_counter()
    if desktop_mode:
        stems = separate_desktop(filepath, out_dir)
    else:
        stems = separate_dsp(filepath, out_dir, progress=progress)
    elapsed = time.perf_counter() - t0

    mode = "demucs" if desktop_mode else "dsp"
    entry = save_stems_metadata(out_dir, filepath, stems, mode, elapsed)
    return stems, entry, False
