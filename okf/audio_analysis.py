"""
Pure NumPy BPM + musical-key detector.

Ported from the original librosa-based implementation in
Octave_live_wire/backing_track_creator/audio_analyser.py, then rebuilt on
raw NumPy (STFT, onset envelope, autocorrelation tempo, chroma) instead of
librosa. That drops librosa/numba/scipy/audioread/soundfile as hard
dependencies -- none of them have reliable python-for-android recipes, so
this file is what makes the Android build possible at all. It also fixes
two real accuracy issues in the original:

1. Tempo octave errors: naive autocorrelation/beat trackers commonly lock
   onto 2x or 0.5x the true BPM (e.g. reporting 70 instead of 140). This
   version explicitly re-scores the 1x/2x/0.5x candidates against the
   autocorrelation curve and keeps whichever has the strongest support.
2. Key detection used a raw dot product against the Krumhansl-Schmuckler
   profiles, which is biased by the chroma vector's absolute energy.
   Textbook Krumhansl-Schmuckler compares *correlation* (mean-centered),
   which is what's implemented here -- plus a confidence score (gap
   between the best and second-best key) so a caller can tell a confident
   "F# Minor" from a coin-flip between two relative keys.

Only WAV input is supported natively (via the stdlib `wave` module) so
there are zero third-party decoding dependencies. Desktop callers that
need MP3/FLAC/etc. can pre-convert with ffmpeg/pydub before calling in.
"""

import wave

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (Krumhansl & Kessler, 1982).
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

FLAT_EQUIVALENTS = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
}


def load_wav(path, target_sr=22050, mono=True):
    """Decode a WAV file to a float32 NumPy array in [-1, 1], resampled to target_sr."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        y = y.reshape(-1, n_channels)
        if mono:
            y = y.mean(axis=1)

    if target_sr and sr != target_sr:
        y = _resample(y if y.ndim == 1 else y, sr, target_sr)
        sr = target_sr

    return y.astype(np.float32), sr


def _resample(y, sr_from, sr_to):
    if sr_from == sr_to or len(y) == 0:
        return y
    duration = len(y) / sr_from
    n_target = max(1, int(round(duration * sr_to)))
    x_old = np.linspace(0.0, duration, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
    if y.ndim == 1:
        return np.interp(x_new, x_old, y).astype(np.float32)
    return np.stack(
        [np.interp(x_new, x_old, y[:, c]) for c in range(y.shape[1])], axis=1
    ).astype(np.float32)


def _stft_magnitude(y, n_fft=2048, hop_length=512):
    """Magnitude spectrogram, shape (n_fft // 2 + 1, n_frames)."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop_length

    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_fft, n_frames),
        strides=(y.strides[0], y.strides[0] * hop_length),
    ).copy()
    frames *= window[:, None]

    spec = np.fft.rfft(frames, axis=0)
    return np.abs(spec)


def _onset_envelope(S_mag, sr, hop_length):
    """Spectral-flux onset strength envelope, one value per STFT frame."""
    S_db = 20.0 * np.log10(S_mag + 1e-6)
    flux = np.diff(S_db, axis=1)
    flux = np.maximum(flux, 0.0)
    onset_env = flux.sum(axis=0)
    onset_env = np.concatenate([[0.0], onset_env])

    onset_env = onset_env - onset_env.min()
    peak = onset_env.max()
    if peak > 0:
        onset_env = onset_env / peak
    return onset_env


def _estimate_tempo(onset_env, sr, hop_length, bpm_range=(50, 220)):
    """Autocorrelation-based tempo with octave-error correction."""
    frame_rate = sr / hop_length

    ac = np.correlate(onset_env, onset_env, mode="full")
    ac = ac[len(ac) // 2:]
    ac[0] = ac[1] if len(ac) > 1 else ac[0]  # zero-lag is always the max; ignore it

    min_lag = max(1, int(frame_rate * 60.0 / bpm_range[1]))
    max_lag = min(len(ac) - 1, int(frame_rate * 60.0 / bpm_range[0]))
    if max_lag <= min_lag:
        return 120.0, 0.0

    segment = ac[min_lag:max_lag + 1]
    peak_offset = int(np.argmax(segment))
    lag = min_lag + peak_offset
    bpm = 60.0 * frame_rate / lag
    confidence = float(segment[peak_offset] / (ac.max() + 1e-8))

    def support(candidate_bpm):
        lag_c = frame_rate * 60.0 / candidate_bpm
        idx = int(round(lag_c))
        if idx < 1 or idx >= len(ac):
            return -np.inf
        return ac[idx]

    # Octave-error correction: the strongest autocorrelation peak is often
    # at 2x or 0.5x the perceptual tempo. Re-score the neighborhood and
    # keep whichever candidate the autocorrelation actually supports best.
    candidates = [bpm, bpm * 2.0, bpm / 2.0]
    candidates = [c for c in candidates if bpm_range[0] * 0.5 <= c <= bpm_range[1] * 2.0]
    best_bpm = max(candidates, key=support)

    return float(best_bpm), confidence


def _chroma_from_magnitude(S_mag, sr, n_fft):
    """Fold an STFT magnitude spectrogram into a 12-bin chromagram."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    freqs = freqs.copy()
    freqs[0] = freqs[1] * 0.5 if len(freqs) > 1 else 1.0

    midi = 69.0 + 12.0 * np.log2(freqs / 440.0)
    pitch_class = np.mod(np.round(midi), 12).astype(int)

    chroma = np.zeros((12, S_mag.shape[1]), dtype=np.float64)
    for pc in range(12):
        mask = pitch_class == pc
        if mask.any():
            chroma[pc] = S_mag[mask].sum(axis=0)
    return chroma


def detect_key(chroma_vector):
    """
    Krumhansl-Schmuckler key detection via Pearson correlation.

    Returns (key_name, is_minor, best_score, confidence, alternate_key).
    `confidence` is the correlation gap between the best and second-best
    key -- a low value means the piece sits ambiguously between two keys
    (usually relative major/minor, which share 6 of 7 scale tones).
    """
    x = chroma_vector - chroma_vector.mean()
    x_norm = np.linalg.norm(x) + 1e-8

    scores = {}
    for i, pitch in enumerate(PITCH_CLASSES):
        for mode, base_profile in (("Major", MAJOR_PROFILE), ("Minor", MINOR_PROFILE)):
            profile = np.roll(base_profile, i)
            p = profile - profile.mean()
            p_norm = np.linalg.norm(p) + 1e-8
            scores[f"{pitch} {mode}"] = float(np.dot(x, p) / (x_norm * p_norm))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_key, best_score = ranked[0]
    alternate_key, alt_score = ranked[1]
    confidence = float(best_score - alt_score)
    is_minor = "Minor" in best_key
    return best_key, is_minor, best_score, confidence, alternate_key


def to_flat_notation(key_name):
    root, mode = key_name.split(" ", 1)
    return f"{FLAT_EQUIVALENTS.get(root, root)} {mode}"


class AudioAnalyzer:
    """Cross-platform (desktop + Android) BPM/key/waveform analyzer."""

    N_FFT = 2048
    HOP_LENGTH = 512

    @classmethod
    def analyze_file(cls, filepath, target_sr=22050):
        try:
            y, sr = load_wav(filepath, target_sr=target_sr, mono=True)
            if len(y) == 0:
                raise ValueError("Empty audio signal")

            S_mag = _stft_magnitude(y, n_fft=cls.N_FFT, hop_length=cls.HOP_LENGTH)

            onset_env = _onset_envelope(S_mag, sr, cls.HOP_LENGTH)
            bpm, bpm_confidence = _estimate_tempo(onset_env, sr, cls.HOP_LENGTH)

            chroma = _chroma_from_magnitude(S_mag, sr, cls.N_FFT)
            chroma_avg = chroma.mean(axis=1)
            chroma_norm = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-8)

            key_name, is_minor, key_score, key_confidence, alternate_key = detect_key(
                chroma_norm
            )

            step = max(1, len(y) // 1000)
            samples = y[::step]
            max_abs = float(np.max(np.abs(samples))) if len(samples) else 1.0
            if max_abs == 0:
                max_abs = 1.0
            waveform_data = (samples / max_abs).astype(np.float32).tolist()

            duration = float(len(y) / sr)

            return {
                "bpm": round(bpm, 2),
                "bpm_confidence": round(bpm_confidence, 3),
                "key": to_flat_notation(key_name),
                "is_minor": is_minor,
                "key_confidence": round(key_confidence, 3),
                "alternate_key": to_flat_notation(alternate_key),
                "duration": duration,
                "sample_rate": int(sr),
                "waveform_data": waveform_data[:1000],
            }
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}") from e
