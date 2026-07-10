import numpy as np
import pytest

from okf.audio_analysis import (
    MAJOR_PROFILE,
    MINOR_PROFILE,
    PITCH_CLASSES,
    AudioAnalyzer,
    _chroma_from_magnitude,
    _estimate_tempo,
    _onset_envelope,
    _resample,
    _stft_magnitude,
    detect_key,
    load_wav,
    relative_key,
    to_flat_notation,
)


# -- load_wav -----------------------------------------------------------------

def test_load_wav_mono_pcm16_roundtrips_amplitude(wav_factory):
    path = wav_factory(freq=440.0, sr=22050, duration=1.0)
    y, sr = load_wav(path, target_sr=22050)
    assert sr == 22050
    assert len(y) == pytest.approx(22050, abs=2)
    assert np.max(np.abs(y)) <= 1.0
    assert np.max(np.abs(y)) > 0.3  # our test tone has amplitude 0.5


def test_load_wav_stereo_downmixes_to_mono(wav_factory):
    path = wav_factory(stereo=True, sr=22050, duration=1.0)
    y, sr = load_wav(path, target_sr=22050, mono=True)
    assert y.ndim == 1


def test_load_wav_resamples_when_target_differs(wav_factory):
    path = wav_factory(sr=44100, duration=1.0)
    y, sr = load_wav(path, target_sr=22050)
    assert sr == 22050
    assert len(y) == pytest.approx(22050, abs=2)


# -- _resample ------------------------------------------------------------

def test_resample_identity_returns_same_length():
    y = np.random.randn(1000).astype(np.float32)
    out = _resample(y, 22050, 22050)
    assert len(out) == len(y)


def test_resample_empty_array_returns_empty():
    out = _resample(np.array([], dtype=np.float32), 22050, 44100)
    assert len(out) == 0


def test_resample_halves_length_when_downsampling():
    y = np.random.randn(1000).astype(np.float32)
    out = _resample(y, 44100, 22050)
    assert len(out) == pytest.approx(500, abs=2)


# -- STFT / onset -----------------------------------------------------------

def test_stft_magnitude_shape_matches_fft_bins():
    y = np.random.randn(22050).astype(np.float32)
    S = _stft_magnitude(y, n_fft=2048, hop_length=512)
    assert S.shape[0] == 1025  # 2048 // 2 + 1
    assert S.shape[1] > 0
    assert np.all(S >= 0)


def test_onset_envelope_is_normalized_0_to_1():
    y = np.random.randn(22050).astype(np.float32)
    S = _stft_magnitude(y)
    env = _onset_envelope(S, sr=22050, hop_length=512)
    assert env.min() >= 0.0
    assert env.max() <= 1.0 + 1e-6


# -- tempo --------------------------------------------------------------

def test_estimate_tempo_detects_click_track_bpm(wav_factory):
    path = wav_factory(freq=110.0, sr=22050, duration=6.0, click_bpm=120.0)
    y, sr = load_wav(path, target_sr=22050)
    S = _stft_magnitude(y)
    env = _onset_envelope(S, sr, 512)
    bpm, confidence = _estimate_tempo(env, sr, 512)
    # Octave-tolerant: 120 BPM may be detected as 60/120/240 depending on
    # which harmonic the autocorrelation favors, but it must be one of them.
    assert any(abs(bpm - target) < 5 for target in (60.0, 120.0, 240.0))
    assert confidence > 0


def test_estimate_tempo_returns_default_when_no_range_available():
    env = np.zeros(3)
    bpm, confidence = _estimate_tempo(env, sr=22050, hop_length=512)
    assert bpm == 120.0
    assert confidence == 0.0


# -- chroma / key ---------------------------------------------------------

def test_chroma_peaks_at_correct_pitch_class_for_a440(wav_factory):
    path = wav_factory(freq=440.0, sr=22050, duration=2.0)
    y, sr = load_wav(path, target_sr=22050)
    S = _stft_magnitude(y, n_fft=2048, hop_length=512)
    chroma = _chroma_from_magnitude(S, sr, n_fft=2048)
    chroma_avg = chroma.mean(axis=1)
    a_index = PITCH_CLASSES.index("A")
    assert int(np.argmax(chroma_avg)) == a_index


@pytest.mark.parametrize("i,pitch", list(enumerate(PITCH_CLASSES)))
def test_detect_key_recovers_exact_major_profile(i, pitch):
    profile = np.roll(MAJOR_PROFILE, i)
    key_name, is_minor, score, confidence, alt = detect_key(profile)
    assert key_name == f"{pitch} Major"
    assert is_minor is False
    assert confidence > 0


def test_detect_key_recovers_exact_minor_profile():
    profile = np.roll(MINOR_PROFILE, 3)  # D# / Eb Minor
    key_name, is_minor, score, confidence, alt = detect_key(profile)
    assert is_minor is True
    assert key_name == "D# Minor"


def test_detect_key_best_score_beats_alternate():
    profile = np.roll(MAJOR_PROFILE, 0)
    key_name, is_minor, score, confidence, alt = detect_key(profile)
    assert score >= 0
    assert confidence >= 0  # best is never worse than the runner-up


def test_relative_key_maps_both_directions():
    assert relative_key("C Major") == "A Minor"
    assert relative_key("A Minor") == "C Major"
    assert relative_key("F# Minor") == "A Major"
    for i, pitch in enumerate(PITCH_CLASSES):
        for mode in ("Major", "Minor"):
            name = f"{pitch} {mode}"
            assert relative_key(relative_key(name)) == name  # involution


def test_detect_key_tie_break_prefers_stronger_tonic():
    # Chroma sitting exactly between C Major and A Minor, but with the
    # energy tilted toward A: the tonic tie-break must pick A Minor.
    chroma = (np.roll(MAJOR_PROFILE, 0) + np.roll(MINOR_PROFILE, 9)) / 2.0
    chroma[9] += 1.5  # extra weight on A
    chroma[0] -= 0.5  # less weight on C
    key_name, is_minor, _, _, _ = detect_key(chroma / np.linalg.norm(chroma))
    assert key_name == "A Minor"
    assert is_minor is True


def test_detect_key_endpoint_check_votes_for_tonic_at_track_edges():
    # Globally ambiguous between C Major / A Minor, but the track opens
    # and closes with strong A: the endpoint check should tip it to A Minor.
    ambiguous = (np.roll(MAJOR_PROFILE, 0) + np.roll(MINOR_PROFILE, 9)) / 2.0
    frames = np.tile(ambiguous[:, None], (1, 100))
    frames[9, :10] += 4.0   # strong A at the start
    frames[9, -10:] += 4.0  # strong A at the end
    chroma_avg = frames.mean(axis=1)
    key_name, is_minor, _, _, _ = detect_key(
        chroma_avg / np.linalg.norm(chroma_avg), chroma_frames=frames
    )
    assert key_name == "A Minor"


# -- flat notation --------------------------------------------------------

@pytest.mark.parametrize(
    "sharp,flat",
    [("C#", "Db"), ("D#", "Eb"), ("F#", "Gb"), ("G#", "Ab"), ("A#", "Bb")],
)
def test_to_flat_notation_converts_sharps(sharp, flat):
    assert to_flat_notation(f"{sharp} Major") == f"{flat} Major"


def test_to_flat_notation_leaves_naturals_unchanged():
    assert to_flat_notation("C Major") == "C Major"
    assert to_flat_notation("E Minor") == "E Minor"


# -- full pipeline --------------------------------------------------------

def test_analyze_file_returns_expected_keys(wav_factory):
    path = wav_factory(freq=220.0, sr=22050, duration=2.0, click_bpm=100.0)
    result = AudioAnalyzer.analyze_file(path)
    for key in (
        "bpm", "bpm_confidence", "key", "is_minor", "key_confidence",
        "relative_key", "alternate_key", "duration", "sample_rate",
        "waveform_data",
    ):
        assert key in result


def test_analyze_file_waveform_data_capped_at_1000_samples(wav_factory):
    path = wav_factory(duration=5.0)
    result = AudioAnalyzer.analyze_file(path)
    assert len(result["waveform_data"]) <= 1000


def test_analyze_file_handles_silence_without_crashing(wav_factory):
    path = wav_factory(silence=True, duration=1.0)
    result = AudioAnalyzer.analyze_file(path)
    assert result["bpm"] > 0
    assert all(abs(v) <= 1.0 for v in result["waveform_data"])


def test_analyze_file_raises_wrapped_exception_for_bad_path():
    with pytest.raises(Exception, match="Analysis failed"):
        AudioAnalyzer.analyze_file("this/path/does/not/exist.wav")
