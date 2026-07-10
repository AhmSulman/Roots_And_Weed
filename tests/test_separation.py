import os

import numpy as np
import pytest

from okf import separation


def test_median_filter_axis_preserves_shape():
    S = np.random.rand(50, 80)
    out = separation._median_filter_axis(S, win=5, axis=1)
    assert out.shape == S.shape


def test_median_filter_axis_no_op_for_window_of_one():
    S = np.random.rand(10, 10)
    out = separation._median_filter_axis(S, win=1, axis=0)
    assert np.array_equal(out, S)


def test_hpss_masks_sum_to_original_magnitude():
    S = np.abs(np.random.randn(100, 60)) + 0.1
    harmonic, percussive = separation._hpss(S)
    assert harmonic.shape == S.shape
    assert percussive.shape == S.shape
    # harmonic + percussive masks are a soft partition of the same magnitude
    assert np.allclose(harmonic + percussive, S, atol=1e-6)


def test_stft_istft_roundtrip_reconstructs_signal():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    S = separation._stft_complex(y, n_fft=2048, hop_length=512)
    y_reconstructed = separation._istft(S, hop_length=512, n_fft=2048, length=len(y))

    # Not bit-exact (windowed overlap-add), but should closely track the
    # original away from the very start/end edge-effects.
    mid = slice(2048, len(y) - 2048)
    correlation = np.corrcoef(y[mid], y_reconstructed[mid])[0, 1]
    assert correlation > 0.9


def test_separate_dsp_produces_all_four_stems_for_stereo(wav_factory, tmp_path):
    path = wav_factory(stereo=True, duration=2.0, click_bpm=120.0)
    out_dir = tmp_path / "stems"
    stems = separation.separate_dsp(path, str(out_dir))

    assert set(stems.keys()) == {"harmonic", "percussive", "instrumental", "vocals_estimate"}
    for stem_path in stems.values():
        assert os.path.exists(stem_path)


def test_separate_dsp_mono_input_skips_stereo_only_stems(wav_factory, tmp_path):
    path = wav_factory(stereo=False, duration=1.0)
    out_dir = tmp_path / "stems_mono"
    stems = separation.separate_dsp(path, str(out_dir))

    assert "harmonic" in stems
    assert "percussive" in stems
    assert "instrumental" not in stems
    assert "vocals_estimate" not in stems


def test_is_torch_available_returns_bool():
    assert isinstance(separation.is_torch_available(), bool)


def test_separate_desktop_raises_clear_error_without_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(separation, "is_torch_available", lambda: False)
    with pytest.raises(RuntimeError, match="Demucs/torch not installed"):
        separation.separate_desktop("some_file.wav", str(tmp_path))
