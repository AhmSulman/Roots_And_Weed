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


def test_separate_dsp_produces_vocals_and_instrumental_for_stereo(wav_factory, tmp_path):
    path = wav_factory(stereo=True, duration=2.0, click_bpm=120.0)
    out_dir = tmp_path / "stems"
    stems = separation.separate_dsp(path, str(out_dir))

    assert set(stems.keys()) == {"vocals", "instrumental"}
    for stem_path in stems.values():
        assert os.path.exists(stem_path)


def test_separate_dsp_mono_input_raises_clear_error(wav_factory, tmp_path):
    path = wav_factory(stereo=False, duration=1.0)
    out_dir = tmp_path / "stems_mono"

    with pytest.raises(ValueError, match="stereo"):
        separation.separate_dsp(path, str(out_dir))


def test_separate_dsp_reports_monotonic_progress_to_completion(wav_factory, tmp_path):
    path = wav_factory(stereo=True, duration=1.0)
    seen = []
    separation.separate_dsp(path, str(tmp_path / "stems"), progress=seen.append)

    assert seen == sorted(seen)  # monotonically increasing
    assert seen[-1] == 1.0
    assert all(0.0 <= p <= 1.0 for p in seen)


def test_separate_with_cache_reuses_stems_and_records_metadata(wav_factory, tmp_path):
    path = wav_factory(stereo=True, duration=1.0)
    out_dir = str(tmp_path / "stems")

    stems1, meta1, from_cache1 = separation.separate_with_cache(path, out_dir, False)
    assert from_cache1 is False
    assert meta1["mode"] == "dsp"
    assert meta1["elapsed_seconds"] >= 0
    assert os.path.exists(os.path.join(out_dir, separation.STEMS_INDEX_FILENAME))

    stems2, meta2, from_cache2 = separation.separate_with_cache(path, out_dir, False)
    assert from_cache2 is True
    assert stems2 == stems1


def test_separate_with_cache_invalidates_when_source_changes(wav_factory, tmp_path):
    path = wav_factory(stereo=True, duration=1.0)
    out_dir = str(tmp_path / "stems")
    separation.separate_with_cache(path, out_dir, False)

    os.utime(path, (0, 0))  # simulate the source file changing
    _, _, from_cache = separation.separate_with_cache(path, out_dir, False)
    assert from_cache is False


def test_load_stems_metadata_rejects_deleted_stem(wav_factory, tmp_path):
    path = wav_factory(stereo=True, duration=1.0)
    out_dir = str(tmp_path / "stems")
    stems, _, _ = separation.separate_with_cache(path, out_dir, False)

    os.remove(stems["vocals"])
    assert separation.load_stems_metadata(out_dir, path) is None


def test_stems_index_survives_concurrent_saves(wav_factory, tmp_path):
    # read-modify-write race: without the index lock, parallel saves drop
    # each other's entries (last writer wins)
    from concurrent.futures import ThreadPoolExecutor

    out_dir = str(tmp_path / "stems")
    os.makedirs(out_dir)
    paths = [wav_factory(name=f"s{i}.wav", stereo=True, duration=0.3) for i in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda p: separation.save_stems_metadata(out_dir, p, {}, "dsp", 0.1),
            paths,
        ))

    index = separation._load_index(out_dir)
    assert len(index) == len(paths)  # no entry lost to a concurrent writer


def test_separate_dsp_parallel_runs_produce_independent_stems(wav_factory, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    out_dir = str(tmp_path / "stems")
    paths = [
        wav_factory(name=f"track{i}.wav", stereo=True, duration=0.5)
        for i in range(4)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda p: separation.separate_with_cache(p, out_dir, False), paths
        ))

    all_stem_paths = [p for stems, _, _ in results for p in stems.values()]
    assert len(set(all_stem_paths)) == 8  # 4 tracks x 2 stems, no collisions
    assert all(os.path.exists(p) for p in all_stem_paths)
    assert len(separation._load_index(out_dir)) == 4


def test_write_wav_stereo_preserves_channel_count_and_length(tmp_path):
    import wave as wave_mod

    sr = 22050
    y = np.stack([np.linspace(-0.5, 0.5, sr), np.linspace(0.5, -0.5, sr)], axis=1)
    path = str(tmp_path / "stereo.wav")
    separation._write_wav(path, y, sr)

    with wave_mod.open(path, "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getnframes() == sr  # not doubled by mis-labeled interleave


def test_is_torch_available_returns_bool():
    assert isinstance(separation.is_torch_available(), bool)


def test_separate_desktop_raises_clear_error_without_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(separation, "is_torch_available", lambda: False)
    with pytest.raises(RuntimeError, match="Demucs/torch not installed"):
        separation.separate_desktop("some_file.wav", str(tmp_path))
