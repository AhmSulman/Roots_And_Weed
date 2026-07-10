import subprocess

import pytest

from okf import mp3_convert


def test_ffmpeg_available_returns_bool():
    assert isinstance(mp3_convert.ffmpeg_available(), bool)


def test_convert_desktop_raises_when_ffmpeg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mp3_convert.shutil, "which", lambda name: None)
    with pytest.raises(mp3_convert.ConversionError, match="ffmpeg was not found"):
        mp3_convert._convert_desktop(
            "song.mp3", str(tmp_path / "song.wav"), target_sr=22050
        )


def test_convert_desktop_invokes_ffmpeg_with_expected_args(monkeypatch, tmp_path):
    monkeypatch.setattr(mp3_convert.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mp3_convert.subprocess, "run", fake_run)

    out_path = str(tmp_path / "out.wav")
    mp3_convert._convert_desktop("song.mp3", out_path, target_sr=22050)

    assert captured["cmd"][0] == "ffmpeg"
    assert "song.mp3" in captured["cmd"]
    assert out_path in captured["cmd"]
    assert "22050" in captured["cmd"]


def test_convert_desktop_raises_conversion_error_on_ffmpeg_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(mp3_convert.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="invalid data found")

    monkeypatch.setattr(mp3_convert.subprocess, "run", fake_run)

    with pytest.raises(mp3_convert.ConversionError, match="ffmpeg failed"):
        mp3_convert._convert_desktop("song.mp3", str(tmp_path / "out.wav"), target_sr=22050)


def test_convert_mp3_to_wav_dispatches_to_desktop_when_not_android(monkeypatch, tmp_path):
    monkeypatch.setattr(mp3_convert, "platform", "win")
    called = {}
    monkeypatch.setattr(
        mp3_convert, "_convert_desktop",
        lambda i, o, sr: called.update(path=o),
    )
    out_path = str(tmp_path / "out.wav")
    mp3_convert.convert_mp3_to_wav("song.mp3", out_path)
    assert called["path"] == out_path


def test_convert_mp3_to_wav_dispatches_to_android_path(monkeypatch, tmp_path):
    monkeypatch.setattr(mp3_convert, "platform", "android")
    called = {}
    monkeypatch.setattr(
        mp3_convert, "_convert_android",
        lambda i, o, sr: called.update(path=o),
    )
    out_path = str(tmp_path / "out.wav")
    mp3_convert.convert_mp3_to_wav("song.mp3", out_path)
    assert called["path"] == out_path
