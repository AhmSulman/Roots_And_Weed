import wave

import numpy as np
import pytest


def make_wav(path, freq=220.0, sr=22050, duration=2.0, stereo=False, silence=False, click_bpm=None):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    if silence:
        sig = np.zeros_like(t)
    else:
        sig = 0.5 * np.sin(2 * np.pi * freq * t)
        if click_bpm:
            period = 60.0 / click_bpm
            phase = np.mod(t, period)
            clicks = (phase < 0.01).astype(np.float64)
            sig = sig * 0.3 + clicks * 0.9

    if stereo:
        data = np.stack([sig, sig * 0.7], axis=1)
        n_channels = 2
    else:
        data = sig
        n_channels = 1

    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return str(path)


@pytest.fixture
def wav_factory(tmp_path):
    def _factory(name="test.wav", **kwargs):
        return make_wav(tmp_path / name, **kwargs)

    return _factory
