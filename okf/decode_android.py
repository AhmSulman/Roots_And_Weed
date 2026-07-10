"""Android audio decoder -> common in-memory PCM stream (y float32, sr).

WAV: stdlib `wave`. MP3/other: Android's own MediaExtractor/MediaCodec via
pyjnius -- no ffmpeg, no extra APK size. Returns PCM in memory (no temp WAV).
"""

import numpy as np

from okf.audio_analysis import load_wav, _resample


class DecodeError(Exception):
    pass


def decode(path, target_sr=22050, mono=True):
    if path.lower().endswith(".wav"):
        return load_wav(path, target_sr=target_sr, mono=mono)
    return _decode_android(path, target_sr, mono)


def _decode_android(path, target_sr, mono):
    try:
        from jnius import autoclass
    except ImportError as e:
        raise DecodeError(
            "pyjnius unavailable -- this path only runs inside a built APK, "
            "not a desktop interpreter. Use okf.decode_desktop there."
        ) from e

    MediaExtractor = autoclass("android.media.MediaExtractor")
    MediaCodec = autoclass("android.media.MediaCodec")
    MediaFormat = autoclass("android.media.MediaFormat")
    BufferInfo = autoclass("android.media.MediaCodec$BufferInfo")

    extractor = MediaExtractor()
    extractor.setDataSource(path)

    track_index, audio_format = -1, None
    for i in range(extractor.getTrackCount()):
        fmt = extractor.getTrackFormat(i)
        mime = fmt.getString(MediaFormat.KEY_MIME)
        if mime and mime.startswith("audio/"):
            track_index, audio_format = i, fmt
            break
    if track_index < 0:
        raise DecodeError("No audio track found in file")

    extractor.selectTrack(track_index)
    mime = audio_format.getString(MediaFormat.KEY_MIME)
    src_sr = audio_format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
    src_channels = audio_format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

    codec = MediaCodec.createDecoderByType(mime)
    codec.configure(audio_format, None, None, 0)
    codec.start()

    TIMEOUT_US = 10000
    buf_info = BufferInfo()
    pcm_chunks = []
    input_done = output_done = False

    while not output_done:
        if not input_done:
            in_index = codec.dequeueInputBuffer(TIMEOUT_US)
            if in_index >= 0:
                in_buffer = codec.getInputBuffer(in_index)
                sample_size = extractor.readSampleData(in_buffer, 0)
                if sample_size < 0:
                    codec.queueInputBuffer(in_index, 0, 0, 0,
                                           MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    input_done = True
                else:
                    codec.queueInputBuffer(in_index, 0, sample_size,
                                           extractor.getSampleTime(), 0)
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

    codec.stop(); codec.release(); extractor.release()

    y = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16).astype(np.float32) / 32768.0
    if src_channels > 1:
        y = y.reshape(-1, src_channels)
        if mono:
            y = y.mean(axis=1)
    if target_sr and src_sr != target_sr:
        y = _resample(y, src_sr, target_sr)
        src_sr = target_sr
    return y.astype(np.float32), src_sr
