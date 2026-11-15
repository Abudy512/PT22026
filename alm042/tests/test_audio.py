import numpy as np
import soundfile as sf

from voiceforge.audio import load_audio


def test_load_audio_converts_stereo_to_mono(tmp_path):
    audio = np.random.uniform(-1, 1, (16000, 2)).astype(np.float32)

    file_path = tmp_path / "test_audio.wav"
    sf.write(str(file_path), audio, 16000)

    waveform = load_audio(file_path)

    assert waveform.shape[0] == 1


def test_load_audio_resamples_audio(tmp_path):
    audio = np.random.uniform(-1, 1, 16000).astype(np.float32)

    file_path = tmp_path / "test_audio.wav"
    sf.write(str(file_path), audio, 16000)

    waveform = load_audio(file_path, target_sample_rate=22050)

    assert waveform.shape[1] == 22050