
import numpy as np
import pytest
import soundfile as sf
import torch

from voiceforge.audio import (
    load_audio,
    mel_spectrogram,
    mel_to_audio,
    save_audio,
)


def test_load_audio_converts_stereo_to_mono(tmp_path):
    audio = np.random.randn(22050, 2).astype(np.float32)

    file_path = tmp_path / "stereo.wav"
    sf.write(str(file_path), audio, 22050)

    waveform = load_audio(file_path)

    assert waveform.shape[0] == 1
    assert waveform.shape[1] == 22050


def test_load_audio_resamples_audio(tmp_path):
    audio = np.random.randn(16000).astype(np.float32)

    file_path = tmp_path / "audio.wav"
    sf.write(str(file_path), audio, 16000)

    waveform = load_audio(file_path)

    assert waveform.shape[0] == 1
    assert waveform.shape[1] == 22050


def test_mel_spectrogram_output_shape():
    waveform = torch.randn(1, 22050)

    mel = mel_spectrogram(waveform)

    assert mel.shape[0] == 1
    assert mel.shape[1] == 80
    assert torch.isfinite(mel).all()


def test_mel_to_audio_returns_finite_waveform():
    waveform = torch.sin(
        torch.linspace(0, 100, 5512)
    ).unsqueeze(0)

    mel = mel_spectrogram(waveform)

    audio = mel_to_audio(
        mel,
        n_iter=4,
    )

    assert audio.ndim == 1
    assert audio.numel() > 0
    assert torch.isfinite(audio).all()


def test_save_audio_creates_wav_file(tmp_path):
    waveform = torch.sin(
        torch.linspace(0, 100, 22050)
    )

    output_path = tmp_path / "test.wav"

    result = save_audio(
        waveform,
        output_path,
    )

    assert result.exists()

    saved_audio, sample_rate = sf.read(
        str(output_path)
    )

    assert sample_rate == 22050
    assert len(saved_audio) == 22050


def test_mel_to_audio_rejects_invalid_dimensions():
    invalid_mel = torch.randn(2, 80, 100)

    with pytest.raises(
        ValueError,
        match="Only one Mel spectrogram",
    ):
        mel_to_audio(invalid_mel)
