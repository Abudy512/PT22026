
import pytest
import torch

import voiceforge.tts as tts_module
from voiceforge.tts import VoiceForgeTTS


class FakeAcousticModel(torch.nn.Module):
    """Small fake model for testing the TTS pipeline."""

    def __init__(self):
        super().__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1))
        self.received_text = None
        self.received_lengths = None

    def forward(self, text, text_lengths):
        self.received_text = text
        self.received_lengths = text_lengths

        return torch.ones(1, 80, 100)


@pytest.fixture
def fake_model():
    return FakeAcousticModel()


@pytest.fixture
def tts(fake_model, monkeypatch):
    def fake_mel_to_audio(mel, n_iter=32):
        assert mel.shape == (80, 100)
        assert n_iter == 32
        return torch.zeros(22050)

    monkeypatch.setattr(
        tts_module,
        "mel_to_audio",
        fake_mel_to_audio,
    )

    return VoiceForgeTTS(fake_model, device="cpu")


def test_synthesize_returns_waveform(tts):
    waveform = tts.synthesize("Hello world.")

    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 1
    assert waveform.shape[0] == 22050


def test_synthesize_processes_text(fake_model, tts):
    tts.synthesize("Hello world.")

    assert fake_model.received_text is not None
    assert fake_model.received_text.ndim == 2
    assert fake_model.received_lengths.item() > 0


def test_synthesize_rejects_empty_text(tts):
    with pytest.raises(ValueError, match="cannot be empty"):
        tts.synthesize("   ")


def test_synthesize_rejects_non_string_text(tts):
    with pytest.raises(TypeError, match="must be a string"):
        tts.synthesize(123)


def test_synthesize_rejects_invalid_model_output(fake_model, monkeypatch):
    def invalid_forward(text, text_lengths):
        return torch.ones(80, 100)

    fake_model.forward = invalid_forward

    pipeline = VoiceForgeTTS(fake_model, device="cpu")

    with pytest.raises(ValueError, match="3D Mel spectrogram"):
        pipeline.synthesize("Hello world.")


def test_save_creates_output_file(tts, tmp_path, monkeypatch):
    saved = {}

    def fake_save_audio(waveform, path, sample_rate=22050):
        saved["waveform"] = waveform
        saved["path"] = path
        saved["sample_rate"] = sample_rate

        with open(path, "wb") as audio_file:
            audio_file.write(b"fake audio")

    monkeypatch.setattr(
        tts_module,
        "save_audio",
        fake_save_audio,
    )

    output_path = tmp_path / "generated" / "test.wav"

    waveform = tts.save("Hello world.", output_path)

    assert output_path.exists()
    assert waveform.ndim == 1
    assert saved["sample_rate"] == 22050
    assert saved["path"] == str(output_path)
