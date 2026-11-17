
"""Tests for the public Python API.

These tests use the documented import style, so they fail if the package
exports ever drift away from what the README and docs promise.
"""

from __future__ import annotations

import pytest
import torch

import voiceforge
from voiceforge import VoiceForgeTTS, generate_speech, load_model
from voiceforge.checkpoint import get_model_config


def test_public_exports_are_available():
    for name in voiceforge.__all__:
        assert hasattr(voiceforge, name), name


def test_version_is_a_string():
    assert isinstance(voiceforge.__version__, str)
    assert voiceforge.__version__.count(".") >= 1


def test_documented_import_style_works():
    from voiceforge import VoiceForgeTTS as Imported

    assert Imported is VoiceForgeTTS


def test_from_pretrained_loads_checkpoint(tiny_checkpoint):
    tts = VoiceForgeTTS.from_pretrained(tiny_checkpoint, device="cpu")

    assert tts.checkpoint_path == str(tiny_checkpoint)
    assert tts.model_config["architecture"] == "tacotron_lite"
    assert tts.model.training is False
    assert tts.sample_rate == 22050


def test_synthesize_returns_waveform(tiny_checkpoint):
    tts = VoiceForgeTTS.from_pretrained(
        tiny_checkpoint,
        device="cpu",
        griffin_lim_iterations=2,
    )

    waveform = tts.synthesize("Hello world.")

    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 1
    assert waveform.numel() > 0
    assert torch.isfinite(waveform).all()


def test_save_writes_a_wav_file(tiny_checkpoint, tmp_path):
    tts = VoiceForgeTTS.from_pretrained(
        tiny_checkpoint,
        device="cpu",
        griffin_lim_iterations=2,
    )

    output = tmp_path / "nested" / "out.wav"

    tts.save("Hello world.", output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        VoiceForgeTTS.from_pretrained(tmp_path / "missing.pt", device="cpu")


def test_empty_text_raises_value_error(tiny_checkpoint):
    tts = VoiceForgeTTS.from_pretrained(tiny_checkpoint, device="cpu")

    with pytest.raises(ValueError, match="cannot be empty"):
        tts.synthesize("   ")


def test_non_string_text_raises_type_error(tiny_checkpoint):
    tts = VoiceForgeTTS.from_pretrained(tiny_checkpoint, device="cpu")

    with pytest.raises(TypeError, match="must be a string"):
        tts.synthesize(123)


def test_unsupported_characters_raise_value_error(tiny_checkpoint):
    tts = VoiceForgeTTS.from_pretrained(tiny_checkpoint, device="cpu")

    with pytest.raises(ValueError, match="Unsupported characters"):
        tts.synthesize("Hello 123")


def test_generate_speech_helper(tiny_checkpoint, tmp_path):
    output = tmp_path / "helper.wav"

    result = generate_speech(
        "Hello world.",
        output,
        checkpoint=tiny_checkpoint,
        device="cpu",
        griffin_lim_iterations=2,
    )

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_generate_speech_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_speech(
            "Hello.",
            tmp_path / "out.wav",
            checkpoint=tmp_path / "missing.pt",
            device="cpu",
        )


def test_load_model_is_exported(tiny_checkpoint):
    model, config, metadata = load_model(tiny_checkpoint)

    assert config["architecture"] == "tacotron_lite"
    assert metadata["epoch"] == 1
    assert model.training is False


def test_exported_config_matches_checkpoint(tiny_model):
    config = get_model_config(tiny_model)

    rebuilt = voiceforge.ModelConfig.from_dict(config)

    assert rebuilt.vocab_size == config["vocab_size"]


@pytest.mark.parametrize(
    "name",
    ["VoiceForgeTTS", "generate_speech", "load_model", "AcousticModel"],
)
def test_public_objects_are_documented(name):
    obj = getattr(voiceforge, name)

    assert obj.__doc__ is not None
    assert len(obj.__doc__.strip()) > 20


@pytest.mark.parametrize(
    "method",
    ["from_pretrained", "synthesize", "save"],
)
def test_public_methods_are_documented(method):
    doc = getattr(VoiceForgeTTS, method).__doc__

    assert doc is not None
    assert len(doc.strip()) > 20
