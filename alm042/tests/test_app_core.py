
"""Tests for the shared application logic."""

from __future__ import annotations

import pytest

from voiceforge.app_core import (
    MAX_TEXT_LENGTH,
    AppError,
    VoiceForgeApp,
)


def make_app(tmp_path, checkpoint, **kwargs):
    return VoiceForgeApp(
        checkpoint_path=checkpoint,
        device="cpu",
        output_dir=tmp_path / "app_outputs",
        griffin_lim_iterations=2,
        max_mel_frames=32,
        **kwargs,
    )


def test_validate_text_accepts_english():
    assert VoiceForgeApp.validate_text("  Hello world.  ") == "Hello world."


def test_validate_text_rejects_empty():
    with pytest.raises(AppError, match="enter some English text"):
        VoiceForgeApp.validate_text("   ")


def test_validate_text_rejects_non_string():
    with pytest.raises(AppError, match="string"):
        VoiceForgeApp.validate_text(123)


def test_validate_text_rejects_too_long_text():
    with pytest.raises(AppError, match="too long"):
        VoiceForgeApp.validate_text("a" * (MAX_TEXT_LENGTH + 1))


def test_validate_text_rejects_unsupported_characters():
    with pytest.raises(AppError, match="Unsupported characters"):
        VoiceForgeApp.validate_text("Hello 123")


def test_missing_checkpoint_reports_clear_error(tmp_path):
    app = make_app(tmp_path, tmp_path / "missing.pt")

    with pytest.raises(AppError, match="not found"):
        app.synthesize("Hello world.")


def test_synthesize_writes_valid_wav(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    result = app.synthesize("Hello world.")

    from pathlib import Path

    path = Path(result.path)

    assert path.is_file()
    assert path.stat().st_size > 0
    assert result.filename.endswith(".wav")
    assert result.duration_seconds > 0
    assert result.text == "Hello world."


def test_synthesize_creates_unique_files(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    first = app.synthesize("Hello world.")
    second = app.synthesize("Hello world.")

    assert first.filename != second.filename


def test_model_is_loaded_lazily(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    assert app.is_loaded is False

    app.load()

    assert app.is_loaded is True


def test_resolve_audio_rejects_traversal(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    with pytest.raises(AppError):
        app.resolve_audio("../secret.wav")

    with pytest.raises(AppError):
        app.resolve_audio("missing.wav")

    with pytest.raises(AppError):
        app.resolve_audio("file.txt")


def test_resolve_audio_returns_generated_file(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    result = app.synthesize("Hello world.")
    resolved = app.resolve_audio(result.filename)

    assert resolved.is_file()
    assert resolved.name == result.filename
