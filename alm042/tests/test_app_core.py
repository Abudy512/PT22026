
"""Tests for the shared application logic."""

from __future__ import annotations

import pytest

from voiceforge.app_core import (
    ENGINE_KOKORO,
    ENGINE_VOICEFORGE,
    MAX_TEXT_LENGTH,
    AppError,
    VoiceForgeApp,
)


def make_app(tmp_path, checkpoint, **kwargs):
    kwargs.setdefault("engine", ENGINE_VOICEFORGE)

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


def test_resolve_engine_rejects_unknown(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    with pytest.raises(AppError, match="Unknown speech engine"):
        app.resolve_engine("does-not-exist")


def test_auto_engine_prefers_kokoro_when_installed(
    tmp_path,
    tiny_checkpoint,
    monkeypatch,
):
    import voiceforge.app_core as core

    monkeypatch.setattr(core, "kokoro_is_installed", lambda: True)

    app = VoiceForgeApp(
        checkpoint_path=tiny_checkpoint,
        device="cpu",
        output_dir=tmp_path / "out",
        engine="auto",
    )

    assert app.resolve_engine() == ENGINE_KOKORO


def test_auto_engine_falls_back_to_custom_model(
    tmp_path,
    tiny_checkpoint,
    monkeypatch,
):
    import voiceforge.app_core as core

    monkeypatch.setattr(core, "kokoro_is_installed", lambda: False)

    app = VoiceForgeApp(
        checkpoint_path=tiny_checkpoint,
        device="cpu",
        output_dir=tmp_path / "out",
        engine="auto",
    )

    assert app.resolve_engine() == ENGINE_VOICEFORGE


def test_validate_text_allows_digits_for_kokoro():
    assert (
        VoiceForgeApp.validate_text("Room 204 at 9:30", engine=ENGINE_KOKORO)
        == "Room 204 at 9:30"
    )


def test_validate_text_rejects_digits_for_custom_model():
    with pytest.raises(AppError, match="Unsupported characters"):
        VoiceForgeApp.validate_text("Room 204", engine=ENGINE_VOICEFORGE)


def test_describe_engines_reports_both_backends(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    info = app.describe_engines()

    assert set(info) >= {"default", "kokoro", "voiceforge"}
    assert "voices" in info["kokoro"]
    assert info["voiceforge"]["sample_rate"] == 22050


def test_kokoro_voice_is_validated(tmp_path, tiny_checkpoint):
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        VoiceForgeApp(
            checkpoint_path=tiny_checkpoint,
            device="cpu",
            output_dir=tmp_path / "out",
            kokoro_voice="not_a_voice",
        )


def test_synthesize_returns_engine_and_sample_rate(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    result = app.synthesize("Hello world.")

    assert result.engine == ENGINE_VOICEFORGE
    assert result.sample_rate == 22050


def test_set_engine_validates_name(tmp_path, tiny_checkpoint):
    app = make_app(tmp_path, tiny_checkpoint)

    app.set_engine(ENGINE_KOKORO)
    assert app.requested_engine == ENGINE_KOKORO

    with pytest.raises(AppError, match="Unknown speech engine"):
        app.set_engine("nope")
