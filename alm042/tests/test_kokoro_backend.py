
"""Tests for the Kokoro-82M backend.

Real generation is skipped unless ``VOICEFORGE_TEST_KOKORO=1`` is set, so the
suite never downloads 300 MB of weights by accident. The integration is
exercised manually with that flag plus the real-model verification script.
"""

from __future__ import annotations

import os

import pytest
import soundfile as sf

import voiceforge.kokoro_backend as kb
from voiceforge.kokoro_backend import (
    DEFAULT_VOICE,
    KOKORO_SAMPLE_RATE,
    KOKORO_VOICES,
    KokoroBackend,
    KokoroError,
    kokoro_is_installed,
)

RUN_REAL = os.environ.get("VOICEFORGE_TEST_KOKORO") == "1"


def test_package_detection_returns_bool():
    assert isinstance(kokoro_is_installed(), bool)


def test_default_voice_is_american():
    assert DEFAULT_VOICE.startswith("a")
    assert DEFAULT_VOICE in KOKORO_VOICES


def test_unknown_voice_is_rejected():
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        KokoroBackend(voice="not_a_voice")


def test_speed_range_is_validated():
    with pytest.raises(ValueError, match="Speed"):
        KokoroBackend(speed=5.0)


def test_missing_package_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(kb, "kokoro_is_installed", lambda: False)

    backend = KokoroBackend()

    with pytest.raises(KokoroError, match="pip install kokoro"):
        backend.load()


def test_sample_rate_is_24khz():
    assert KokoroBackend.SAMPLE_RATE == 24000 == KOKORO_SAMPLE_RATE


def test_configure_cache_respects_explicit_hf_home(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("HF_HOME", str(explicit))

    assert kb.configure_cache(tmp_path / "fallback") == str(explicit)


def test_configure_cache_returns_none_when_default_is_writable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(kb, "_is_writable", lambda path: True)

    assert kb.configure_cache(tmp_path / "local") is None


def test_configure_cache_falls_back_when_default_is_not_writable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(kb, "_is_writable", lambda path: False)

    target = tmp_path / "local"
    result = kb.configure_cache(target)

    assert result == str(target.resolve())
    assert os.environ["HF_HOME"] == result
    assert target.is_dir()
    assert os.environ["HF_HUB_CACHE"] == str(target.resolve() / "hub")


def test_is_writable_detects_a_writable_directory(tmp_path):
    assert kb._is_writable(tmp_path / "fresh") is True


@pytest.mark.skipif(
    not RUN_REAL,
    reason="set VOICEFORGE_TEST_KOKORO=1 to run real Kokoro generation",
)
def test_real_generation_writes_valid_wav(tmp_path):
    backend = KokoroBackend(voice="af_heart")

    output = tmp_path / "kokoro.wav"
    backend.save("Hello, this is a Kokoro test.", output)

    info = sf.info(str(output))

    assert output.is_file()
    assert output.stat().st_size > 1000
    assert info.samplerate == KOKORO_SAMPLE_RATE
    assert info.frames > 0
