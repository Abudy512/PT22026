
"""Tests for the generated-audio quality checker."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from scripts.check_audio import analyze_file, analyze_waveform, build_parser

SAMPLE_RATE = 22050


def make_speech_like(seconds: float = 2.0) -> np.ndarray:
    """A signal with speech-like loud/quiet structure."""

    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE

    # Envelope that swells and fades, like syllables.
    envelope = 0.2 + 0.7 * (0.5 * (1 + np.sin(2 * np.pi * 1.2 * time)))
    audio = 0.8 * envelope * np.sin(2 * np.pi * 180 * time)

    return audio.astype(np.float64)


def test_healthy_signal_passes():
    report = analyze_waveform(make_speech_like())

    assert report["passed"] is True
    assert report["problems"] == []
    assert report["peak"] > 0.5
    assert report["energy_variation"] > 0.15


def test_silence_is_detected():
    report = analyze_waveform(np.zeros(SAMPLE_RATE))

    assert report["passed"] is False
    assert "silent" in report["problems"]


def test_non_finite_samples_are_detected():
    audio = make_speech_like()
    audio[100] = np.nan
    audio[200] = np.inf

    report = analyze_waveform(audio)

    assert "non_finite_samples" in report["problems"]
    assert report["non_finite_samples"] == 2


def test_clipping_is_detected():
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    clipped = np.sign(np.sin(2 * np.pi * 200 * time)) * 0.999

    report = analyze_waveform(clipped)

    assert "clipping" in report["problems"]
    assert report["clipping_ratio"] > 0.5


def test_flat_tone_is_flagged_as_buzz():
    time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * 220 * time)

    report = analyze_waveform(tone)

    assert report["passed"] is False
    assert "flat_buzz" in report["problems"]
    assert report["energy_variation"] < 0.15


def test_dc_offset_is_detected():
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = 0.5 + 0.1 * np.sin(2 * np.pi * 220 * time)

    report = analyze_waveform(audio)

    assert "dc_offset" in report["problems"]


def test_empty_waveform_is_rejected():
    report = analyze_waveform(np.array([]))

    assert report["passed"] is False
    assert report["problems"] == ["empty"]


def test_analyze_file_reads_wav(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(str(path), make_speech_like().astype(np.float32), SAMPLE_RATE)

    report = analyze_file(path)

    assert report["path"] == str(path)
    assert report["samples"] > 0
    assert report["duration_seconds"] == pytest.approx(2.0, abs=0.01)
    assert report["passed"] is True


def test_analyze_file_handles_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    mono = make_speech_like(0.5)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(str(path), stereo.astype(np.float32), SAMPLE_RATE)

    report = analyze_file(path)

    assert report["samples"] == len(mono)


def test_parser_defaults():
    args = build_parser().parse_args([])

    assert args.output_dir == "outputs/audio_checks"
    assert args.n_iter == 32
    assert args.text is None
