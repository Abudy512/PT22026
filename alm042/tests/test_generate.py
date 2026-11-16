
"""Tests for the speech generation command line script."""

from __future__ import annotations

import json

import pytest
import soundfile as sf
import torch

from scripts.generate import build_parser, generate, main
from voiceforge.model import AcousticModel
from voiceforge.text import CHAR_TO_ID

VOCAB_SIZE = len(CHAR_TO_ID)


def write_checkpoint(path, with_config=True):
    """Create a small, untrained checkpoint that loads quickly."""

    model = AcousticModel(
        vocab_size=VOCAB_SIZE,
        n_mels=80,
        embedding_dim=16,
        hidden_dim=16,
    )

    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "loss": 1.0,
    }

    if with_config:
        checkpoint["model_config"] = {
            "architecture": "legacy_interpolate",
            "vocab_size": VOCAB_SIZE,
            "n_mels": 80,
            "embedding_dim": 16,
            "hidden_dim": 16,
        }

    torch.save(checkpoint, path)

    return path


def make_args(tmp_path, checkpoint, **overrides):
    """Build a parsed argument namespace with sensible fast defaults."""

    argv = [
        "Hello world.",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(tmp_path / "out.wav"),
        "--device",
        "cpu",
        "--n-iter",
        "4",
        "--max-frames",
        "48",
    ]

    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])

    return build_parser().parse_args(argv)


def test_generate_creates_valid_wav(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")

    exit_code = main(
        [
            "Hello world.",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "hello.wav"),
            "--device",
            "cpu",
            "--n-iter",
            "4",
            "--max-frames",
            "48",
        ]
    )

    output = tmp_path / "hello.wav"

    assert exit_code == 0
    assert output.is_file()
    assert output.stat().st_size > 0

    audio, sample_rate = sf.read(str(output))

    assert sample_rate == 22050
    assert len(audio) > 0


def test_generate_reports_valid_audio_metadata(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    args = make_args(tmp_path, checkpoint)

    summaries = generate(args)

    assert len(summaries) == 1

    summary = summaries[0]

    assert summary["text"] == "Hello world."
    assert summary["sample_rate"] == 22050
    assert summary["samples"] > 0
    assert summary["duration_seconds"] > 0
    assert 0 < summary["peak"] <= 1.0
    assert summary["output"].endswith("out.wav")


def test_generate_writes_multiple_sentences(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")

    exit_code = main(
        [
            "First sentence.",
            "Second sentence.",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "samples"),
            "--device",
            "cpu",
            "--n-iter",
            "4",
            "--max-frames",
            "48",
        ]
    )

    files = sorted((tmp_path / "samples").glob("*.wav"))

    assert exit_code == 0
    assert len(files) == 2

    for path in files:
        audio, sample_rate = sf.read(str(path))
        assert sample_rate == 22050
        assert len(audio) > 0


def test_generate_rejects_missing_checkpoint(tmp_path):
    args = build_parser().parse_args(
        [
            "Hello.",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--device",
            "cpu",
        ]
    )

    with pytest.raises(FileNotFoundError):
        generate(args)


def test_main_returns_error_code_for_missing_checkpoint(tmp_path):
    exit_code = main(
        [
            "Hello.",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 1


def test_generate_requires_text(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    args = build_parser().parse_args(
        ["--checkpoint", str(checkpoint), "--device", "cpu"]
    )

    with pytest.raises(ValueError, match="No text was provided"):
        generate(args)


def test_single_output_option_rejects_multiple_sentences(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    args = build_parser().parse_args(
        [
            "One.",
            "Two.",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "one.wav"),
            "--device",
            "cpu",
        ]
    )

    with pytest.raises(ValueError, match="single sentence"):
        generate(args)


def test_generate_reads_text_file(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    text_file = tmp_path / "sentences.txt"
    text_file.write_text("First line.\n\nSecond line.\n", encoding="utf-8")

    exit_code = main(
        [
            "--text-file",
            str(text_file),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "from_file"),
            "--device",
            "cpu",
            "--n-iter",
            "4",
            "--max-frames",
            "48",
        ]
    )

    files = sorted((tmp_path / "from_file").glob("*.wav"))

    assert exit_code == 0
    assert len(files) == 2


def test_generate_loads_legacy_checkpoint_without_config(tmp_path):
    """Old checkpoints predate ``model_config`` and must still load."""

    checkpoint = write_checkpoint(tmp_path / "legacy.pt", with_config=False)
    args = make_args(tmp_path, checkpoint)

    summaries = generate(args)

    assert summaries[0]["samples"] > 0


def test_generate_summary_is_json_serialisable(tmp_path, capsys):
    checkpoint = write_checkpoint(tmp_path / "model.pt")

    main(
        [
            "Hello world.",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "hello.wav"),
            "--device",
            "cpu",
            "--n-iter",
            "4",
            "--max-frames",
            "48",
        ]
    )

    captured = capsys.readouterr()

    assert "Summary:" in captured.out
    assert "hello.wav" in captured.out

    payload = captured.out.split("Summary:", 1)[1].strip()
    parsed = json.loads(payload)

    assert parsed[0]["text"] == "Hello world."


def test_generated_audio_is_finite_and_not_silent(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    args = make_args(tmp_path, checkpoint)

    summaries = generate(args)

    audio, _ = sf.read(summaries[0]["output"])

    assert abs(audio).max() > 0
    assert summaries[0]["peak"] > 0.5
