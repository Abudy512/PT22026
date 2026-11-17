
"""Tests for the LJSpeech dataset loader."""

import numpy as np
import pytest
import soundfile as sf
import torch

from voiceforge.dataset import LJSpeechDataset


def create_test_dataset(tmp_path):
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()

    audio = np.random.uniform(
        -1, 1, 22050
    ).astype(np.float32)

    sf.write(
        str(wavs_dir / "LJ001-0001.wav"),
        audio,
        22050,
    )

    metadata_path = tmp_path / "metadata.csv"

    metadata_path.write_text(
        "LJ001-0001|Hello World!|hello world!\n",
        encoding="utf-8",
    )

    return metadata_path


def create_multi_dataset(tmp_path):
    """Create three recordings of different lengths."""

    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()

    durations = [0.3, 0.4, 2.0]
    rows = []
    words = ["First", "Second", "Third"]

    for index, seconds in enumerate(durations):
        audio_id = f"LJ002-{index + 1:04d}"

        time = np.arange(int(22050 * seconds)) / 22050
        audio = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

        sf.write(str(wavs_dir / f"{audio_id}.wav"), audio, 22050)

        text = f"{words[index]} sentence."
        rows.append(f"{audio_id}|{text}|{text.lower()}")

    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return metadata_path


def test_dataset_loads_correct_number_of_samples(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    dataset = LJSpeechDataset(metadata_path)

    assert len(dataset) == 1


def test_dataset_returns_correct_data(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    dataset = LJSpeechDataset(metadata_path)

    sample = dataset[0]

    assert sample["text"] == "hello world!"
    assert isinstance(sample["text_sequence"], torch.Tensor)
    assert isinstance(sample["waveform"], torch.Tensor)
    assert isinstance(sample["mel"], torch.Tensor)


def test_dataset_audio_and_mel_shapes(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    dataset = LJSpeechDataset(metadata_path)

    sample = dataset[0]

    assert sample["waveform"].shape[0] == 1
    assert sample["mel"].shape[0] == 1
    assert sample["mel"].shape[1] == 80
    assert torch.isfinite(sample["mel"]).all()


def test_dataset_filters_by_duration(tmp_path):
    metadata_path = create_multi_dataset(tmp_path)

    dataset = LJSpeechDataset(metadata_path, max_duration_seconds=1.0)

    assert len(dataset) == 2
    assert all("LJ002-0001" in str(path) or "LJ002-0002" in str(path)
               for _, path in dataset.samples)


def test_dataset_truncates_long_mel(tmp_path):
    metadata_path = create_multi_dataset(tmp_path)

    dataset = LJSpeechDataset(metadata_path, max_mel_frames=20)

    sample = dataset[3 - 1]

    assert sample["mel"].shape[2] <= 20
    # The waveform must be cut to the same signal length.
    assert sample["waveform"].shape[1] <= 20 * 256


def test_dataset_skips_missing_audio_when_not_strict(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    with metadata_path.open("a", encoding="utf-8") as file:
        file.write("LJ009-9999|Missing file.|missing file.\n")

    with pytest.warns(UserWarning, match="missing"):
        dataset = LJSpeechDataset(metadata_path, strict=False)

    assert len(dataset) == 1
    assert len(dataset.skipped) == 1


def test_dataset_raises_for_missing_audio_when_strict(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    with metadata_path.open("a", encoding="utf-8") as file:
        file.write("LJ009-9999|Missing file.|missing file.\n")

    with pytest.raises(FileNotFoundError):
        LJSpeechDataset(metadata_path)


def test_dataset_rejects_empty_selection(tmp_path):
    metadata_path = create_multi_dataset(tmp_path)

    with pytest.raises(ValueError, match="No usable samples"):
        LJSpeechDataset(metadata_path, max_duration_seconds=0.05)


def test_dataset_rejects_invalid_limits(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    with pytest.raises(ValueError, match="Maximum duration"):
        LJSpeechDataset(metadata_path, max_duration_seconds=0)

    with pytest.raises(ValueError, match="Maximum Mel frames"):
        LJSpeechDataset(metadata_path, max_mel_frames=0)


def test_clean_transcript_strips_unmatched_quote():
    from voiceforge.dataset import clean_transcript

    assert clean_transcript('"an unclosed quotation,') == "an unclosed quotation,"
    assert clean_transcript('a trailing quotation,"') == "a trailing quotation,"


def test_clean_transcript_keeps_balanced_quotes():
    from voiceforge.dataset import clean_transcript

    text = '"hello," she said'

    assert clean_transcript(text) == text


def test_dataset_cleans_unmatched_quote_from_real_row_shape(tmp_path):
    """LJSpeech has rows shaped like ``id|"text|"text`` (unclosed quote)."""

    metadata_path = create_test_dataset(tmp_path)

    with metadata_path.open("a", encoding="utf-8") as file:
        file.write('LJ001-0001|"a quoted line,|"a quoted line,\n')

    dataset = LJSpeechDataset(metadata_path)

    text = dataset.samples[-1][0]

    assert '"' not in text
    assert text.startswith("a quoted line")


def test_dataset_rejects_unsupported_transcript_when_strict(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    with metadata_path.open("a", encoding="utf-8") as file:
        file.write("LJ001-0001|Numbers 123|numbers 123\n")

    with pytest.raises(ValueError, match="Unsupported characters"):
        LJSpeechDataset(metadata_path)


def test_dataset_skips_unsupported_transcript_when_not_strict(tmp_path):
    metadata_path = create_test_dataset(tmp_path)

    with metadata_path.open("a", encoding="utf-8") as file:
        file.write("LJ001-0001|Numbers 123|numbers 123\n")

    with pytest.warns(UserWarning, match="skipped"):
        dataset = LJSpeechDataset(metadata_path, strict=False)

    assert len(dataset) == 1
    assert any("Unsupported" in entry for entry in dataset.skipped)
