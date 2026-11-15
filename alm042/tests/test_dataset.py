
import numpy as np
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
