"""Shared pytest configuration and fixtures.

The project is imported from its root directory (``alm042``).  Adding that
folder to ``sys.path`` here means the tests work no matter which directory
pytest is started from.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voiceforge.model import AcousticModel  # noqa: E402
from voiceforge.text import CHAR_TO_ID  # noqa: E402

VOCAB_SIZE = len(CHAR_TO_ID)

SENTENCES = [
    "Hello world.",
    "This is a short test.",
    "VoiceForge speaks clearly.",
]


def build_tiny_model(embedding_dim: int = 8, hidden_dim: int = 8) -> AcousticModel:
    """Create a very small model so tests stay fast."""

    model = AcousticModel(
        vocab_size=VOCAB_SIZE,
        n_mels=80,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )

    # Make the duration predictor output roughly 20 frames instead of a
    # random (possibly single-frame) length.  exp(3.0) is about 20.
    with torch.no_grad():
        model.duration_projection[2].bias.fill_(3.0)

    return model


@pytest.fixture
def tiny_model():
    return build_tiny_model()


@pytest.fixture
def tiny_dataset(tmp_path):
    """
    Build a miniature LJSpeech-style dataset.

    Returns ``(metadata_path, wavs_dir)`` with three short recordings.
    """

    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()

    sample_rate = 22050
    seconds = 0.6

    for index, sentence in enumerate(SENTENCES, start=1):
        audio_id = f"LJ001-{index:04d}"

        time = np.arange(int(sample_rate * seconds)) / sample_rate

        # A falling tone with a soft envelope so the audio is not silent
        # and has some energy variation over time.
        envelope = np.linspace(1.0, 0.2, time.size)
        audio = 0.3 * envelope * np.sin(2 * np.pi * 220 * time)

        sf.write(
            str(wavs_dir / f"{audio_id}.wav"),
            audio.astype(np.float32),
            sample_rate,
        )

    metadata_path = tmp_path / "metadata.csv"

    rows = [
        f"LJ001-{index:04d}|{sentence}|{sentence}"
        for index, sentence in enumerate(SENTENCES, start=1)
    ]

    metadata_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return metadata_path, wavs_dir


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_model):
    """Save the tiny model as a checkpoint and return its path."""

    path = tmp_path / "checkpoints" / "tiny.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": tiny_model.state_dict(),
            "optimizer_state_dict": {},
            "loss": 1.0,
            "model_config": {
                "architecture": "legacy_interpolate",
                "vocab_size": VOCAB_SIZE,
                "n_mels": 80,
                "embedding_dim": 8,
                "hidden_dim": 8,
            },
        },
        path,
    )

    return path
