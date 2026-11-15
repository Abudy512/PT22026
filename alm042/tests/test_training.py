
import torch

from voiceforge.model import AcousticModel
from voiceforge.text import CHAR_TO_ID, text_to_sequence
from voiceforge.training import train_model


def create_training_batch():
    texts = [
        torch.tensor(text_to_sequence("cat"), dtype=torch.long),
        torch.tensor(text_to_sequence("hello"), dtype=torch.long),
    ]

    padded_texts = torch.nn.utils.rnn.pad_sequence(
        texts,
        batch_first=True,
        padding_value=0,
    )

    return {
        "text": padded_texts,
        "text_lengths": torch.tensor([3, 5]),
        "mel": torch.randn(2, 80, 16),
        "mel_lengths": torch.tensor([12, 16]),
    }


def create_model():
    return AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        n_mels=80,
        embedding_dim=32,
        hidden_dim=32,
    )


def test_training_returns_finite_loss():
    model = create_model()
    batch = create_training_batch()

    history = train_model(
        model,
        [batch],
        epochs=1,
        device="cpu",
    )

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]))
    assert history[0] >= 0


def test_training_updates_model_weights():
    model = create_model()

    original_weights = (
        model.mel_projection.weight.detach().clone()
    )

    batch = create_training_batch()

    train_model(
        model,
        [batch],
        epochs=1,
        device="cpu",
    )

    updated_weights = model.mel_projection.weight.detach()

    assert not torch.equal(original_weights, updated_weights)


def test_training_saves_checkpoint(tmp_path):
    model = create_model()
    batch = create_training_batch()

    checkpoint_path = tmp_path / "checkpoints" / "model.pt"

    train_model(
        model,
        [batch],
        epochs=1,
        device="cpu",
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.is_file()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "loss" in checkpoint
