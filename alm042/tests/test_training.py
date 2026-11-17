
"""Tests for the training loop, losses and checkpointing."""

from __future__ import annotations

import json

import pytest
import torch

from voiceforge.model import AcousticModel
from voiceforge.text import CHAR_TO_ID, text_to_sequence
from voiceforge.training import (
    compute_losses,
    guided_attention_loss,
    train_model,
)


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
        attention_dim=16,
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
    # The architecture must be stored so the model can be rebuilt later.
    assert checkpoint["model_config"]["architecture"] == "tacotron_lite"


def test_training_rejects_invalid_epochs():
    with pytest.raises(ValueError, match="at least 1"):
        train_model(create_model(), [], epochs=0, device="cpu")


def test_training_rejects_invalid_learning_rate():
    with pytest.raises(ValueError, match="greater than zero"):
        train_model(create_model(), [], epochs=1, learning_rate=0, device="cpu")


def test_training_rejects_empty_dataloader():
    with pytest.raises(ValueError, match="empty"):
        train_model(create_model(), [], epochs=1, device="cpu")


def test_compute_losses_returns_all_terms():
    model = create_model()
    batch = create_training_batch()

    total, terms = compute_losses(
        model,
        batch,
        torch.device("cpu"),
    )

    assert torch.isfinite(total)
    assert set(terms) == {
        "mel",
        "postnet",
        "stop",
        "duration",
        "attention",
    }
    assert all(value >= 0 for value in terms.values())


def test_guided_attention_prefers_diagonal_alignment():
    frames, characters = 8, 8

    diagonal = torch.eye(frames, characters).unsqueeze(0)
    reversed_diagonal = torch.flip(diagonal, dims=[2])
    lengths = torch.tensor([characters])

    good = guided_attention_loss(diagonal, lengths)
    bad = guided_attention_loss(reversed_diagonal, lengths)

    assert good < bad
    assert good < 0.1


def test_guided_attention_ignores_padding():
    attention = torch.eye(4, 8).unsqueeze(0)
    lengths = torch.tensor([4])

    loss = guided_attention_loss(attention, lengths)

    assert torch.isfinite(loss)


def test_training_with_validation_records_history(tmp_path):
    model = create_model()
    batch = create_training_batch()

    history_path = tmp_path / "history.json"
    checkpoint_path = tmp_path / "best.pt"

    train_model(
        model,
        [batch],
        epochs=2,
        device="cpu",
        checkpoint_path=checkpoint_path,
        validation_dataloader=[batch],
        history_path=history_path,
    )

    entries = json.loads(history_path.read_text(encoding="utf-8"))

    assert len(entries) == 2
    assert entries[0]["epoch"] == 1
    assert entries[0]["val_loss"] is not None
    assert "train_loss" in entries[1]
    assert checkpoint_path.is_file()


def test_training_validation_is_deterministic_for_fixed_model():
    model = create_model()
    batch = create_training_batch()

    model.eval()

    with torch.inference_mode():
        first, _ = compute_losses(model, batch, torch.device("cpu"))
        second, _ = compute_losses(model, batch, torch.device("cpu"))

    assert torch.equal(first, second)
