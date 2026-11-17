
"""Tests for the attention-based acoustic model."""

import pytest
import torch

from voiceforge.model import AcousticModel


def create_test_inputs():
    text = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 0, 0],
    ])

    text_lengths = torch.tensor([5, 3])

    return text, text_lengths


def test_model_output_shape():
    model = AcousticModel()
    text, text_lengths = create_test_inputs()
    mel_lengths = torch.tensor([10, 6])

    output = model(text, text_lengths, mel_lengths)

    assert output.shape == (2, 80, 10)


def test_model_output_is_finite():
    model = AcousticModel()
    text, text_lengths = create_test_inputs()
    mel_lengths = torch.tensor([10, 6])

    output = model(text, text_lengths, mel_lengths)

    assert torch.isfinite(output).all()


def test_model_supports_training():
    model = AcousticModel()
    text, text_lengths = create_test_inputs()
    mel_lengths = torch.tensor([10, 6])

    output = model(text, text_lengths, mel_lengths)
    loss = output.abs().mean()

    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.mel_projection.weight.grad is not None


def test_model_predicts_positive_mel_lengths():
    model = AcousticModel()
    model.eval()

    text, text_lengths = create_test_inputs()

    with torch.no_grad():
        lengths = model.predict_mel_lengths(
            text,
            text_lengths,
        )

    assert lengths.shape == (2,)
    assert torch.all(lengths > 0)


def test_model_generates_mel_without_target_lengths():
    model = AcousticModel()
    model.eval()

    text, text_lengths = create_test_inputs()

    with torch.no_grad():
        predicted_lengths = model.predict_mel_lengths(
            text,
            text_lengths,
        )

        mel = model(text, text_lengths)

    assert mel.shape == (
        2,
        80,
        int(predicted_lengths.max().item()),
    )

    assert torch.isfinite(mel).all()


def test_forward_train_returns_all_outputs(tiny_model):
    text, text_lengths = create_test_inputs()
    targets = torch.randn(2, 80, 8)
    mel_lengths = torch.tensor([8, 5])

    output = tiny_model.forward_train(
        text,
        text_lengths,
        targets,
        mel_lengths,
    )

    assert output["mel"].shape == (2, 80, 8)
    assert output["mel_postnet"].shape == (2, 80, 8)
    assert output["stop_logits"].shape == (2, 8)
    assert output["attention"].shape == (2, 8, 5)


def test_attention_weights_are_normalised(tiny_model):
    text, text_lengths = create_test_inputs()
    targets = torch.randn(2, 80, 6)
    mel_lengths = torch.tensor([6, 4])

    output = tiny_model.forward_train(
        text,
        text_lengths,
        targets,
        mel_lengths,
    )

    attention = output["attention"]

    # Every frame attends to exactly one distribution over characters.
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones(2, 6),
        atol=1e-5,
    )


def test_attention_ignores_padding(tiny_model):
    text, text_lengths = create_test_inputs()
    targets = torch.randn(2, 80, 6)
    mel_lengths = torch.tensor([6, 4])

    output = tiny_model.forward_train(
        text,
        text_lengths,
        targets,
        mel_lengths,
    )

    attention = output["attention"]

    # Sample 1 has only three real characters; positions 3 and 4 are padding.
    assert attention[1, :, 3:].abs().max().item() < 1e-6


def test_postnet_adds_a_residual(tiny_model):
    text, text_lengths = create_test_inputs()
    targets = torch.randn(2, 80, 6)
    mel_lengths = torch.tensor([6, 6])

    output = tiny_model.forward_train(
        text,
        text_lengths,
        targets,
        mel_lengths,
    )

    difference = (output["mel_postnet"] - output["mel"]).abs().max()
    assert difference.item() > 0


def test_generate_with_explicit_lengths(tiny_model):
    text, text_lengths = create_test_inputs()

    mel, lengths = tiny_model.generate(
        text,
        text_lengths,
        lengths=torch.tensor([12, 6]),
    )

    assert mel.shape == (2, 80, 12)
    assert lengths.tolist() == [12, 6]

    # Frames past a sample's length must be zeroed.
    assert mel[1, :, 6:].abs().max().item() == 0.0


def test_generate_respects_max_frames(tiny_model):
    text = torch.tensor([[1, 2, 3]])
    text_lengths = torch.tensor([3])

    mel, lengths = tiny_model.generate(
        text,
        text_lengths,
        lengths=torch.tensor([500]),
        max_frames=20,
    )

    assert mel.shape == (1, 80, 20)
    assert lengths.tolist() == [20]


def test_generate_with_stop_token_terminates(tiny_model):
    text = torch.tensor([[1, 2, 3, 4]])
    text_lengths = torch.tensor([4])

    mel, _ = tiny_model.generate(
        text,
        text_lengths,
        lengths=torch.tensor([50]),
        use_stop_token=True,
    )

    assert mel.shape[2] >= 1
    assert torch.isfinite(mel).all()


def test_model_rejects_zero_text_length():
    model = AcousticModel()
    text = torch.tensor([[1, 2, 3]])
    text_lengths = torch.tensor([0])

    with pytest.raises(ValueError, match="greater than zero"):
        model(text, text_lengths, torch.tensor([5]))


def test_model_rejects_too_long_text_length():
    model = AcousticModel()
    text = torch.tensor([[1, 2, 3]])
    text_lengths = torch.tensor([9])

    with pytest.raises(ValueError, match="cannot exceed"):
        model(text, text_lengths, torch.tensor([5]))


def test_model_rejects_two_dimensional_mel_lengths():
    model = AcousticModel()
    text = torch.tensor([[1, 2, 3]])
    text_lengths = torch.tensor([3])

    with pytest.raises(ValueError, match="one-dimensional"):
        model(text, text_lengths, torch.tensor([[5]]))


def test_model_config_is_stored_on_instance():
    model = AcousticModel(hidden_dim=64, attention_dim=32)

    assert model.config.hidden_dim == 64
    assert model.config.attention_dim == 32
    assert model.config.architecture == "tacotron_lite"
    assert model.config.n_mels == 80


def test_encoder_memory_shape(tiny_model):
    text, text_lengths = create_test_inputs()

    memory, mask = tiny_model.encode(text, text_lengths)

    assert memory.shape == (2, 5, tiny_model.hidden_dim)
    assert mask.shape == (2, 5)
    assert mask[1].tolist() == [True, True, True, False, False]
