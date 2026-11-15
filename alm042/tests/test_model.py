
import torch
from torch.nn.utils.rnn import pad_sequence

from voiceforge.model import AcousticModel
from voiceforge.text import CHAR_TO_ID, text_to_sequence


def create_test_inputs():
    sequences = [
        torch.tensor(text_to_sequence("cat"), dtype=torch.long),
        torch.tensor(text_to_sequence("hello"), dtype=torch.long),
    ]

    texts = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )

    text_lengths = torch.tensor([3, 5])
    mel_lengths = torch.tensor([12, 16])

    return texts, text_lengths, mel_lengths


def test_acoustic_model_output_shape():
    model = AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        n_mels=80,
        embedding_dim=32,
        hidden_dim=32,
    )

    texts, text_lengths, mel_lengths = create_test_inputs()

    output = model(texts, text_lengths, mel_lengths)

    assert output.shape == (2, 80, 16)


def test_acoustic_model_output_is_finite():
    model = AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        embedding_dim=32,
        hidden_dim=32,
    )

    texts, text_lengths, mel_lengths = create_test_inputs()

    output = model(texts, text_lengths, mel_lengths)

    assert torch.isfinite(output).all()


def test_acoustic_model_supports_training():
    model = AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        embedding_dim=32,
        hidden_dim=32,
    )

    texts, text_lengths, mel_lengths = create_test_inputs()

    output = model(texts, text_lengths, mel_lengths)

    loss = output.mean()
    loss.backward()

    assert model.mel_projection.weight.grad is not None
    assert torch.isfinite(model.mel_projection.weight.grad).all()
