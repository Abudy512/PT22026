
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
