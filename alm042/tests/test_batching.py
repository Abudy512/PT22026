
import torch

from voiceforge.batching import collate_batch


def test_collate_batch_output_shapes():
    samples = [
        {
            "text": torch.tensor([3, 1, 20]),
            "mel": torch.randn(80, 12),
        },
        {
            "text": torch.tensor([8, 5, 12, 12, 15]),
            "mel": torch.randn(80, 16),
        },
    ]

    batch = collate_batch(samples)

    assert batch["text"].shape == (2, 5)
    assert batch["text_lengths"].tolist() == [3, 5]
    assert batch["mel"].shape == (2, 80, 16)
    assert batch["mel_lengths"].tolist() == [12, 16]


def test_collate_batch_pads_text_with_zero():
    samples = [
        {
            "text": torch.tensor([3, 1, 20]),
            "mel": torch.randn(80, 12),
        },
        {
            "text": torch.tensor([8, 5, 12, 12, 15]),
            "mel": torch.randn(80, 16),
        },
    ]

    batch = collate_batch(samples)

    assert batch["text"][0, 3:].tolist() == [0, 0]


def test_collate_batch_supports_tuple_samples():
    samples = [
        (torch.tensor([3, 1, 20]), torch.randn(80, 12)),
        (torch.tensor([8, 5]), torch.randn(80, 10)),
    ]

    batch = collate_batch(samples)

    assert batch["text"].shape == (2, 3)
    assert batch["mel"].shape == (2, 80, 12)


def test_collate_batch_rejects_invalid_mel():
    samples = [
        {
            "text": torch.tensor([3, 1, 20]),
            "mel": torch.randn(80),
        }
    ]

    try:
        collate_batch(samples)
        assert False, "Expected ValueError"
    except ValueError:
        pass
