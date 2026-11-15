
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(samples):
    """
    Combine individual dataset samples into one training batch.

    Each sample must be either:
      - A dictionary containing "text" and "mel"
      - A tuple containing (text, mel)

    Text shape: (text_length,)
    Mel shape:  (n_mels, mel_length)
    """

    texts = []
    mels = []

    for sample in samples:
        if isinstance(sample, dict):
            text = sample["text"]
            mel = sample["mel"]
        else:
            text, mel = sample

        text = torch.as_tensor(text, dtype=torch.long)
        mel = torch.as_tensor(mel, dtype=torch.float32)

        # Remove an optional batch dimension from the Mel spectrogram.
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)

        if text.ndim != 1:
            raise ValueError("Each text sequence must be one-dimensional")

        if mel.ndim != 2:
            raise ValueError("Each Mel spectrogram must be two-dimensional")

        if text.numel() == 0 or mel.shape[1] == 0:
            raise ValueError("Text and Mel sequences cannot be empty")

        texts.append(text)
        mels.append(mel)

    text_lengths = torch.tensor(
        [text.numel() for text in texts],
        dtype=torch.long,
    )

    mel_lengths = torch.tensor(
        [mel.shape[1] for mel in mels],
        dtype=torch.long,
    )

    padded_texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=0,
    )

    # Pad Mel spectrograms along their time dimension.
    padded_mels = pad_sequence(
        [mel.transpose(0, 1) for mel in mels],
        batch_first=True,
        padding_value=0.0,
    ).transpose(1, 2)

    return {
        "text": padded_texts,
        "text_lengths": text_lengths,
        "mel": padded_mels,
        "mel_lengths": mel_lengths,
    }
