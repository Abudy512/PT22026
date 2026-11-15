
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(samples):
    """
    Combine dataset samples into one padded training batch.

    Supports:
      - Dictionary samples using text_sequence and mel.
      - Dictionary samples using tensor-valued text and mel.
      - Tuple samples containing (text, mel).
    """

    texts = []
    mels = []

    for sample in samples:
        if isinstance(sample, dict):
            # LJSpeechDataset stores the original sentence in "text".
            # The model needs the numerical sequence instead.
            if "text_sequence" in sample:
                text = sample["text_sequence"]
            else:
                text = sample["text"]

            mel = sample["mel"]

        else:
            text, mel = sample

        text = torch.as_tensor(text, dtype=torch.long)
        mel = torch.as_tensor(mel, dtype=torch.float32)

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
