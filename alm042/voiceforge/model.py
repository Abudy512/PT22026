
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from voiceforge.text import CHAR_TO_ID


class AcousticModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = len(CHAR_TO_ID),
        n_mels: int = 80,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.mel_projection = nn.Linear(
            hidden_dim * 2,
            n_mels,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> torch.Tensor:

        if text.ndim != 2:
            raise ValueError("Text must have shape (batch, text_length)")

        if text_lengths.ndim != 1 or mel_lengths.ndim != 1:
            raise ValueError("Length tensors must be one-dimensional")

        if text.shape[0] != text_lengths.shape[0]:
            raise ValueError("Text batch and text lengths do not match")

        if text.shape[0] != mel_lengths.shape[0]:
            raise ValueError("Text batch and Mel lengths do not match")

        if torch.any(text_lengths <= 0) or torch.any(mel_lengths <= 0):
            raise ValueError("Sequence lengths must be greater than zero")

        embedded = self.embedding(text)

        packed = pack_padded_sequence(
            embedded,
            text_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.encoder(packed)

        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        mel_sequences = []

        for index in range(text.shape[0]):
            valid_text_length = int(text_lengths[index].item())
            target_mel_length = int(mel_lengths[index].item())

            hidden = encoded[index, :valid_text_length]

            hidden = hidden.transpose(0, 1).unsqueeze(0)

            expanded = F.interpolate(
                hidden,
                size=target_mel_length,
                mode="linear",
                align_corners=False,
            )

            expanded = expanded.squeeze(0).transpose(0, 1)

            mel = self.mel_projection(expanded)

            mel_sequences.append(mel)

        padded_mels = nn.utils.rnn.pad_sequence(
            mel_sequences,
            batch_first=True,
            padding_value=0.0,
        )

        return padded_mels.transpose(1, 2)
