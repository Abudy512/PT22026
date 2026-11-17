
"""LJSpeech dataset loader for VoiceForge."""

from __future__ import annotations

import warnings
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from voiceforge.audio import SAMPLE_RATE, load_audio, mel_spectrogram
from voiceforge.text import get_unsupported_characters, text_to_sequence


def clean_transcript(text: str) -> str:
    """
    Remove a stray opening or closing quotation mark.

    A handful of LJSpeech rows begin with an unclosed ``"`` that continues
    onto the next row.  Keeping it would teach the model to say a quotation
    mark, so an *unmatched* surrounding quote is removed.  Balanced quotes
    (for example ``"hello," she said``) are left untouched.
    """

    text = text.strip()

    if text.count('"') != 1:
        return text

    if text.startswith('"'):
        return text[1:].strip()

    if text.endswith('"'):
        return text[:-1].strip()

    return text


class LJSpeechDataset(Dataset):
    """
    Loads text and audio samples from the LJSpeech dataset.

    Expected dataset structure::

        LJSpeech/
        ├── metadata.csv
        └── wavs/
            ├── LJ001-0001.wav
            ├── LJ001-0002.wav
            └── ...

    ``metadata.csv`` is pipe-separated with three columns:
    ``id|raw transcription|normalised transcription``.

    Optional filters
    ----------------
    ``max_duration_seconds``
        Keep only recordings shorter than this.  Training on short clips
        reduces the number of autoregressive decoder steps per batch, which
        matters a lot on a small GPU.
    ``max_mel_frames``
        Truncate each Mel spectrogram to this many frames.
    ``strict``
        When True (default) a missing WAV file raises ``FileNotFoundError``.
        When False the row is skipped with a warning, which is useful for
        real-world datasets with a few damaged entries.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        wavs_dir: str | Path | None = None,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = 80,
        max_duration_seconds: float | None = None,
        max_mel_frames: int | None = None,
        strict: bool = True,
    ):
        self.metadata_path = Path(metadata_path)

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        if max_duration_seconds is not None and max_duration_seconds <= 0:
            raise ValueError(
                "Maximum duration must be greater than zero"
            )

        if max_mel_frames is not None and max_mel_frames <= 0:
            raise ValueError(
                "Maximum Mel frames must be greater than zero"
            )

        self.wavs_dir = (
            Path(wavs_dir)
            if wavs_dir is not None
            else self.metadata_path.parent / "wavs"
        )

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration_seconds = max_duration_seconds
        self.max_mel_frames = max_mel_frames

        self.samples: list[tuple[str, Path]] = []
        self.skipped: list[str] = []

        with self.metadata_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                columns = line.split("|")

                if len(columns) < 2:
                    raise ValueError(
                        f"Invalid metadata at line {line_number}"
                    )

                audio_id = columns[0].strip()
                text = columns[2] if len(columns) >= 3 else columns[1]

                if not text.strip():
                    text = columns[1]

                text = clean_transcript(text)

                # The vocabulary is fixed at load time: reject (or skip)
                # rows whose transcript the model could never encode.
                unsupported = get_unsupported_characters(text)

                if unsupported:
                    message = (
                        f"Unsupported characters {unsupported} in "
                        f"transcript at line {line_number}"
                    )

                    if strict:
                        raise ValueError(message)

                    self.skipped.append(f"line {line_number}: {message}")
                    continue

                audio_path = self.wavs_dir / f"{audio_id}.wav"

                if not audio_path.is_file():
                    if strict:
                        raise FileNotFoundError(
                            f"Audio file not found: {audio_path}"
                        )

                    self.skipped.append(str(audio_path))
                    continue

                if max_duration_seconds is not None:
                    duration = sf.info(str(audio_path)).duration

                    if duration > max_duration_seconds:
                        continue

                self.samples.append((text, audio_path))

        if not self.samples:
            raise ValueError(
                "No usable samples were found. Check the metadata path, "
                "the wavs folder and the duration filter."
            )

        if self.skipped:
            warnings.warn(
                f"{len(self.skipped)} dataset rows were skipped "
                "(missing audio or unsupported transcript characters).",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        text, audio_path = self.samples[index]

        waveform = load_audio(
            audio_path,
            target_sample_rate=self.sample_rate,
        )

        if self.max_mel_frames is not None:
            # Cut the audio first so the waveform and the Mel spectrogram
            # describe exactly the same signal.
            max_samples = self.max_mel_frames * 256
            waveform = waveform[:, :max_samples]

        text_sequence = torch.tensor(
            text_to_sequence(text),
            dtype=torch.long,
        )

        mel = mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
        )

        if self.max_mel_frames is not None:
            # Mel frames are counted with ``center=True``, so cutting the
            # waveform alone still yields one extra frame.
            mel = mel[:, :, :self.max_mel_frames]

        return {
            "text": text,
            "text_sequence": text_sequence,
            "waveform": waveform,
            "mel": mel,
        }
