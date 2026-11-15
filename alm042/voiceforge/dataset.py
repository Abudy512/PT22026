
from pathlib import Path

import torch
from torch.utils.data import Dataset

from voiceforge.audio import SAMPLE_RATE, load_audio, mel_spectrogram
from voiceforge.text import text_to_sequence


class LJSpeechDataset(Dataset):
    """
    Loads text and audio samples from the LJSpeech dataset.

    Expected dataset structure:

    LJSpeech/
    ├── metadata.csv
    └── wavs/
        ├── LJ001-0001.wav
        ├── LJ001-0002.wav
        └── ...
    """

    def __init__(
        self,
        metadata_path: str | Path,
        wavs_dir: str | Path | None = None,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = 80,
    ):
        self.metadata_path = Path(metadata_path)

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.wavs_dir = (
            Path(wavs_dir)
            if wavs_dir is not None
            else self.metadata_path.parent / "wavs"
        )

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.samples = []

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

                audio_id = columns[0]
                text = columns[2] if len(columns) >= 3 else columns[1]

                if not text.strip():
                    text = columns[1]

                audio_path = self.wavs_dir / f"{audio_id}.wav"

                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path}"
                    )

                self.samples.append((text, audio_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text, audio_path = self.samples[index]

        waveform = load_audio(
            audio_path,
            target_sample_rate=self.sample_rate,
        )

        text_sequence = torch.tensor(
            text_to_sequence(text),
            dtype=torch.long,
        )

        mel = mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
        )

        return {
            "text": text,
            "text_sequence": text_sequence,
            "waveform": waveform,
            "mel": mel,
        }
