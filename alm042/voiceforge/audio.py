from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


SAMPLE_RATE = 22050


def load_audio(
    file_path: str | Path,
    target_sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:

    audio, original_sample_rate = sf.read(
        str(file_path),
        dtype="float32",
        always_2d=True,
    )

    waveform = torch.from_numpy(audio.T.copy())

    # Convert stereo audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert audio to the target sample rate
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            original_sample_rate,
            target_sample_rate,
        )

    return waveform