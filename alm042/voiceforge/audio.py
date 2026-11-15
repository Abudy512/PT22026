
from pathlib import Path

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

    # Convert stereo audio to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert audio to the target sample rate.
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            original_sample_rate,
            target_sample_rate,
        )

    return waveform


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel = transform(waveform)

    return torch.log(mel + 1e-5)


def mel_to_audio(
    log_mel: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_iter: int = 32,
) -> torch.Tensor:
    """
    Convert a log-Mel spectrogram into a waveform using Griffin-Lim.
    """

    if log_mel.ndim == 3:
        if log_mel.shape[0] != 1:
            raise ValueError(
                "Only one Mel spectrogram can be converted at a time"
            )

        log_mel = log_mel.squeeze(0)

    if log_mel.ndim != 2:
        raise ValueError(
            "Log-Mel spectrogram must have two dimensions"
        )

    if log_mel.shape[0] != 80:
        raise ValueError(
            "Expected a Mel spectrogram with 80 Mel bands"
        )

    if log_mel.shape[1] == 0:
        raise ValueError(
            "Mel spectrogram cannot be empty"
        )

    if not torch.isfinite(log_mel).all():
        raise ValueError(
            "Mel spectrogram contains non-finite values"
        )

    if sample_rate <= 0 or n_iter < 1:
        raise ValueError(
            "Sample rate and iteration count must be positive"
        )

    log_mel = log_mel.float()

    # Reverse the logarithm used during preprocessing.
    mel_power = (log_mel.exp() - 1e-5).clamp_min(0.0)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=80,
        sample_rate=sample_rate,
    )

    spectrogram = inverse_mel(mel_power)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=2.0,
        n_iter=n_iter,
    )

    waveform = griffin_lim(spectrogram)

    return waveform


def save_audio(
    waveform: torch.Tensor,
    file_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Save a waveform as a WAV audio file."""

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero"
        )

    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)

    if waveform.ndim != 1:
        raise ValueError(
            "Waveform must be one-dimensional or have shape (1, samples)"
        )

    if waveform.numel() == 0:
        raise ValueError(
            "Waveform cannot be empty"
        )

    if not torch.isfinite(waveform).all():
        raise ValueError(
            "Waveform contains non-finite values"
        )

    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_array = waveform.detach().cpu().numpy()

    sf.write(
        str(output_path),
        audio_array,
        sample_rate,
        subtype="PCM_16",
    )

    return output_path
