
from pathlib import Path

import torch

from voiceforge.audio import mel_to_audio, save_audio
from voiceforge.text import text_to_sequence


class VoiceForgeTTS:
    """
    Main Text-to-Speech interface for VoiceForge.

    Converts English text into speech using:
        1. Text processing
        2. The trained acoustic model
        3. Griffin-Lim audio reconstruction
    """

    def __init__(
        self,
        model,
        device=None,
        sample_rate=22050,
        griffin_lim_iterations=32,
    ):
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.model.eval()

        self.sample_rate = sample_rate
        self.griffin_lim_iterations = griffin_lim_iterations

    @torch.inference_mode()
    def synthesize(self, text: str) -> torch.Tensor:
        """
        Convert text into a waveform tensor.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Convert English text into numerical character IDs.
        sequence = text_to_sequence(text)

        if not sequence:
            raise ValueError("Text produced an empty sequence")

        text_tensor = torch.tensor(
            [sequence],
            dtype=torch.long,
            device=self.device,
        )

        text_lengths = torch.tensor(
            [len(sequence)],
            dtype=torch.long,
            device=self.device,
        )

        # Generate a Mel spectrogram with the acoustic model.
        predicted_mel = self.model(text_tensor, text_lengths)

        if predicted_mel.ndim != 3:
            raise ValueError(
                "The acoustic model must return a 3D Mel spectrogram "
                "with shape [batch, mel_channels, time]"
            )

        if predicted_mel.shape[0] != 1:
            raise ValueError("Expected exactly one generated sample")

        # Convert the generated Mel spectrogram into audio.
        mel = predicted_mel[0].detach().cpu()

        waveform = mel_to_audio(
            mel,
            n_iter=self.griffin_lim_iterations,
        )

        return waveform.detach().cpu().squeeze()

    def save(self, text: str, output_path) -> torch.Tensor:
        """
        Generate speech and save it as a WAV file.
        """

        waveform = self.synthesize(text)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_audio(
            waveform,
            str(output_path),
            sample_rate=self.sample_rate,
        )

        return waveform
