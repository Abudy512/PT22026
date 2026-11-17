"""
Kokoro-82M pretrained text-to-speech backend.

Kokoro is a small, high-quality open-weight TTS model released by hexgrad.
It is **not** trained by this project: VoiceForge uses it as a pretrained
engine and contributes the integration, the interface and the evaluation.
The student's own acoustic model remains available as the
``voiceforge`` backend (see ``voiceforge/tts.py``).

Resources
---------
* Model: https://huggingface.co/hexgrad/Kokoro-82M
* Code:  https://github.com/hexgrad/kokoro
* License: Apache-2.0 (see the model card)

The weights are downloaded from Hugging Face on first use and cached by
``huggingface_hub``. Set ``HF_HOME`` to move the cache, or ``HF_ENDPOINT``
to a mirror if huggingface.co is unreachable.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

#: Curated American-English voices exposed by the interface.
KOKORO_VOICES: dict[str, str] = {
    "af_heart": "American female — warm and clear (default)",
    "af_bella": "American female — expressive",
    "af_nicole": "American female — soft",
    "af_sarah": "American female — neutral",
    "am_michael": "American male — calm",
    "am_adam": "American male — deeper",
    "am_onyx": "American male — low",
}


class KokoroError(RuntimeError):
    """Raised when the Kokoro backend cannot be used."""


def kokoro_is_installed() -> bool:
    """True when the ``kokoro`` package is importable."""

    return importlib.util.find_spec("kokoro") is not None


class KokoroBackend:
    """
    Thin wrapper around :class:`kokoro.KPipeline`.

    The model is loaded lazily so starting the application stays fast, and
    the sample rate is fixed at 24 kHz, which is what Kokoro generates.
    """

    SAMPLE_RATE = KOKORO_SAMPLE_RATE

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        device: str | None = None,
        repo_id: str = KOKORO_REPO_ID,
        speed: float = 1.0,
    ):
        if voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown Kokoro voice {voice!r}. "
                f"Available: {', '.join(KOKORO_VOICES)}"
            )

        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")

        self.voice = voice
        self.repo_id = repo_id
        self.speed = speed
        self.requested_device = device

        self._pipeline = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def device(self) -> str:
        if self._pipeline is not None:
            value = getattr(self._pipeline, "device", None)
            if value:
                return str(value)

        return self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load(self):
        """Create the Kokoro pipeline (downloads weights on first use)."""

        if self._pipeline is not None:
            return self._pipeline

        if not kokoro_is_installed():
            raise KokoroError(
                "The 'kokoro' package is not installed. Install it with "
                "'pip install kokoro' (and 'pip install misaki[en]' if "
                "prompted), then restart the application."
            )

        try:
            from kokoro import KPipeline

            device = self.requested_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self._pipeline = KPipeline(
                lang_code="a",
                repo_id=self.repo_id,
                device=device,
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            raise KokoroError(
                "Kokoro could not be loaded. The pretrained weights are "
                f"downloaded from Hugging Face ({self.repo_id}); check your "
                "internet connection, or set HF_ENDPOINT to a mirror. "
                f"Original error: {error}"
            ) from error

        return self._pipeline

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        """
        Generate speech and return a mono float32 waveform at 24 kHz.

        Kokoro works sentence by sentence, so the chunks are concatenated.
        """

        pipeline = self.load()

        selected_voice = voice or self.voice

        if selected_voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown Kokoro voice {selected_voice!r}. "
                f"Available: {', '.join(KOKORO_VOICES)}"
            )

        chunks: list[np.ndarray] = []

        try:
            for _, _, audio in pipeline(
                text,
                voice=selected_voice,
                speed=self.speed,
            ):
                chunks.append(
                    audio.detach().cpu().numpy()
                    if torch.is_tensor(audio)
                    else np.asarray(audio)
                )
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            raise KokoroError(
                f"Kokoro could not generate speech: {error}"
            ) from error

        if not chunks:
            raise KokoroError("Kokoro returned no audio for this text.")

        waveform = np.concatenate(chunks).astype(np.float32)

        if waveform.size == 0:
            raise KokoroError("Kokoro returned an empty waveform.")

        if not np.isfinite(waveform).all():
            raise KokoroError("Kokoro produced non-finite audio samples.")

        return waveform

    def save(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
    ) -> Path:
        """Synthesise ``text`` and write a 24 kHz 16-bit PCM WAV file."""

        waveform = self.synthesize(text, voice=voice)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            str(output_path),
            waveform,
            self.SAMPLE_RATE,
            subtype="PCM_16",
        )

        return output_path


def describe_environment() -> dict:
    """Information for the health endpoint and the startup banner."""

    return {
        "installed": kokoro_is_installed(),
        "sample_rate": KOKORO_SAMPLE_RATE,
        "repo_id": KOKORO_REPO_ID,
        "voices": list(KOKORO_VOICES),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }
