"""
Application logic shared by every VoiceForge user interface.

Keeping the validation, model loading and file handling here means the web
interface, the optional Gradio interface and the tests all exercise exactly
the same code path - there is no second, "demo-only" implementation that
could behave differently from the real pipeline.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from voiceforge.text import normalize_text, text_to_sequence
from voiceforge.tts import VoiceForgeTTS

#: Longest text the interface accepts.  The model was trained on short
#: LJSpeech clips, so long paragraphs only produce long, unreliable output.
MAX_TEXT_LENGTH = 400

#: Default checkpoint used by the application.
DEFAULT_CHECKPOINT = "checkpoints/voiceforge_attn.pt"

#: Default folder for generated audio (git-ignored).
DEFAULT_OUTPUT_DIR = "outputs/app"


class AppError(Exception):
    """An error whose message is safe and useful to show to the user."""


@dataclass(frozen=True)
class SynthesisResult:
    """Everything the interface needs to display one generation."""

    text: str
    path: str
    filename: str
    duration_seconds: float
    truncated: bool
    padded: bool


class VoiceForgeApp:
    """
    Small facade around :class:`VoiceForgeTTS` for interactive use.

    The model is loaded lazily on the first request so the interface can
    start quickly, and a lock makes it safe for a multi-threaded web server.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        griffin_lim_iterations: int = 32,
        max_mel_frames: int = 2200,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.output_dir = Path(output_dir)
        self.griffin_lim_iterations = griffin_lim_iterations
        self.max_mel_frames = max_mel_frames

        self._pipeline: VoiceForgeTTS | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> VoiceForgeTTS:
        """Load the checkpoint once and reuse it."""

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            if not self.checkpoint_path.is_file():
                raise AppError(
                    f"Model checkpoint not found: {self.checkpoint_path}. "
                    "Train one with 'python -m scripts.train' or point "
                    "--checkpoint at an existing file."
                )

            try:
                self._pipeline = VoiceForgeTTS.from_pretrained(
                    self.checkpoint_path,
                    device=self.device,
                    griffin_lim_iterations=self.griffin_lim_iterations,
                    max_mel_frames=self.max_mel_frames,
                )
            except (ValueError, RuntimeError) as error:
                raise AppError(
                    f"The checkpoint could not be loaded: {error}"
                ) from error

            return self._pipeline

    def device_description(self) -> str:
        """Human-readable device name, loading the model if necessary."""

        pipeline = self.load()
        return str(pipeline.device)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_text(text: object) -> str:
        """
        Check user input and return the cleaned text.

        Raises :class:`AppError` with a message that can be shown directly
        in the interface.
        """

        if not isinstance(text, str):
            raise AppError("Please enter text as a string.")

        cleaned = text.strip()

        if not cleaned:
            raise AppError("Please enter some English text first.")

        if len(cleaned) > MAX_TEXT_LENGTH:
            raise AppError(
                f"Text is too long ({len(cleaned)} characters). "
                f"Please keep it under {MAX_TEXT_LENGTH} characters, "
                "because the model was trained on short sentences."
            )

        if not normalize_text(cleaned):
            raise AppError(
                "The text contains no usable characters. "
                "Please enter English words."
            )

        try:
            text_to_sequence(cleaned)
        except ValueError as error:
            raise AppError(
                f"{error}. VoiceForge supports English letters, digits are "
                "not supported, and the punctuation . , ! ? - ' \" ; : ( )"
            ) from error

        return cleaned

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def synthesize(self, text: object) -> SynthesisResult:
        """Validate ``text``, generate speech and return the WAV details."""

        cleaned = self.validate_text(text)
        pipeline = self.load()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4().hex[:12]}.wav"
        output_path = self.output_dir / filename

        # The pipeline holds CUDA state; one generation at a time.
        with self._lock:
            pipeline.save(cleaned, output_path)
            truncated = bool(pipeline.was_truncated)
            padded = bool(pipeline.was_padded)

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                "Speech generation failed: the audio file was not written."
            )

        info = sf.info(str(output_path))

        if info.frames == 0:
            raise AppError("Speech generation produced an empty audio file.")

        return SynthesisResult(
            text=cleaned,
            path=str(output_path),
            filename=filename,
            duration_seconds=round(info.frames / info.samplerate, 2),
            truncated=truncated,
            padded=padded,
        )

    def resolve_audio(self, filename: str) -> Path:
        """
        Return the path of a generated file, refusing anything outside the
        output folder.
        """

        if not filename or "/" in filename or "\\" in filename:
            raise AppError("Invalid audio file name.")

        if not filename.endswith(".wav"):
            raise AppError("Invalid audio file name.")

        candidate = (self.output_dir / filename).resolve()
        output_root = self.output_dir.resolve()

        if not candidate.is_file() or output_root not in candidate.parents:
            raise AppError("Audio file not found.")

        return candidate
