"""
Background job processing for long-text and batch synthesis.

One worker thread consumes a bounded queue. Synthesis itself stays in
:class:`voiceforge.app_core.VoiceForgeApp`, which serialises model access
with its own lock, so jobs never run models concurrently.

Each job splits every item into engine-sized chunks, synthesises them,
joins the audio and reports real per-item progress (no fake percentages).
"""

from __future__ import annotations

import io
import json
import queue
import threading
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from voiceforge.app_core import (
    ENGINE_KOKORO,
    AppError,
    VoiceForgeApp,
)
from voiceforge.audio_join import AudioJoinError, join_wav_files
from voiceforge.kokoro_backend import KOKORO_VOICES, validate_speed
from voiceforge.text_split import MAX_LONG_TEXT_CHARS, split_text
from voiceforge.uploads import sanitize_filename

# --- statuses -------------------------------------------------------------
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED)

JOB_TYPES = ("long_text", "batch")

#: Chunk sizes per engine. Both stay inside the short-text limit of 400
#: characters, so every chunk goes through the normal validation path.
CHUNK_LIMITS = {
    ENGINE_KOKORO: 380,
    "voiceforge": 250,
}

#: Bounded queue / submission limits.
MAX_JOBS = 4
MAX_ITEMS = 20

DEFAULT_GAP_SECONDS = 0.15

DEFAULT_JOB_ROOT = "outputs/jobs"


class JobQueueFull(RuntimeError):
    """Raised when too many jobs are queued or running."""


class JobNotFound(KeyError):
    """Raised when a job id does not exist."""


class ArchiveUnavailable(RuntimeError):
    """Raised when a job has no completed audio to archive."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobItem:
    id: str
    name: str
    text: str
    status: str = STATUS_QUEUED
    filename: str | None = None
    error: str | None = None
    chunks_total: int = 0
    chunks_completed: int = 0

    def to_dict(self, download_url: str | None = None) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "filename": self.filename,
            "error": self.error,
            "characters": len(self.text),
            "text_preview": self.text[:120],
            "chunks_total": self.chunks_total,
            "chunks_completed": self.chunks_completed,
            "download_url": download_url if self.status == STATUS_COMPLETED else None,
        }


@dataclass
class Job:
    id: str
    type: str
    engine: str
    voice: str | None
    speed: float | None
    items: list[JobItem]
    created_at: str = field(default_factory=_now)
    status: str = STATUS_QUEUED
    error: str | None = None
    finished_at: str | None = None

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.items if item.status == STATUS_COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.items if item.status == STATUS_FAILED)

    @property
    def has_results(self) -> bool:
        return self.completed_count > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "engine": self.engine,
            "voice": self.voice,
            "speed": self.speed,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "total": len(self.items),
            "completed": self.completed_count,
            "failed": self.failed_count,
            "items": [
                item.to_dict(
                    f"/api/jobs/{self.id}/audio/{item.id}"
                    if item.status == STATUS_COMPLETED
                    else None
                )
                for item in self.items
            ],
            "download_url": (
                f"/api/jobs/{self.id}/download" if self.has_results else None
            ),
        }


class JobManager:
    """Bounded background job queue with a single worker thread."""

    def __init__(
        self,
        app: VoiceForgeApp,
        job_root: str | Path = DEFAULT_JOB_ROOT,
        max_jobs: int = MAX_JOBS,
        max_items: int = MAX_ITEMS,
        gap_seconds: float = DEFAULT_GAP_SECONDS,
    ):
        self._app = app
        self.job_root = Path(job_root)
        self.max_jobs = max(1, int(max_jobs))
        self.max_items = max(1, int(max_items))
        self.gap_seconds = gap_seconds

        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[Job | None] = queue.Queue(
            maxsize=self.max_jobs
        )
        self._stopping = threading.Event()

        self._worker = threading.Thread(
            target=self._run,
            name="voiceforge-jobs",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def submit(self, payload: object) -> Job:
        """Validate a submission and queue it."""

        job = self._build_job(payload)

        with self._lock:
            active = sum(
                1
                for existing in self._jobs.values()
                if existing.status in (STATUS_QUEUED, STATUS_RUNNING)
            )

            if active >= self.max_jobs:
                raise JobQueueFull(
                    f"The queue is full ({active} jobs). "
                    "Wait for a job to finish before submitting another."
                )

            self._jobs[job.id] = job

        try:
            self._queue.put_nowait(job)
        except queue.Full as error:
            with self._lock:
                self._jobs.pop(job.id, None)

            raise JobQueueFull(
                "The queue is full. Wait for a job to finish."
            ) from error

        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)

        if job is None:
            raise JobNotFound(job_id)

        return job

    def stop(self) -> None:
        """Stop the worker and fail anything still queued."""

        if self._stopping.is_set():
            return

        self._stopping.set()

        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                break

            if pending is None:
                continue

            with self._lock:
                pending.status = STATUS_FAILED
                pending.error = "Server stopped before the job started."
                pending.finished_at = _now()

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._worker.is_alive():
            self._worker.join(timeout=5)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _build_job(self, payload: object) -> Job:
        if not isinstance(payload, dict):
            raise AppError(
                "Job payload must be a JSON object.",
                code="invalid_job",
            )

        job_type = payload.get("type") or "batch"

        if job_type not in JOB_TYPES:
            raise AppError(
                f"Unknown job type {job_type!r}. Choose from: "
                f"{', '.join(JOB_TYPES)}.",
                code="invalid_job_type",
            )

        engine = self._app.resolve_engine(payload.get("engine"))

        voice: str | None = None
        speed: float | None = None

        if engine == ENGINE_KOKORO:
            voice = payload.get("voice") or (
                list(KOKORO_VOICES)[0]
            )

            if voice not in KOKORO_VOICES:
                raise AppError(
                    f"Unknown Kokoro voice {voice!r}.",
                    code="invalid_voice",
                )

            try:
                speed = validate_speed(
                    payload.get("speed")
                    if payload.get("speed") is not None
                    else 1.0
                )
            except ValueError as error:
                raise AppError(str(error), code="invalid_speed") from error
        elif payload.get("speed") is not None:
            raise AppError(
                "Speech speed is only supported by the Kokoro engine.",
                code="speed_not_supported",
            )

        items = payload.get("items")
        expected = 1 if job_type == "long_text" else None

        if not isinstance(items, list) or not items:
            raise AppError(
                "Provide at least one item.",
                code="invalid_job",
            )

        if len(items) > self.max_items:
            raise AppError(
                f"Too many items: {len(items)}. "
                f"The limit is {self.max_items}.",
                code="too_many_items",
            )

        if expected is not None and len(items) != expected:
            raise AppError(
                "A long_text job takes exactly one item.",
                code="invalid_job",
            )

        job_items: list[JobItem] = []

        for index, entry in enumerate(items):
            if not isinstance(entry, dict):
                raise AppError(
                    f"Item {index + 1} is not a valid object.",
                    code="invalid_job",
                )

            text = entry.get("text")

            if not isinstance(text, str) or not text.strip():
                raise AppError(
                    f"Item {index + 1} has no text.",
                    code="empty_text",
                )

            text = text.strip()

            if len(text) > MAX_LONG_TEXT_CHARS:
                raise AppError(
                    f"Item {index + 1} is too long ({len(text)} characters). "
                    f"The limit is {MAX_LONG_TEXT_CHARS}.",
                    code="text_too_long",
                )

            raw_name = entry.get("name")
            name = (
                sanitize_filename(raw_name, fallback=f"item_{index + 1}")
                .removesuffix(".txt")
                .strip()
                or f"Item {index + 1}"
            )

            job_items.append(
                JobItem(
                    id=uuid.uuid4().hex[:8],
                    name=name[:80],
                    text=text,
                )
            )

        return Job(
            id=uuid.uuid4().hex[:12],
            type=job_type,
            engine=engine,
            voice=voice,
            speed=speed,
            items=job_items,
        )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                job = self._queue.get()
            except Exception:  # pragma: no cover - defensive
                continue

            if job is None:
                return

            try:
                self._process(job)
            except Exception:  # noqa: BLE001 - worker must survive
                print("[jobs] unexpected worker failure:", flush=True)
                traceback.print_exc()

                with self._lock:
                    job.status = STATUS_FAILED
                    job.error = "The job failed unexpectedly."
                    job.finished_at = _now()

    def _update(self, job: Job) -> None:
        """Reserved for future persistence; keeps state changes explicit."""

        with self._lock:
            job.created_at = job.created_at

    def _process(self, job: Job) -> None:
        with self._lock:
            job.status = STATUS_RUNNING

        successes = 0

        for item in job.items:
            with self._lock:
                item.status = STATUS_RUNNING

            chunk_paths: list[str] = []

            try:
                chunks = split_text(item.text, CHUNK_LIMITS[job.engine])

                if not chunks:
                    raise AppError("The item contains no usable text.")

                with self._lock:
                    item.chunks_total = len(chunks)

                for chunk in chunks:
                    result = self._app.synthesize(
                        chunk,
                        engine=job.engine,
                        voice=job.voice,
                        speed=job.speed,
                    )

                    chunk_paths.append(result.path)

                    with self._lock:
                        item.chunks_completed += 1

                target_dir = self.job_root / job.id
                target_dir.mkdir(parents=True, exist_ok=True)

                target = target_dir / f"{item.id}.wav"

                metadata = join_wav_files(
                    chunk_paths,
                    target,
                    gap_seconds=self.gap_seconds,
                )

                with self._lock:
                    item.filename = metadata["filename"]
                    item.status = STATUS_COMPLETED
                    item.error = None

                successes += 1

            except (AppError, AudioJoinError, ValueError) as error:
                message = getattr(error, "message", None) or str(error)

                with self._lock:
                    item.status = STATUS_FAILED
                    item.error = message

            except Exception:  # noqa: BLE001 - isolate one bad item
                print(f"[jobs] item {item.id} failed:", flush=True)
                traceback.print_exc()

                with self._lock:
                    item.status = STATUS_FAILED
                    item.error = "Generation failed unexpectedly."

            finally:
                for path in chunk_paths:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError:
                        pass

        with self._lock:
            job.status = STATUS_COMPLETED if successes else STATUS_FAILED
            job.finished_at = _now()

            if successes == 0:
                job.error = "All items failed."
            elif job.failed_count:
                job.error = f"{job.failed_count} of {len(job.items)} items failed."

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------
    def item_file(self, job_id: str, item_id: str) -> Path:
        """
        Return the WAV path for one completed item.

        The filename comes from the job record, never from the request, so a
        client cannot address files outside the job folder.
        """

        job = self.get(job_id)

        for item in job.items:
            if item.id == item_id and item.filename:
                path = self.job_root / job.id / item.filename

                if path.is_file():
                    return path

        raise FileNotFoundError(item_id)

    def archive(self, job_id: str) -> tuple[bytes, str]:
        """
        Build a ZIP with every completed WAV plus a manifest.

        Returns ``(bytes, download_name)``. Only files recorded on the job
        are included; client input never becomes a path.
        """

        job = self.get(job_id)

        if not job.has_results:
            raise ArchiveUnavailable(
                "This job has no completed audio to download."
            )

        buffer = io.BytesIO()
        used_names: set[str] = set()
        manifest_items: list[dict] = []

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, item in enumerate(job.items, start=1):
                if item.status != STATUS_COMPLETED or not item.filename:
                    manifest_items.append(
                        {
                            "name": item.name,
                            "status": item.status,
                            "filename": None,
                            "error": item.error,
                            "characters": len(item.text),
                        }
                    )
                    continue

                source = self.job_root / job.id / item.filename

                if not source.is_file():
                    manifest_items.append(
                        {
                            "name": item.name,
                            "status": STATUS_FAILED,
                            "filename": None,
                            "error": "Generated file is missing.",
                            "characters": len(item.text),
                        }
                    )
                    continue

                stem = sanitize_filename(item.name, fallback=f"item_{index}")
                arcname = f"{index:02d}_{stem.removesuffix('.txt')}.wav"

                while arcname in used_names:
                    arcname = f"{index:02d}_{stem.removesuffix('.txt')}_{len(used_names)}.wav"

                used_names.add(arcname)
                archive.write(source, arcname=arcname)

                manifest_items.append(
                    {
                        "name": item.name,
                        "status": item.status,
                        "filename": arcname,
                        "error": item.error,
                        "characters": len(item.text),
                    }
                )

            manifest = {
                "job_id": job.id,
                "type": job.type,
                "engine": job.engine,
                "voice": job.voice,
                "speed": job.speed,
                "created_at": job.created_at,
                "completed": job.completed_count,
                "failed": job.failed_count,
                "items": manifest_items,
            }

            archive.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2),
            )

        return buffer.getvalue(), f"voiceforge-{job.id}.zip"
