"""
Persistent audio library backed by SQLite.

One table stores every generated recording. Synthesis stays in
``app_core``; this module only copies the finished WAV into the library
folder and records its metadata, so every workflow (studio, long text,
batch) uses the same write path.

The database lives under ``outputs/`` and is git-ignored.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from voiceforge.app_core import AppError

#: Current schema version (``PRAGMA user_version``).
SCHEMA_VERSION = 1

DEFAULT_DB_PATH = "outputs/library.db"
DEFAULT_AUDIO_DIR = "outputs/library"

MAX_NAME_LENGTH = 80
MAX_SEARCH_LENGTH = 200
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

SOURCE_TYPES = ("studio", "long_text", "batch")

#: Whitelisted sort keys -> ORDER BY clause. Never interpolate user input.
SORT_OPTIONS = {
    "newest": "created_at DESC, id DESC",
    "oldest": "created_at ASC, id ASC",
    "name": "name COLLATE NOCASE ASC, created_at DESC",
    "duration": "duration_seconds DESC, created_at DESC",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_records (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    original_text TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    engine TEXT NOT NULL,
    voice TEXT,
    speed REAL,
    sample_rate INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    file_size INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    job_id TEXT,
    job_item_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_records_source
    ON audio_records(source_filename);

CREATE UNIQUE INDEX IF NOT EXISTS idx_records_job_item
    ON audio_records(job_id, job_item_id)
    WHERE job_item_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_records_created
    ON audio_records(created_at);

CREATE INDEX IF NOT EXISTS idx_records_name
    ON audio_records(name COLLATE NOCASE);

CREATE INDEX IF NOT EXISTS idx_records_engine
    ON audio_records(engine);

CREATE INDEX IF NOT EXISTS idx_records_source_type
    ON audio_records(source_type);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _escape_like(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def validate_name(name: object) -> str:
    """Validate a display name (the WAV filename is never changed)."""

    if not isinstance(name, str):
        raise AppError("The name must be text.", code="invalid_name")

    cleaned = name.strip()

    if not cleaned:
        raise AppError("The name cannot be empty.", code="invalid_name")

    if len(cleaned) > MAX_NAME_LENGTH:
        raise AppError(
            f"The name is too long ({len(cleaned)} characters). "
            f"The limit is {MAX_NAME_LENGTH}.",
            code="invalid_name",
        )

    if any(ord(character) < 32 for character in cleaned):
        raise AppError(
            "The name contains control characters.",
            code="invalid_name",
        )

    return cleaned


def _default_name(text: str) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return "Untitled audio"
    if len(collapsed) <= 60:
        return collapsed
    return collapsed[:57].rstrip() + "\u2026"


class Library:
    """SQLite-backed audio library and its audio folder."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        audio_dir: str | Path = DEFAULT_AUDIO_DIR,
    ):
        self.db_path = Path(db_path)
        self.audio_dir = Path(audio_dir)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._initialize()

    # ------------------------------------------------------------------
    # Database plumbing
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        """Create the schema when missing; never drop existing data."""

        with self._connect() as connection:
            connection.executescript(_SCHEMA)

            version = connection.execute("PRAGMA user_version").fetchone()[0]

            if version < SCHEMA_VERSION:
                # Future migrations belong here, guarded by the version.
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _find_existing(
        self,
        connection: sqlite3.Connection,
        source_filename: str,
        job_id: str | None,
        job_item_id: str | None,
    ) -> dict | None:
        row = connection.execute(
            "SELECT * FROM audio_records WHERE source_filename = ?",
            (source_filename,),
        ).fetchone()

        if row is not None:
            return self._row_to_dict(row)

        if job_id and job_item_id:
            row = connection.execute(
                "SELECT * FROM audio_records "
                "WHERE job_id = ? AND job_item_id = ?",
                (job_id, job_item_id),
            ).fetchone()

            if row is not None:
                return self._row_to_dict(row)

        return None

    def record(
        self,
        *,
        text: str,
        source_filename: str,
        source_path: str | Path,
        engine: str,
        sample_rate: int,
        duration_seconds: float,
        source_type: str,
        voice: str | None = None,
        speed: float | None = None,
        job_id: str | None = None,
        job_item_id: str | None = None,
        name: str | None = None,
    ) -> dict:
        """
        Copy a finished WAV into the library and store its metadata.

        Idempotent: the same generated file (or the same job item) is never
        recorded twice, so repeated polling cannot duplicate records.
        """

        if source_type not in SOURCE_TYPES:
            raise ValueError(f"Unknown source type {source_type!r}")

        source = Path(source_path)

        if not source.is_file():
            raise AppError(
                "The generated audio file is missing.",
                code="audio_missing",
                status=500,
            )

        display_name = (
            validate_name(name) if name else _default_name(text)
        )

        with self._lock:
            connection = self._connect()

            try:
                existing = self._find_existing(
                    connection,
                    source_filename,
                    job_id,
                    job_item_id,
                )

                if existing is not None:
                    return existing

                record_id = uuid.uuid4().hex
                filename = f"{record_id}.wav"
                destination = self.audio_dir / filename

                shutil.copyfile(source, destination)

                timestamp = _now()
                size = destination.stat().st_size

                try:
                    connection.execute(
                        """
                        INSERT INTO audio_records (
                            id, name, original_text, filename, source_filename,
                            engine, voice, speed, sample_rate,
                            duration_seconds, file_size, source_type,
                            job_id, job_item_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record_id,
                            display_name,
                            text,
                            filename,
                            source_filename,
                            engine,
                            voice,
                            speed,
                            int(sample_rate),
                            float(duration_seconds),
                            int(size),
                            source_type,
                            job_id,
                            job_item_id,
                            timestamp,
                            timestamp,
                        ),
                    )
                    connection.commit()
                except sqlite3.IntegrityError:
                    destination.unlink(missing_ok=True)

                    existing = self._find_existing(
                        connection,
                        source_filename,
                        job_id,
                        job_item_id,
                    )

                    if existing is not None:
                        return existing

                    raise

            finally:
                connection.close()

        return self.get(record_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "original_text": row["original_text"],
            "filename": row["filename"],
            "engine": row["engine"],
            "voice": row["voice"],
            "speed": row["speed"],
            "sample_rate": row["sample_rate"],
            "duration_seconds": row["duration_seconds"],
            "file_size": row["file_size"],
            "source_type": row["source_type"],
            "job_id": row["job_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get(self, record_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM audio_records WHERE id = ?",
                (str(record_id),),
            ).fetchone()

        if row is None:
            raise AppError(
                "Audio record not found.",
                code="record_not_found",
                status=404,
            )

        return self._row_to_dict(row)

    def list_records(
        self,
        query: str | None = None,
        sort: str = "newest",
        engine: str | None = None,
        source: str | None = None,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[list[dict], int]:
        """Search, filter, sort and paginate records."""

        if sort not in SORT_OPTIONS:
            raise AppError(
                f"Unknown sort option {sort!r}. Choose from: "
                f"{', '.join(SORT_OPTIONS)}.",
                code="invalid_sort",
            )

        page = max(1, int(page))
        limit = min(max(1, int(limit)), MAX_PAGE_SIZE)

        clauses: list[str] = []
        params: list[object] = []

        if query:
            if len(query) > MAX_SEARCH_LENGTH:
                raise AppError(
                    f"The search text is too long ({len(query)} characters). "
                    f"The limit is {MAX_SEARCH_LENGTH}.",
                    code="invalid_search",
                )

            pattern = f"%{_escape_like(query)}%"
            clauses.append(
                "(name LIKE ? ESCAPE '\\' OR original_text LIKE ? ESCAPE '\\')"
            )
            params.extend([pattern, pattern])

        if engine:
            clauses.append("engine = ?")
            params.append(engine)

        if source:
            if source not in SOURCE_TYPES:
                raise AppError(
                    f"Unknown source type {source!r}.",
                    code="invalid_source",
                )

            clauses.append("source_type = ?")
            params.append(source)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = SORT_OPTIONS[sort]
        offset = (page - 1) * limit

        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM audio_records{where}",
                params,
            ).fetchone()[0]

            rows = connection.execute(
                f"SELECT * FROM audio_records{where} "
                f"ORDER BY {order} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()

        return [self._row_to_dict(row) for row in rows], int(total)

    # ------------------------------------------------------------------
    # Update and delete
    # ------------------------------------------------------------------
    def rename(self, record_id: str, name: object) -> dict:
        cleaned = validate_name(name)

        with self._lock:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE audio_records SET name = ?, updated_at = ? "
                    "WHERE id = ?",
                    (cleaned, _now(), str(record_id)),
                )

                if cursor.rowcount == 0:
                    raise AppError(
                        "Audio record not found.",
                        code="record_not_found",
                        status=404,
                    )

        return self.get(record_id)

    def _safe_audio_path(self, filename: str) -> Path:
        """Resolve a stored filename inside the library folder only."""

        candidate_name = Path(str(filename))

        if candidate_name.name != str(filename) or "/" in str(filename):
            raise AppError(
                "Invalid audio file name.",
                code="invalid_filename",
            )

        root = self.audio_dir.resolve()
        path = (root / candidate_name.name).resolve()

        if root not in path.parents:
            raise AppError(
                "Invalid audio file name.",
                code="invalid_filename",
            )

        return path

    def audio_path(self, record_id: str) -> tuple[dict, Path]:
        record = self.get(record_id)
        path = self._safe_audio_path(record["filename"])

        if not path.is_file():
            raise AppError(
                "The audio file for this record is missing.",
                code="audio_not_found",
                status=404,
            )

        return record, path

    def delete(self, record_id: str) -> dict:
        """
        Delete a record and its WAV.

        The row is deleted inside a transaction; if the file cannot be
        removed the transaction is rolled back so the library never keeps a
        record whose audio vanished, and a missing file is tolerated.
        """

        record = self.get(record_id)
        path = self._safe_audio_path(record["filename"])

        with self._lock:
            connection = self._connect()

            try:
                connection.execute(
                    "DELETE FROM audio_records WHERE id = ?",
                    (str(record_id),),
                )

                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as error:
                    connection.rollback()
                    raise AppError(
                        "The audio file could not be deleted.",
                        code="delete_failed",
                        status=500,
                    ) from error

                connection.commit()
            finally:
                connection.close()

        return {"id": record["id"], "filename": record["filename"]}
