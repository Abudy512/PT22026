"""Optional real-Kokoro API tests.

Gated behind ``VOICEFORGE_TEST_KOKORO=1`` so the default suite never loads
the 82M model or touches the Hugging Face cache.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import zipfile

import pytest

from conftest import error_payload, http_get, http_post_json, wait_for_job

RUN_REAL = os.environ.get("VOICEFORGE_TEST_KOKORO") == "1"

if RUN_REAL:
    # These tests assume the Kokoro weights are already cached. Offline mode
    # stops huggingface_hub from re-validating metadata over the network,
    # which can take minutes on a machine without access to huggingface.co.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

pytestmark = pytest.mark.skipif(
    not RUN_REAL,
    reason="set VOICEFORGE_TEST_KOKORO=1 to run real Kokoro tests",
)


def test_kokoro_generation_and_download(server):
    with http_post_json(
        server + "/generate",
        {"text": "Hello, this is Kokoro.", "engine": "kokoro", "voice": "af_heart"},
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["engine"] == "kokoro"
    assert payload["sample_rate"] == 24000

    with http_get(server + payload["audio_url"]) as response:
        audio = response.read()

    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_kokoro_accepts_digits(server):
    with http_post_json(
        server + "/generate",
        {"text": "Room 204 at 9:30.", "engine": "kokoro", "voice": "am_michael"},
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["engine"] == "kokoro"


def test_kokoro_rejects_unknown_voice(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(
            server + "/generate",
            {"text": "Hello.", "engine": "kokoro", "voice": "not_a_voice"},
        )

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert "voice" in payload["error"].lower()


def _generate(server, text, speed):
    with http_post_json(
        server + "/api/generate",
        {"text": text, "engine": "kokoro", "voice": "af_heart", "speed": speed},
    ) as response:
        return json.loads(response.read())


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_kokoro_speed_values_generate_valid_audio(server, speed):
    result = _generate(server, "Hello from VoiceForge.", speed)

    assert result["engine"] == "kokoro"
    assert result["speed"] == speed
    assert result["duration_seconds"] > 0

    with http_get(server + result["audio_url"]) as response:
        audio = response.read()

    assert response.status == 200
    assert audio[:4] == b"RIFF"


def test_kokoro_speed_actually_changes_duration(server):
    """Real model check: 0.5x is longer than 1.0x, which is longer than 2.0x."""

    text = "The weather is pleasant today."

    slow = _generate(server, text, 0.5)
    normal = _generate(server, text, 1.0)
    fast = _generate(server, text, 2.0)

    assert slow["speed"] == 0.5
    assert normal["speed"] == 1.0
    assert fast["speed"] == 2.0

    assert slow["duration_seconds"] > normal["duration_seconds"] * 1.5
    assert fast["duration_seconds"] < normal["duration_seconds"] * 0.75

    # The audio itself must be valid at every speed.
    for result in (slow, normal, fast):
        with http_get(server + result["audio_url"]) as response:
            audio = response.read()
        assert audio[:4] == b"RIFF"
        assert len(audio) > 1000


def test_kokoro_long_text_job_and_zip(server):
    """Real model: multi-chunk long text, joined WAV and ZIP contents."""

    text = " ".join(
        ["The quick brown fox jumps over the lazy dog."] * 12
    )

    payload = {
        "type": "long_text",
        "engine": "kokoro",
        "voice": "af_heart",
        "speed": 1.0,
        "items": [{"name": "Long text", "text": text}],
    }

    with http_post_json(server + "/api/jobs", payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id, timeout=300)

    assert job["status"] == "completed"
    item = job["items"][0]
    assert item["status"] == "completed"
    assert item["chunks_total"] >= 2
    assert item["chunks_completed"] == item["chunks_total"]

    with http_get(server + item["download_url"]) as response:
        audio = response.read()

    assert audio[:4] == b"RIFF"
    assert len(audio) > 5000

    with http_get(f"{server}/api/jobs/{job_id}/download") as response:
        archive = response.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read("manifest.json"))

    assert "manifest.json" in names
    assert sum(name.endswith(".wav") for name in names) == 1
    assert manifest["completed"] == 1


def test_kokoro_batch_job_with_two_entries(server):
    payload = {
        "type": "batch",
        "engine": "kokoro",
        "voice": "am_michael",
        "speed": 1.25,
        "items": [
            {"name": "First", "text": "Hello from the first batch entry."},
            {"name": "Second", "text": "And this is the second entry."},
        ],
    }

    with http_post_json(server + "/api/jobs", payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id, timeout=300)

    assert job["status"] == "completed"
    assert job["completed"] == 2
    assert job["speed"] == 1.25

    for item in job["items"]:
        with http_get(server + item["download_url"]) as response:
            assert response.read()[:4] == b"RIFF"


def test_kokoro_studio_generation_is_saved_to_the_library(server):
    with http_post_json(
        server + "/api/generate",
        {"text": "Saved in the library.", "engine": "kokoro", "speed": 1.25},
    ) as response:
        result = json.loads(response.read())

    with http_get(server + "/api/library?source=studio") as response:
        payload = json.loads(response.read())

    assert payload["pagination"]["total"] == 1
    record = payload["records"][0]
    assert record["engine"] == "kokoro"
    assert record["speed"] == 1.25
    assert record["original_text"] == result["text"]
    assert record["file_size"] > 0

    with http_get(f"{server}/api/library/{record['id']}/audio") as response:
        assert response.read()[:4] == b"RIFF"


def test_kokoro_long_text_job_creates_one_library_record(server):
    text = " ".join(["A sentence for the library test."] * 14)

    with http_post_json(
        server + "/api/jobs",
        {
            "type": "long_text",
            "engine": "kokoro",
            "voice": "af_heart",
            "items": [{"name": "Library long text", "text": text}],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id, timeout=300)

    with http_get(server + "/api/library?source=long_text") as response:
        payload = json.loads(response.read())

    assert payload["pagination"]["total"] == 1
    assert payload["records"][0]["job_id"] == job_id
    assert payload["records"][0]["name"] == "Library long text"


def test_kokoro_batch_records_each_item_in_the_library(server):
    with http_post_json(
        server + "/api/jobs",
        {
            "type": "batch",
            "engine": "kokoro",
            "voice": "am_michael",
            "items": [
                {"name": "Batch one", "text": "First library batch entry."},
                {"name": "Batch two", "text": "Second library batch entry."},
            ],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id, timeout=300)

    with http_get(server + "/api/library?source=batch") as response:
        payload = json.loads(response.read())

    assert payload["pagination"]["total"] == 2
    assert {record["name"] for record in payload["records"]} == {
        "Batch one",
        "Batch two",
    }
