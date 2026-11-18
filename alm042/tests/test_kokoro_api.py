"""Optional real-Kokoro API tests.

Gated behind ``VOICEFORGE_TEST_KOKORO=1`` so the default suite never loads
the 82M model or touches the Hugging Face cache.
"""

from __future__ import annotations

import json
import os
import urllib.error

import pytest

from conftest import error_payload, http_get, http_post_json

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
