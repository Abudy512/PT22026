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
