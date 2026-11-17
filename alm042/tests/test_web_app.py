
"""End-to-end tests for the web interface.

These tests start a real HTTP server on an ephemeral port and talk to it with
urllib, so they exercise routing, JSON handling, audio serving and error
reporting exactly as a browser would.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from voiceforge.app_core import VoiceForgeApp
from voiceforge.web_app import create_server


@pytest.fixture
def server(tmp_path, tiny_checkpoint):
    app = VoiceForgeApp(
        checkpoint_path=tiny_checkpoint,
        device="cpu",
        output_dir=tmp_path / "app_outputs",
        griffin_lim_iterations=2,
        max_mel_frames=32,
        engine="voiceforge",
    )

    httpd = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        yield base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url: str):
    return urllib.request.urlopen(url, timeout=10)


def post_json(url: str, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=30)


def test_index_page_is_served(server):
    with get(server + "/") as response:
        body = response.read().decode("utf-8")

    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/html")
    assert "VoiceForge" in body
    assert "Generate Speech" in body
    assert "Download WAV" in body
    # Engine selection and the pretrained-model attribution must be present.
    assert "Speech engine" in body
    assert "Kokoro" in body
    assert "Apache-2.0" in body


def test_health_endpoint(server):
    with get(server + "/health") as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["model_loaded"] is False
    assert "engines" in payload
    assert payload["engines"]["voiceforge"]["checkpoint"].endswith(".pt")


def test_generate_returns_audio_url(server):
    with post_json(server + "/generate", {"text": "Hello world."}) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["audio_url"].startswith("/audio/")
    assert payload["audio_url"].endswith(".wav")
    assert payload["duration_seconds"] > 0
    assert payload["engine"] == "voiceforge"
    assert payload["sample_rate"] == 22050


def test_generated_audio_can_be_downloaded(server):
    with post_json(server + "/generate", {"text": "Hello world."}) as response:
        payload = json.loads(response.read())

    with get(server + payload["audio_url"]) as response:
        audio = response.read()

    assert response.status == 200
    assert response.headers["Content-Type"] == "audio/wav"
    assert len(audio) > 0
    assert audio[:4] == b"RIFF"


def test_empty_text_returns_400(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(server + "/generate", {"text": "   "})

    payload = json.loads(error.value.read())

    assert error.value.code == 400
    assert payload["ok"] is False
    assert "enter some English text" in payload["error"]


def test_unsupported_characters_return_400(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(server + "/generate", {"text": "Hello 123"})

    payload = json.loads(error.value.read())

    assert error.value.code == 400
    assert "Unsupported characters" in payload["error"]


def test_invalid_json_returns_400(server):
    request = urllib.request.Request(
        server + "/generate",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)

    assert error.value.code == 400


def test_missing_page_returns_404(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        get(server + "/does-not-exist")

    assert error.value.code == 404


def test_audio_path_traversal_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        get(server + "/audio/../app_core.py")

    assert error.value.code == 404


def test_unknown_engine_returns_400(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(
            server + "/generate",
            {"text": "Hello.", "engine": "does-not-exist"},
        )

    payload = json.loads(error.value.read())

    assert error.value.code == 400
    assert "Unknown speech engine" in payload["error"]


def test_engine_voiceforge_is_accepted_explicitly(server):
    with post_json(
        server + "/generate",
        {"text": "Hello world.", "engine": "voiceforge"},
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["engine"] == "voiceforge"
