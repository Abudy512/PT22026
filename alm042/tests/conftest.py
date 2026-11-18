"""Shared fixtures for the VoiceForge test suite."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voiceforge.app_core import VoiceForgeApp  # noqa: E402
from voiceforge.checkpoint import get_model_config  # noqa: E402
from voiceforge.model import AcousticModel  # noqa: E402
from voiceforge.text import CHAR_TO_ID  # noqa: E402
from voiceforge.web_app import create_server  # noqa: E402


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A very small custom-model checkpoint so tests stay fast."""

    model = AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        n_mels=80,
        embedding_dim=8,
        hidden_dim=8,
        attention_dim=8,
    )

    # Predict roughly 20 Mel frames instead of a random tiny length.
    with torch.no_grad():
        model.duration_projection[2].bias.fill_(3.0)

    path = tmp_path / "tiny.pt"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "loss": 1.0,
            "model_config": get_model_config(model),
        },
        path,
    )

    return path


@pytest.fixture
def app(tmp_path, tiny_checkpoint):
    return VoiceForgeApp(
        checkpoint_path=tiny_checkpoint,
        device="cpu",
        output_dir=tmp_path / "app_outputs",
        griffin_lim_iterations=2,
        max_mel_frames=32,
        engine="voiceforge",
    )


@pytest.fixture
def server(app):
    httpd = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        yield url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def http_get(url: str):
    """GET without letting urllib normalise the path."""

    request = urllib.request.Request(url, method="GET")
    return urllib.request.urlopen(request, timeout=15)


def http_post_json(url: str, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=240)


def error_payload(error: urllib.error.HTTPError) -> dict:
    return json.loads(error.read())
