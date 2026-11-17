"""Integration test for the training script.

This test needs the LJSpeech dataset and is skipped automatically when the
dataset is not present, so the rest of the suite still runs on a clean
checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train import DEFAULT_METADATA, main
from voiceforge.checkpoint import load_model

DATASET_AVAILABLE = Path(DEFAULT_METADATA).is_file()

pytestmark = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="LJSpeech dataset not present",
)


def test_training_script_runs_end_to_end(tmp_path):
    checkpoint = tmp_path / "model.pt"
    history = tmp_path / "history.json"

    exit_code = main(
        [
            "--max-samples",
            "16",
            "--val-samples",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--max-seconds",
            "5",
            "--hidden-dim",
            "32",
            "--embedding-dim",
            "16",
            "--attention-dim",
            "16",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--log-every",
            "0",
        ]
    )

    assert exit_code == 0
    assert checkpoint.is_file()

    entries = json.loads(history.read_text(encoding="utf-8"))

    assert len(entries) == 1
    assert entries[0]["epoch"] == 1
    assert entries[0]["val_loss"] is not None

    # The saved checkpoint must be loadable by the inference pipeline with
    # the same architecture and configuration.
    model, config, metadata = load_model(checkpoint)

    assert config["architecture"] == "tacotron_lite"
    assert config["hidden_dim"] == 32
    assert metadata["epoch"] == 1
    assert sum(p.numel() for p in model.parameters()) == 92_690
