
"""Tests for checkpoint loading and architecture dispatch."""

from __future__ import annotations

import pytest
import torch

from voiceforge.architectures import ARCH_LEGACY, ARCH_TACOTRON
from voiceforge.checkpoint import (
    build_model,
    get_model_config,
    infer_model_config,
    load_checkpoint,
    load_model,
)
from voiceforge.model import AcousticModel, ModelConfig
from voiceforge.model_legacy import LegacyAcousticModel
from voiceforge.text import CHAR_TO_ID


def test_config_round_trip():
    config = ModelConfig(
        vocab_size=40,
        n_mels=80,
        hidden_dim=192,
    )

    restored = ModelConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.architecture == ARCH_TACOTRON


def test_config_ignores_unknown_keys():
    config = ModelConfig.from_dict(
        {"vocab_size": 39, "hidden_dim": 128, "unknown": "ignored"}
    )

    assert config.vocab_size == 39
    assert config.hidden_dim == 128


def test_get_model_config_for_attention_model(tiny_model):
    config = get_model_config(tiny_model)

    assert config["architecture"] == ARCH_TACOTRON
    assert config["vocab_size"] == len(CHAR_TO_ID)
    assert config["hidden_dim"] == 8


def test_get_model_config_for_legacy_model():
    model = LegacyAcousticModel(vocab_size=38, n_mels=80, embedding_dim=16, hidden_dim=32)

    config = get_model_config(model)

    assert config["architecture"] == ARCH_LEGACY
    assert config["embedding_dim"] == 16
    assert config["hidden_dim"] == 32


def test_infer_config_from_legacy_state_dict():
    model = LegacyAcousticModel(vocab_size=38, n_mels=80, embedding_dim=16, hidden_dim=32)

    config = infer_model_config(model.state_dict())

    assert config["architecture"] == ARCH_LEGACY
    assert config["n_mels"] == 80


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown model architecture"):
        build_model({"architecture": "does_not_exist"})


def test_build_model_creates_attention_model():
    config = ModelConfig(vocab_size=38, hidden_dim=32, attention_dim=16)

    model = build_model(config.to_dict())

    assert isinstance(model, AcousticModel)
    assert model.hidden_dim == 32


def test_load_attention_checkpoint(tiny_checkpoint):
    model, config, metadata = load_model(tiny_checkpoint)

    assert isinstance(model, AcousticModel)
    assert config["architecture"] == ARCH_TACOTRON
    assert metadata["epoch"] == 1
    assert model.training is False


def test_load_legacy_checkpoint(tiny_legacy_checkpoint):
    model, config, metadata = load_model(tiny_legacy_checkpoint)

    assert isinstance(model, LegacyAcousticModel)
    assert config["architecture"] == ARCH_LEGACY
    assert metadata["epoch"] == 1


def test_load_checkpoint_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_rejects_non_dictionary(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], path)

    with pytest.raises(ValueError, match="must be a dictionary"):
        load_checkpoint(path)


def test_load_checkpoint_requires_model_state_dict(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({"epoch": 1}, path)

    with pytest.raises(ValueError, match="valid 'model_state_dict'"):
        load_checkpoint(path)


def test_attention_checkpoint_without_config_is_rejected(tmp_path, tiny_model):
    path = tmp_path / "no_config.pt"

    torch.save(
        {"model_state_dict": tiny_model.state_dict(), "epoch": 1},
        path,
    )

    # Without model_config the loader assumes the legacy architecture, and
    # the weights do not match it, so a clear error must be raised.
    with pytest.raises(ValueError):
        load_model(path)


def test_weights_that_do_not_match_architecture_raise(tmp_path):
    legacy = LegacyAcousticModel(vocab_size=38, n_mels=80, embedding_dim=16, hidden_dim=32)

    path = tmp_path / "mismatch.pt"

    torch.save(
        {
            "model_state_dict": legacy.state_dict(),
            "model_config": ModelConfig(hidden_dim=8).to_dict(),
            "epoch": 1,
        },
        path,
    )

    with pytest.raises(ValueError, match="do not match|does not match"):
        load_model(path)


def test_round_trip_preserves_weights(tmp_path):
    """A model built from its own config must reload exactly."""

    config = ModelConfig(vocab_size=38, hidden_dim=16, attention_dim=8).to_dict()

    original = build_model(config)

    path = tmp_path / "round_trip.pt"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "model_config": config,
            "epoch": 1,
        },
        path,
    )

    restored, restored_config, _ = load_model(path)

    assert restored_config["hidden_dim"] == 16

    original_state = original.state_dict()
    restored_state = restored.state_dict()

    assert original_state.keys() == restored_state.keys()

    for key in original_state:
        assert torch.equal(original_state[key], restored_state[key]), key


def test_checkpoint_without_duration_head_reports_clear_error(tmp_path):
    """Checkpoints from before the duration predictor must fail clearly."""

    legacy = LegacyAcousticModel(
        vocab_size=38,
        n_mels=80,
        embedding_dim=16,
        hidden_dim=32,
    )

    state = legacy.state_dict()
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith("duration_projection")
    }

    path = tmp_path / "old_smoke.pt"

    torch.save(
        {
            "model_state_dict": state,
            "model_config": {
                "architecture": ARCH_LEGACY,
                "vocab_size": 38,
                "n_mels": 80,
                "embedding_dim": 16,
                "hidden_dim": 32,
            },
            "epoch": 1,
        },
        path,
    )

    with pytest.raises(ValueError, match="before the duration predictor"):
        load_model(path)

