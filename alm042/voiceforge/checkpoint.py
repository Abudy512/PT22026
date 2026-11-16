"""
Checkpoint helpers for VoiceForge.

A checkpoint is a dictionary produced by the training script.  The most
important entries are:

    model_state_dict  the neural network weights
    model_config      the hyper-parameters needed to rebuild the network
    epoch / loss      training metadata (optional)

Older checkpoints created before ``model_config`` was introduced are still
supported: the configuration is inferred from the shape of the weight
tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from voiceforge.model import AcousticModel

# Architecture identifiers stored inside checkpoints.
ARCH_LEGACY = "legacy_interpolate"
ARCH_TACOTRON = "tacotron_lite"


def infer_model_config(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """
    Rebuild a model configuration from the weight shapes.

    This allows old checkpoints (which did not store a configuration) to be
    loaded without knowing how they were created.
    """

    if "embedding.weight" not in state_dict:
        raise ValueError(
            "Checkpoint does not contain an 'embedding.weight' tensor "
            "and is not a recognised VoiceForge model."
        )

    vocab_size, embedding_dim = state_dict["embedding.weight"].shape

    if "mel_projection.weight" in state_dict:
        n_mels, doubled_hidden = state_dict["mel_projection.weight"].shape
        hidden_dim = doubled_hidden // 2
    else:
        n_mels, hidden_dim = 80, 128

    return {
        "architecture": ARCH_LEGACY,
        "vocab_size": int(vocab_size),
        "n_mels": int(n_mels),
        "embedding_dim": int(embedding_dim),
        "hidden_dim": int(hidden_dim),
    }


def build_model(config: Mapping[str, Any]) -> AcousticModel:
    """
    Create an untrained model from a configuration dictionary.
    """

    architecture = config.get("architecture", ARCH_LEGACY)

    if architecture != ARCH_LEGACY:
        raise ValueError(
            f"Unknown model architecture: {architecture!r}. "
            f"Supported: {ARCH_LEGACY!r}."
        )

    return AcousticModel(
        vocab_size=int(config["vocab_size"]),
        n_mels=int(config["n_mels"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load and validate a VoiceForge checkpoint file.

    Returns a dictionary with ``state_dict``, ``config`` and ``metadata``.
    """

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    raw = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "Checkpoint must be a dictionary containing at least "
            "'model_state_dict'."
        )

    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        config = raw.get("model_config")
    else:
        # Some checkpoints store the bare state dict.
        state_dict = raw
        config = None

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "Checkpoint does not contain a valid 'model_state_dict'."
        )

    if config is None:
        config = infer_model_config(state_dict)

    metadata = {
        "epoch": raw.get("epoch"),
        "loss": raw.get("loss"),
        "history": raw.get("history"),
    }

    return {
        "state_dict": state_dict,
        "config": dict(config),
        "metadata": metadata,
    }


def load_model(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[AcousticModel, dict[str, Any], dict[str, Any]]:
    """
    Load a trained model and its configuration from a checkpoint.

    Returns ``(model, config, metadata)``.  The model is moved to evaluation
    mode so it can be used for inference straight away.
    """

    contents = load_checkpoint(checkpoint_path, map_location=map_location)

    model = build_model(contents["config"])

    try:
        model.load_state_dict(contents["state_dict"])
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint weights do not match the model architecture. "
            "The checkpoint may have been produced by a different version "
            f"of VoiceForge. Original error: {error}"
        ) from error

    model.eval()

    return model, contents["config"], contents["metadata"]
