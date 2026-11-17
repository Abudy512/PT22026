
"""Tests for the objective evaluation harness."""

from __future__ import annotations

import json

import pytest
import torch

from scripts.evaluate import (
    _energy_variation,
    _masked_l1,
    build_parser,
    evaluate_checkpoint,
    evaluate_generated_audio,
    evaluate_model,
    evaluate_vocoder,
    main,
)
from voiceforge.dataset import LJSpeechDataset


def test_masked_l1_ignores_padding():
    prediction = torch.tensor([[[1.0, 2.0, 100.0], [1.0, 2.0, 100.0]]])
    target = torch.zeros(1, 2, 3)
    lengths = torch.tensor([2])

    # Only the first two frames are valid, so the 100s must be ignored.
    assert _masked_l1(prediction, target, lengths) == pytest.approx(1.5)


def test_energy_variation_is_positive_for_varying_signal():
    time = torch.arange(22050, dtype=torch.float32) / 22050

    envelope = torch.linspace(1.0, 0.1, time.numel())
    varying = envelope * torch.sin(2 * torch.pi * 200 * time)

    variation = _energy_variation(varying)

    assert variation is not None
    assert variation > 0.1


def test_energy_variation_is_near_zero_for_constant_signal():
    constant = torch.ones(22050)

    variation = _energy_variation(constant)

    assert variation is not None
    assert variation < 0.01


def test_energy_variation_returns_none_for_tiny_signal():
    assert _energy_variation(torch.ones(100)) is None


def test_evaluate_model_returns_expected_metrics(tiny_model, tiny_dataset):
    metadata_path, _ = tiny_dataset
    dataset = LJSpeechDataset(metadata_path)

    results = evaluate_model(
        tiny_model,
        dataset,
        indices=[0, 1],
        device=torch.device("cpu"),
    )

    assert results["samples"] == 2
    assert results["samples_skipped"] == 0
    assert results["mel_l1"] >= 0
    assert results["duration_mae_frames"] >= 0
    assert 0 <= results["duration_relative_error"]
    assert results["target_mel_std"] > 0
    assert results["spectral_detail_ratio"] > 0


def test_evaluate_model_rejects_model_without_duration_predictor(tiny_dataset):
    metadata_path, _ = tiny_dataset
    dataset = LJSpeechDataset(metadata_path)

    class NoDuration(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="predict_mel_lengths"):
        evaluate_model(
            NoDuration(),
            dataset,
            indices=[0],
            device=torch.device("cpu"),
        )


def test_evaluate_model_rejects_empty_indices(tiny_model, tiny_dataset):
    metadata_path, _ = tiny_dataset
    dataset = LJSpeechDataset(metadata_path)

    with pytest.raises(ValueError, match="No valid samples"):
        evaluate_model(
            tiny_model, dataset, indices=[], device=torch.device("cpu")
        )


def test_evaluate_vocoder_measures_round_trip(tiny_dataset):
    metadata_path, _ = tiny_dataset
    dataset = LJSpeechDataset(metadata_path)

    results = evaluate_vocoder(dataset, indices=[0], n_iter=2)

    assert results["samples"] == 1
    assert results["mel_rmse"] > 0
    assert results["mel_correlation"] > 0.5


def test_evaluate_generated_audio_returns_statistics(tiny_model, tiny_dataset):
    metadata_path, _ = tiny_dataset
    dataset = LJSpeechDataset(metadata_path)

    results = evaluate_generated_audio(
        tiny_model,
        dataset,
        indices=[0, 1],
        device=torch.device("cpu"),
        n_iter=2,
    )

    assert results["samples"] >= 1
    assert results["generated_energy_variation"] is not None
    assert results["real_energy_variation"] is not None
    assert results["average_generated_seconds"] > 0


def test_evaluate_checkpoint_returns_complete_results(
    tiny_checkpoint,
    tiny_dataset,
):
    metadata_path, _ = tiny_dataset

    results = evaluate_checkpoint(
        checkpoint_path=tiny_checkpoint,
        metadata_path=metadata_path,
        samples=2,
        vocoder_samples=1,
        n_iter=2,
        device="cpu",
        progress=False,
    )

    assert results["version"] == 1
    assert results["model"]["samples"] == 2
    assert results["vocoder"]["samples"] == 1
    assert results["config"]["vocab_size"] == 38
    assert results["checkpoint_metadata"]["epoch"] == 1


def test_evaluate_checkpoint_uses_held_out_samples(
    tiny_checkpoint,
    tiny_dataset,
):
    metadata_path, _ = tiny_dataset

    results = evaluate_checkpoint(
        checkpoint_path=tiny_checkpoint,
        metadata_path=metadata_path,
        samples=2,
        vocoder_samples=1,
        n_iter=2,
        device="cpu",
        progress=False,
    )

    # Three samples in total, so the last two are evaluated.
    assert results["dataset"]["evaluated_indices"] == [1, 2]


def test_evaluate_checkpoint_rejects_invalid_sample_count(
    tiny_checkpoint,
    tiny_dataset,
):
    metadata_path, _ = tiny_dataset

    with pytest.raises(ValueError, match="at least 1"):
        evaluate_checkpoint(
            checkpoint_path=tiny_checkpoint,
            metadata_path=metadata_path,
            samples=0,
        )


def test_evaluate_checkpoint_rejects_missing_checkpoint(tiny_dataset):
    metadata_path, _ = tiny_dataset

    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(
            checkpoint_path="checkpoints/does-not-exist.pt",
            metadata_path=metadata_path,
            samples=1,
            device="cpu",
        )


def test_main_writes_json_file(tiny_checkpoint, tiny_dataset, tmp_path):
    metadata_path, _ = tiny_dataset
    output_path = tmp_path / "results" / "evaluation.json"

    exit_code = main(
        [
            "--checkpoint",
            str(tiny_checkpoint),
            "--metadata",
            str(metadata_path),
            "--samples",
            "2",
            "--vocoder-samples",
            "1",
            "--n-iter",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "model" in payload
    assert "vocoder" in payload
    assert "generated_audio" in payload


def test_main_returns_error_for_missing_checkpoint(tiny_dataset, tmp_path):
    metadata_path, _ = tiny_dataset

    exit_code = main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--metadata",
            str(metadata_path),
            "--samples",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 1


def test_parser_defaults():
    args = build_parser().parse_args([])

    assert args.samples == 64
    assert args.vocoder_samples == 4
    assert args.n_iter == 32
    assert args.output == "outputs/evaluation.json"
