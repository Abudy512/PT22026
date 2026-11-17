
"""Tests for the training speed-ups: Mel cache, bucketed batches, LR schedule."""

from __future__ import annotations

import pytest
import torch

from voiceforge.batching import LengthBucketSampler
from voiceforge.dataset import CachedMelDataset, LJSpeechDataset
from voiceforge.training import _build_scheduler


# ----------------------------------------------------------------------
# Mel cache
# ----------------------------------------------------------------------
def test_cache_matches_uncached_mel(tmp_path, tiny_dataset):
    metadata_path, _ = tiny_dataset

    dataset = LJSpeechDataset(metadata_path)
    cached = CachedMelDataset(dataset, tmp_path / "cache")

    expected = dataset[0]["mel"]
    actual = cached[0]["mel"]

    assert torch.equal(expected, actual)


def test_cache_writes_then_reads(tmp_path, tiny_dataset):
    metadata_path, _ = tiny_dataset

    cached = CachedMelDataset(LJSpeechDataset(metadata_path), tmp_path / "cache")

    cached[1]
    assert cached.misses == 1
    assert cached.hits == 0

    cached[1]
    assert cached.misses == 1
    assert cached.hits == 1

    assert cached.cache_size_bytes() > 0


def test_cache_hit_does_not_need_the_audio_file(tmp_path, tiny_dataset):
    metadata_path, wavs_dir = tiny_dataset

    dataset = LJSpeechDataset(metadata_path)
    cached = CachedMelDataset(dataset, tmp_path / "cache")

    cached[2]

    # Remove every WAV: a cache hit must still work.
    for path in wavs_dir.glob("*.wav"):
        path.unlink()

    sample = cached[2]

    assert sample["mel"].shape[0] == 1
    assert sample["text_sequence"].numel() > 0


def test_cache_length_matches_dataset(tmp_path, tiny_dataset):
    metadata_path, _ = tiny_dataset

    dataset = LJSpeechDataset(metadata_path)
    cached = CachedMelDataset(dataset, tmp_path / "cache")

    assert len(cached) == len(dataset)


# ----------------------------------------------------------------------
# Length-bucketed sampler
# ----------------------------------------------------------------------
def test_bucketed_sampler_groups_similar_lengths():
    lengths = [100] * 12 + [500] * 12

    sampler = LengthBucketSampler(
        lengths,
        batch_size=4,
        buffer_batches=1,
        shuffle=False,
        seed=0,
    )

    for batch in sampler:
        batch_lengths = [lengths[index] for index in batch]
        assert max(batch_lengths) == min(batch_lengths)


def test_bucketed_sampler_covers_every_sample_once():
    lengths = list(range(1, 41))

    sampler = LengthBucketSampler(lengths, batch_size=4, seed=0)
    seen = [index for batch in sampler for index in batch]

    assert sorted(seen) == list(range(40))
    assert len(seen) == len(set(seen))


def test_bucketed_sampler_respects_batch_size():
    sampler = LengthBucketSampler([10] * 12, batch_size=5, seed=0)

    batches = list(sampler)

    assert all(len(batch) == 5 for batch in batches)
    assert len(batches) == 2  # drop_last removes the final 2 samples


def test_bucketed_sampler_without_drop_last_keeps_remainder():
    sampler = LengthBucketSampler(
        [10] * 12,
        batch_size=5,
        drop_last=False,
        seed=0,
    )

    batches = list(sampler)

    assert len(batches) == 3
    assert len(batches[-1]) == 2
    assert len(sampler) == 3


def test_bucketed_sampler_is_deterministic_for_same_seed():
    lengths = list(range(1, 41))

    first = list(LengthBucketSampler(lengths, batch_size=4, seed=123))
    second = list(LengthBucketSampler(lengths, batch_size=4, seed=123))

    assert first == second


def test_bucketed_sampler_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="at least 1"):
        LengthBucketSampler([1, 2, 3], batch_size=0)


# ----------------------------------------------------------------------
# Learning-rate schedule
# ----------------------------------------------------------------------
def test_scheduler_warms_up_then_decays():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)

    scheduler = _build_scheduler(
        optimizer,
        total_steps=20,
        warmup_steps=4,
        peak_learning_rate=1e-3,
        minimum_learning_rate=1e-4,
    )

    rates = []

    for _ in range(20):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Warm-up: increasing from the first step.
    assert rates[0] < rates[1] < rates[3]

    # Decay: strictly falling after warm-up.
    assert rates[5] > rates[10] > rates[-1]

    # Ends near the requested floor.
    assert rates[-1] == pytest.approx(1e-4, rel=0.2)


def test_scheduler_reaches_peak_at_end_of_warmup():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)

    scheduler = _build_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=5,
        peak_learning_rate=1e-3,
        minimum_learning_rate=1e-4,
    )

    for _ in range(5):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, rel=0.15)
