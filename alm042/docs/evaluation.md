# VoiceForge Evaluation

This document explains how VoiceForge is evaluated, what each number means,
and what the measurements showed. The numbers in this file are produced by
`scripts/evaluate.py`; they are copied from real runs and are not estimates.

Evaluation is deliberately split into three parts. A text-to-speech system
can fail in three different places, and mixing the three measurements
together hides which part is broken.

| Part | Question it answers |
|---|---|
| Acoustic model | Is the predicted Mel spectrogram close to real speech, and is the predicted duration right? |
| Vocoder | How much detail does Griffin-Lim lose when it reconstructs *real* speech? |
| Generated audio | Does audio produced from text have the amplitude structure of speech? |

## How to run

```bash
python -m scripts.evaluate \
    --checkpoint checkpoints/voiceforge.pt \
    --samples 64 \
    --vocoder-samples 4 \
    --n-iter 32 \
    --output outputs/evaluation.json
```

The script writes a JSON file and prints the same content. Per-sample
progress is printed unless `--quiet` is used.

### Held-out samples

The evaluator uses the **last** `--samples` rows of `metadata.csv`
(indices `13036`–`13099` for `--samples 64`). Training uses a prefix of the
dataset, so these samples are not seen during training. This is what makes
the comparison a test of generalisation rather than memorisation.

## Metric definitions

### `mel_l1` (lower is better)

Mean absolute error between the log-Mel spectrogram predicted by the model
and the real log-Mel spectrogram, computed only over valid (non-padded)
frames. The log-Mel values are `log(power + 1e-5)`, so the units are
natural-log power. A perfect model would score 0.0.

### `duration_mae_frames` and `duration_relative_error` (lower is better)

The model predicts how many Mel frames a sentence should last. These compare
that prediction with the true number of frames. One frame is 256 samples
(about 11.6 ms) at 22,050 Hz. `duration_relative_error` is the absolute
error divided by the true length, so 0.25 means "off by 25 % on average".

### `spectral_detail_ratio` (closer to 1.0 is better)

The standard deviation of the predicted Mel divided by the standard
deviation of the real Mel. A value near 1.0 means the prediction has a
similar amount of spectral detail as real speech. A value well below 1.0
means the model is producing an over-smoothed, blurry spectrogram — the
classic failure mode of under-trained or poorly aligned TTS models.

### `mel_rmse` and `mel_correlation` (vocoder only)

A **real** Mel spectrogram is converted to audio with Griffin-Lim and then
converted back to a Mel spectrogram. The difference between the original and
the round-tripped version measures the reconstruction algorithm alone,
because the input was correct. `mel_correlation` is the Pearson correlation
between the two flattened spectrograms.

### `generated_energy_variation` vs `real_energy_variation`

The coefficient of variation (standard deviation ÷ mean) of short-term
energy, computed with 1024-sample windows. Natural speech switches between
loud and quiet syllables, so real speech has a value well above zero. A flat
electronic buzz has a value close to zero. If the generated value is much
lower than the real value, the generated audio lacks speech-like dynamics.

## Baseline results

Model: the original `legacy_interpolate` architecture, 2 epochs on 128
LJSpeech samples (`checkpoints/voiceforge.pt`, epoch 2, reported training
loss 3.0855).

Command:

```bash
python -m scripts.evaluate --checkpoint checkpoints/voiceforge.pt \
    --samples 64 --vocoder-samples 4 --n-iter 32 --quiet
```

| Measurement | Value |
|---|---|
| Samples evaluated (held-out) | 64 (indices 13036–13099) |
| `mel_l1` | **3.0158** |
| `duration_mae_frames` | **151.56 frames** |
| `duration_relative_error` | **25.86 %** |
| `predicted_mel_std` | 1.8968 |
| `target_mel_std` | 3.9511 |
| `spectral_detail_ratio` | **0.4801** |
| Vocoder `mel_rmse` (real Mel round-trip) | 0.8584 |
| Vocoder `mel_correlation` | 0.9789 |
| `generated_energy_variation` | **0.0999** |
| `real_energy_variation` | 0.7967 |
| Average generated length | 5.43 s |

### What these numbers mean

1. **The vocoder works.** A real Mel spectrogram survives the Griffin-Lim
   round-trip with a correlation of 0.979 and an RMSE of 0.86. The
   reconstruction step is therefore not the main problem.

2. **The acoustic model is the problem.** `mel_l1` of 3.02 is very high and
   `spectral_detail_ratio` of 0.48 means the predicted spectrogram has less
   than half the detail of real speech. The model produces a smooth average
   of possible sounds rather than the sharp harmonic structure of speech.

3. **The duration predictor is poor.** Being 151 frames (about 1.8 seconds)
   off on average, or 26 %, is not accurate enough for intelligible
   prosody.

4. **The generated audio does not sound like speech.** Its energy variation
   (0.10) is roughly eight times flatter than real speech (0.80), which is
   consistent with a buzzing, droning output rather than syllables.

The root cause is architectural: the original model stretches each character
embedding linearly across time with `F.interpolate`. It has no mechanism to
learn *which* audio frames belong to *which* character, so it cannot learn
pronunciation or timing. This is analysed in the technical report.

## Important limitation of these metrics

These measurements are automatic and objective, but they **do not measure
intelligibility**. A spectrogram can have low error and still be
unintelligible, and vice versa. A proper evaluation would need a listening
test (for example MOS scoring with several listeners), which was not
performed for this project. No subjective quality claim is made anywhere in
this repository unless it is backed by an actual listening test.
