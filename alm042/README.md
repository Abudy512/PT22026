# VoiceForge

**A custom English text-to-speech system built from scratch with PyTorch.**

VoiceForge is a university project that implements the complete text-to-speech
pipeline itself — text normalisation, a character-level acoustic model, Mel
spectrogram generation, Griffin-Lim waveform reconstruction, a Python library
and an interactive web application. It does **not** wrap a pretrained TTS
model or call an external speech service.

> **Current status:** the pipeline is complete and tested, but the acoustic
> model is small and trained on a limited subset of LJSpeech. The generated
> audio is recognisably synthesised speech and is **not expected to be fully
> intelligible**. The measurements behind that statement are in
> [docs/evaluation.md](docs/evaluation.md).

| | |
|---|---|
| Course | Professional Training (YN3012170034) |
| University | Yunnan University |
| Student | Almaqtari Abdulrahman Mohammed Najib (20243120042) |
| Branch | `alm042` |
| Model | Tacotron-lite, 2,997,410 parameters |
| Dataset | LJSpeech 1.1 (13,100 clips, single speaker, 22,050 Hz) |
| Tests | 165 automated tests |

---

## Table of contents

1. [What VoiceForge does](#1-what-voiceforge-does)
2. [Features](#2-features)
3. [How it works](#3-how-it-works)
4. [Repository structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Dataset setup](#6-dataset-setup)
7. [Model checkpoints](#7-model-checkpoints)
8. [Training](#8-training)
9. [Generating speech](#9-generating-speech)
10. [Running the application](#10-running-the-application)
11. [Evaluation](#11-evaluation)
12. [Testing](#12-testing)
13. [Known limitations](#13-known-limitations)
14. [Future improvements](#14-future-improvements)
15. [Documentation](#15-documentation)

---

## 1. What VoiceForge does

A text-to-speech (TTS) system turns written text into a spoken waveform. The
hard part is not producing sound — it is learning *which* part of the audio
belongs to *which* character, and doing so with a single speaker and a small
GPU.

VoiceForge solves this with its own attention-based acoustic model:

```
 English text
      │
      ▼
 normalise text ──► characters ──► character IDs (38-symbol vocabulary)
      │
      ▼
 acoustic model (Tacotron-lite) ──► Mel spectrogram (80 bands × time)
      │
      ▼
 Griffin-Lim phase reconstruction ──► waveform (22,050 Hz)
      │
      ▼
 WAV file / browser playback / Python API
```

## 2. Features

- **Own model, own pipeline** — no pretrained TTS weights, no external API.
- **Character-level text normalisation** — smart quotes, accents, brackets and
  whitespace are normalised before encoding.
- **Attention-based acoustic model** — convolutional encoder, bidirectional
  GRU, location-sensitive attention, GRU decoder, postnet and a duration
  predictor.
- **Two decoding modes** — length-controlled decoding (stable) and fully
  autoregressive decoding with a stop token.
- **Griffin-Lim vocoder** — reproducible, dependency-free waveform
  reconstruction.
- **Command line generation** — `python -m scripts.generate`.
- **Interactive web application** — text input, playback, WAV download,
  validation messages and a loading indicator.
- **Python library** — `from voiceforge import VoiceForgeTTS`.
- **Objective evaluation tools** — Mel error, duration error, spectral detail,
  vocoder round-trip and audio defect detection.
- **165 automated tests** covering text, audio, dataset, model, checkpointing,
  training, generation, evaluation, the application and the public API.

## 3. How it works

| Stage | What happens | Where |
|---|---|---|
| Normalisation | Lower-case, ASCII, standard punctuation, collapsed whitespace | `voiceforge/text.py` |
| Encoding | 38-character vocabulary → integer IDs | `voiceforge/text.py` |
| Mel analysis | 80-band log-Mel, `n_fft=1024`, `hop=256` | `voiceforge/audio.py` |
| Acoustic model | Encoder → attention → decoder → postnet | `voiceforge/model.py` |
| Reconstruction | Inverse Mel scale + Griffin-Lim | `voiceforge/audio.py` |
| Application | Validation, model caching, HTTP interface | `voiceforge/app_core.py`, `voiceforge/web_app.py` |

The full architecture, including layer shapes, parameter counts, loss terms
and decoding modes, is documented in
[docs/architecture.md](docs/architecture.md).

### Model at a glance

| Component | Output | Parameters |
|---|---|---|
| Character embedding | (batch, text_len, 128) | 4,864 |
| 3 × Conv1d + BatchNorm + ReLU | (batch, 256, text_len) | 821,504 |
| Bidirectional GRU | (batch, text_len, 256) | 296,448 |
| Location-sensitive attention | context (batch, 256) | 193,632 |
| Decoder GRU + Mel/stop heads | (batch, 80) per frame | 632,913 |
| Postnet (4 × Conv1d) | residual correction | 862,704 |
| Duration head | log frames | 33,025 |
| **Total** | | **2,997,410** |

## 4. Repository structure

```
alm042/
├── app.py                        # Launch the web application
├── app_gradio.py                 # Optional Gradio front-end (needs gradio)
├── requirements.txt
├── checkpoints/                  # Model weights (git-ignored)
├── data/                         # LJSpeech dataset (git-ignored)
├── docs/
│   ├── api.md                    # Python API reference
│   ├── architecture.md           # Model and training design
│   └── evaluation.md             # Metrics, results and interpretation
├── outputs/                      # Generated audio (git-ignored)
├── scripts/
│   ├── check_audio.py            # Detect NaN, silence, clipping, buzz
│   ├── check_text_vocabulary.py  # Verify dataset text is encodable
│   ├── evaluate.py               # Objective model evaluation
│   ├── generate.py               # Command-line speech generation
│   └── train.py                  # Training entry point
├── tests/                        # 165 automated tests
└── voiceforge/
    ├── __init__.py               # Public API
    ├── app_core.py               # Application logic (shared by UIs)
    ├── architectures.py          # Architecture identifiers
    ├── audio.py                  # Audio I/O, Mel, Griffin-Lim
    ├── batching.py               # Padding and collation
    ├── checkpoint.py             # Save/load and architecture dispatch
    ├── dataset.py                # LJSpeech loader
    ├── model.py                  # Tacotron-lite model
    ├── model_legacy.py           # Original baseline model (preserved)
    ├── text.py                   # Normalisation and vocabulary
    ├── training.py               # Training loop and losses
    ├── tts.py                    # VoiceForgeTTS inference pipeline
    ├── web_app.py                # HTTP interface
    └── templates/index.html      # Web UI
```

## 5. Installation

Requires **Python 3.11+**.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

PyTorch is listed without a version so pip installs a build matching your
machine. For CUDA, install the wheel from the official index first, for
example:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Verify the environment:

```bash
python -c "import torch, torchaudio, soundfile; print(torch.__version__, torch.cuda.is_available())"
```

The web interface (`app.py`) uses only the Python standard library, so no
extra packages are needed. See [section 10](#10-running-the-application) for
the optional Gradio front-end.

## 6. Dataset setup

VoiceForge uses the public **LJSpeech 1.1** dataset (13,100 short English
clips, one female speaker, about 24 hours, 22,050 Hz).

1. Download it from <https://keithito.com/LJ-Speech-Dataset/>.
2. Extract it so that the folder layout is exactly:

```
data/
└── LJSpeech-1.1/
    └── LJSpeech-1.1/
        ├── metadata.csv
        └── wavs/
            ├── LJ001-0001.wav
            └── ...
```

3. Verify that every transcript can be encoded by VoiceForge's 38-character
   vocabulary:

```bash
python -m scripts.check_text_vocabulary
```

Expected output ends with `SUCCESS: All dataset transcripts are supported.`

The dataset is **not** included in this repository (several gigabytes). It is
excluded by `.gitignore`.

## 7. Model checkpoints

Checkpoints are also excluded from Git (about 34 MB each). Two are used:

| File | Description |
|---|---|
| `checkpoints/voiceforge_attn.pt` | **Current model.** Tacotron-lite, epoch 14, validation loss 1.3123 |
| `checkpoints/voiceforge.pt` | Original baseline (GRU + linear interpolation), kept for comparison |

To obtain them you can either:

* copy the trained files into `checkpoints/`, or
* train a fresh model with the command in [section 8](#8-training).

Checkpoints store their architecture and hyper-parameters, so both models load
automatically:

```python
from voiceforge import load_model

model, config, metadata = load_model("checkpoints/voiceforge_attn.pt")
print(config["architecture"], metadata["epoch"])
```

Checkpoints created before the duration predictor existed are detected and
reported with a clear error instead of being loaded incorrectly.

## 8. Training

Training is CPU/GPU agnostic and controlled from the command line:

```bash
python -m scripts.train \
    --max-samples 3000 \
    --val-samples 64 \
    --epochs 24 \
    --batch-size 32 \
    --max-seconds 5 \
    --learning-rate 0.001 \
    --checkpoint checkpoints/voiceforge_attn.pt \
    --history outputs/training_history.json
```

Options worth knowing:

| Option | Meaning |
|---|---|
| `--max-samples` | Number of training rows (0 = all 13,036 usable rows) |
| `--max-seconds` | Keep only clips shorter than this (speed on small GPUs) |
| `--batch-size` | Batch size; 32 fits comfortably in 6 GB |
| `--val-samples` | Held-out rows used for validation and best-checkpoint selection |
| `--arch` | `tacotron_lite` (default) or the legacy baseline |
| `--amp` | Mixed precision (measured slower for this model; off by default) |

What the last full run actually did:

| Setting | Value |
|---|---|
| Data | 3,000 clips ≤ 5.0 s |
| Epochs run | 14 (stopped when validation plateaued) |
| Batch size | 32 |
| Optimiser | Adam, lr 1e-3, gradient clipping 1.0 |
| Training time | 50.8 minutes on an RTX 3060 Laptop (6 GB, ~1.6 GB used) |
| Best validation loss | 1.3123 |

The script refuses to select the evaluator's held-out rows, saves the best
epoch, writes a JSON history, and verifies the saved checkpoint can be
reloaded before reporting success.

## 9. Generating speech

### Command line

```bash
python -m scripts.generate "Hello, welcome to VoiceForge!" --output outputs/hello.wav
```

Several sentences at once:

```bash
python -m scripts.generate --output-dir outputs/samples \
    "First sentence." "Second sentence."
```

The script validates the checkpoint, writes the WAV file, reads it back and
prints a JSON summary (duration, sample count, peak level).

### Python

```python
from voiceforge import VoiceForgeTTS

tts = VoiceForgeTTS.from_pretrained("checkpoints/voiceforge_attn.pt")
waveform = tts.synthesize("Hello, welcome to VoiceForge!")   # torch.Tensor
tts.save("Hello, welcome to VoiceForge!", "output.wav")
```

One-shot helper:

```python
from voiceforge import generate_speech

generate_speech("Hello world.", "output.wav")
```

The complete API (arguments, attributes, errors, timings) is documented in
[docs/api.md](docs/api.md).

## 10. Running the application

```bash
python app.py
```

Then open <http://127.0.0.1:7860>.

Useful options:

```bash
python app.py --checkpoint checkpoints/voiceforge_attn.pt --port 7861 --open
python app.py --device cpu          # force CPU
```

The interface provides:

- an English text box with a character counter and example sentences,
- a **Generate Speech** button with a loading indicator,
- in-browser playback and a **Download WAV** link,
- clear validation and error messages (empty text, unsupported characters,
  missing checkpoint),
- an explicit notice that the model is experimental.

Everything runs locally; no text or audio leaves the machine.

### Optional Gradio front-end

A second front-end is included for the originally planned framework:

```bash
pip install gradio
python app_gradio.py
```

> **Note:** `app_gradio.py` could not be executed in the development
> environment because Gradio was not installed and the machine had no network
> access to install it. It uses the same application core as `app.py`, so the
> model integration is identical, but treat it as untested until you run it.
> `app.py` is the verified interface.

## 11. Evaluation

Reproduce the numbers in [docs/evaluation.md](docs/evaluation.md):

```bash
# Acoustic model, duration and vocoder metrics on 64 held-out sentences
python -m scripts.evaluate --checkpoint checkpoints/voiceforge_attn.pt \
    --samples 64 --vocoder-samples 4 --output outputs/evaluation.json

# Audio defect checks (NaN, silence, clipping, DC offset, flat buzz)
python -m scripts.check_audio --checkpoint checkpoints/voiceforge_attn.pt \
    --output-dir outputs/audio_checks
```

Summary of the measured comparison (identical 64 held-out samples):

| Metric | Baseline | Current model | Verdict |
|---|---|---|---|
| Mel L1 (lower better) | 3.02 | 5.70 | worse (exposure bias) |
| Duration error | 25.9 % | **14.3 %** | better |
| Spectral detail (target 1.0) | 0.48 | **0.61** | better |
| Vocoder Mel correlation | 0.979 | 0.979 | unchanged |
| Audio files passing all checks | 1 / 4 | **4 / 4** | better |

## 12. Testing

```bash
python -m pytest tests -q
```

```
165 passed
```

The suite covers text processing, audio analysis and reconstruction, dataset
parsing and filtering, model shapes and masking, checkpoint round-trips and
architecture dispatch, the training loop, generation, evaluation metrics,
audio defect detection, the web application (through a real HTTP server,
including downloads and error responses) and the public API.

Tests that need the LJSpeech dataset skip automatically when it is absent.

## 13. Known limitations

1. **Not intelligible.** The model reproduces rhythm well (1.01× duration on a
   memorised sentence) but correlates only 0.37 with the real spectrogram.
   Attention remains diffuse, so the sounds are not accurate.
2. **Small training budget.** 3,000 short clips for 14 epochs is far below the
   tens of hours and thousands of steps a production Tacotron needs.
3. **Exposure bias.** Teacher-forced error is 0.50, free-running error is 5.70:
   the decoder drifts when fed its own output.
4. **Griffin-Lim ceiling.** It is a phase-reconstruction baseline, not a neural
   vocoder; the result is inherently buzzy.
5. **English, single speaker.** 38-character ASCII vocabulary, no digits, no
   prosody or emotion control.
6. **CPU inference is slow.** GPU generation takes ~0.4–0.7 s per sentence.
7. **Gradio front-end untested** (see section 10).

## 14. Future improvements

- Train much longer on the full dataset (10k+ steps) and use scheduled
  sampling to reduce exposure bias.
- Replace location-sensitive attention with forward attention or stepwise
  monotonic attention for faster, more reliable alignment.
- Replace Griffin-Lim with a small neural vocoder (HiFi-GAN).
- Move from characters to phonemes for better pronunciation.
- Add multi-speaker support and prosody controls.

## 15. Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Model design, losses, training configuration, limitations |
| [docs/evaluation.md](docs/evaluation.md) | Metric definitions, before/after results, interpretation |
| [docs/api.md](docs/api.md) | Python API reference, errors, timings |

### Acknowledgements

- **LJSpeech 1.1** — Keith Ito and Linda Johnson, public domain dataset.
- **Tacotron** (Wang et al., 2017) and **Tacotron 2** (Shen et al., 2018) —
  architecture inspiration.
- **Chorowski et al., 2015** — location-sensitive attention.
- **Griffin & Lim, 1984** — phase reconstruction algorithm.
- **PyTorch** and **torchaudio** for the implementation platform.

The technical report for the course is prepared separately and is not part of
this repository yet.
