# VoiceForge

**An English text-to-speech application with a pretrained natural-speech engine
and a student-built acoustic model.**

VoiceForge is a university project with **two speech engines**:

1. **Kokoro-82M** — a pretrained, open-weight TTS model by
   [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0). This is
   the default engine and is what makes the application produce natural
   speech. It was **not** trained by this project; the work here is the
   integration, interface, testing and evaluation. See
   [docs/kokoro.md](docs/kokoro.md).
2. **The VoiceForge custom model** — a Tacotron-lite acoustic model written
   and trained from scratch by the student (text processing, attention
   encoder-decoder, duration predictor, Mel spectrograms, Griffin-Lim). It is
   kept as an experimental backend and as evidence of the development work.

> **Current status:** both engines work in the application. Kokoro produces
> natural speech at ~30× faster than real time. The student's own model is
> **not intelligible** — that is measured and documented in
> [docs/evaluation.md](docs/evaluation.md), not hidden.

| | |
|---|---|
| Course | Professional Training (YN3012170034) |
| University | Yunnan University |
| Student | Almaqtari Abdulrahman Mohammed Najib (20243120042) |
| Branch | `alm042` |
| Main engine | Kokoro-82M (pretrained, Apache-2.0) |
| Custom model | Tacotron-lite, 2,997,410 parameters |
| Dataset (custom model) | LJSpeech 1.1 (13,100 clips, single speaker, 22,050 Hz) |
| Tests | 194 automated tests (1 optional Kokoro test skipped by default) |

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

- **Two speech engines** — pretrained Kokoro-82M for natural speech (default)
  and the student-built model, selectable in the interface.
- **Character-level text normalisation** — smart quotes, accents, brackets and
  whitespace are normalised before encoding (custom model).
- **Attention-based acoustic model** — convolutional encoder, bidirectional
  GRU, location-sensitive attention, GRU decoder, postnet and a duration
  predictor (student-built).
- **Two decoding modes** — length-controlled decoding (stable) and fully
  autoregressive decoding with a stop token.
- **Griffin-Lim vocoder** — reproducible, dependency-free waveform
  reconstruction for the custom model; Kokoro uses an iSTFTNet decoder.
- **Seven Kokoro voices** — American English female and male voices.
- **Command line generation** — `python -m scripts.generate`.
- **Interactive web application** — engine and voice selectors, text input,
  playback, WAV download, validation messages and a loading indicator.
- **Python library** — `from voiceforge import VoiceForgeTTS` and
  `from voiceforge.kokoro_backend import KokoroBackend`.
- **Objective evaluation tools** — Mel error, duration error, spectral detail,
  vocoder round-trip and audio defect detection.
- **194 automated tests** covering text, audio, dataset, model, checkpointing,
  training, generation, evaluation, both engines, the application and the
  public API.

## 3. How it works

### Kokoro-82M (default engine)

```
 English text → misaki G2P → phonemes → text encoder + duration predictor
              → style-conditioned features → iSTFTNet decoder → 24 kHz waveform
```

### VoiceForge custom model (experimental)

The full model, training configuration and measured limitations are described
in [docs/architecture.md](docs/architecture.md) and
[docs/training-v2.md](docs/training-v2.md).

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
├── tests/                        # 194 automated tests
└── voiceforge/
    ├── __init__.py               # Public API
    ├── app_core.py               # Application logic (shared by UIs)
    ├── architectures.py          # Architecture identifiers
    ├── audio.py                  # Audio I/O, Mel, Griffin-Lim
    ├── batching.py               # Padding, collation, length bucketing
    ├── checkpoint.py             # Save/load and architecture dispatch
    ├── dataset.py                # LJSpeech loader + Mel cache
    ├── kokoro_backend.py         # Kokoro-82M pretrained backend
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

### Kokoro weights (first run)

`pip install -r requirements.txt` also installs the `kokoro` package. The
pretrained weights (~330 MB) are downloaded from Hugging Face the first time
the Kokoro engine is used, then cached. If huggingface.co is unreachable, use
a mirror:

```bash
set HF_ENDPOINT=https://hf-mirror.com      # Windows
export HF_ENDPOINT=https://hf-mirror.com   # Linux / macOS
```

Set `HF_HOME` to control where the cache lives. Details:
[docs/kokoro.md](docs/kokoro.md).

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

### Kokoro-82M (pretrained engine)

```python
from voiceforge.kokoro_backend import KokoroBackend

tts = KokoroBackend(voice="af_heart")      # or af_bella, am_michael, ...
tts.save("Hello, this is Kokoro speaking through VoiceForge.", "out.wav")
```

24 kHz output, ~0.03 real-time factor after warm-up. See
[docs/kokoro.md](docs/kokoro.md).

## 10. Running the application

```bash
python app.py
```

Then open <http://127.0.0.1:7860>.

Useful options:

```bash
python app.py --engine kokoro --voice am_michael   # pretrained engine (default)
python app.py --engine voiceforge                  # student-built model
python app.py --engine auto --open                 # pick automatically
python app.py --device cpu                         # force CPU
python app.py --port 7861 --open
```

The interface provides:

- a **speech engine selector** (Kokoro / VoiceForge custom),
- a **Kokoro voice selector** (seven American English voices),
- an English text box with a character counter and example sentences,
- a **Generate Speech** button with a loading indicator,
- in-browser playback and a **Download WAV** link,
- clear validation and error messages, with a notice explaining which engine
  is pretrained and which is the student's own experimental model.

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
194 passed, 1 skipped
```

The suite covers text processing, audio analysis and reconstruction, dataset
parsing and filtering, model shapes and masking, checkpoint round-trips and
architecture dispatch, the training loop, generation, evaluation metrics,
audio defect detection, both speech engines, the web application (through a
real HTTP server, including downloads and error responses) and the public API.

Tests that need the LJSpeech dataset skip automatically when it is absent. The
one optional test that performs real Kokoro generation runs only with:

```bash
set VOICEFORGE_TEST_KOKORO=1
python -m pytest tests/test_kokoro_backend.py -q
```

## 13. Known limitations

### Kokoro-82M (main engine)

1. **Pretrained, not ours.** The model was trained by hexgrad; this project
   integrates and evaluates it. No fine-tuning was performed.
2. **First-run download.** ~330 MB from Hugging Face unless the cache is
   pre-populated.
3. **English-first.** Only American English voices are exposed.
4. **Grapheme-to-phoneme errors.** Unusual proper nouns can be mispronounced
   by `misaki`/`espeak-ng`.
5. **No prosody control** beyond voice choice and a global speed factor.

### VoiceForge custom model (experimental)

1. **Not intelligible.** The student confirmed by listening that no words are
   understandable, despite strong alignment improvements in v2.
2. **Small training budget.** 6,000 clips and 5,000 steps is far below what a
   production Tacotron needs.
3. **Exposure bias.** Teacher-forced error 0.38, free-running 5.14 in v2.
4. **Griffin-Lim ceiling.** A phase-reconstruction baseline, not a neural
   vocoder.
5. **English, single speaker.** 38-character ASCII vocabulary and no digits.
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
| [docs/kokoro.md](docs/kokoro.md) | Kokoro-82M: why, architecture, licence, install, performance, comparison |
| [docs/architecture.md](docs/architecture.md) | Custom model design, losses, training configuration, limitations |
| [docs/training-v2.md](docs/training-v2.md) | The v2 training run: configuration, speed-ups, measured results |
| [docs/evaluation.md](docs/evaluation.md) | Metric definitions, before/after results, interpretation |
| [docs/api.md](docs/api.md) | Python API reference, errors, timings |

### Acknowledgements

- **Kokoro-82M** by hexgrad — <https://huggingface.co/hexgrad/Kokoro-82M> —
  Apache-2.0. Used as the pretrained main engine; **not trained by this
  project**.
- **LJSpeech 1.1** — Keith Ito and Linda Johnson, public domain dataset.
- **Tacotron** (Wang et al., 2017) and **Tacotron 2** (Shen et al., 2018) —
  architecture inspiration for the custom model.
- **Chorowski et al., 2015** — location-sensitive attention.
- **Griffin & Lim, 1984** — phase reconstruction algorithm.
- **PyTorch** and **torchaudio** for the implementation platform.

The technical report for the course is prepared separately and is not part of
this repository yet.
