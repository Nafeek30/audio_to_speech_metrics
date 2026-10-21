# Speech Analysis & Clinical Language Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-158%20passed-brightgreen.svg)]()

An end-to-end clinical speech and language processing system designed to take **spoken audio or continuous transcripts**, transcribe and diarize speakers, segment speech into clinical Communication Units (**C-units** / utterances) using a specialized transformer segmenter, and compute standardized **Clinical Language Sample Analysis (LSA)** metrics (CLAN MLU, FREQ, and TAALED lexical diversity).

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Quickstart & Installation](#quickstart--installation)
3. [Where Do I Put My Files?](#where-do-i-put-my-files)
4. [How to Run the Code](#how-to-run-the-code)
   - [Mode 1: Full End-to-End Analysis (Audio → Metrics)](#1-full-end-to-end-analysis-audio--metrics)
   - [Mode 2: Audio Transcription Only (Audio → Text)](#2-audio-transcription-only-audio--text)
   - [Mode 3: Speech Metric Analysis Only (Text → Metrics)](#3-speech-metric-analysis-only-text--metrics)
5. [Understanding Output Artifacts](#understanding-output-artifacts)
6. [Supported Clinical & Lexical Metrics](#supported-clinical--lexical-metrics)
7. [Research Benchmarks & Scientific Guarantees](#research-benchmarks--scientific-guarantees)
8. [Testing & Verification](#testing--verification)

---

## System Architecture

```
[Raw Audio Recording (.wav, .mp3)] 
       │ 
       ▼ (1. Upstream Diarization + ASR: Whisper / MOSS / SenseVoice / NeMo)
[Diarized Transcript with Speaker IDs]
       │
       ▼ (2. Target Speaker Isolation: Extract Target Child Speech)
[Target Child Raw Speech Stream]
       │
       ▼ (3. Normalization: Lowercase, strip punctuation into unbroken continuous stream)
[Pure Continuous Paragraph Stream]
       │
       ▼ (4. Clinical Utterance Segmenter: MiniLM-L12 + Auxiliary Count Loss Head)
[Predicted Clinical Communication Units (C-Units)]
       │
       ▼ (5. Downstream Clinical & Lexical Engines: Native CLAN & TAALED)
[Clinical LSA Metrics: NTW, NDW, NTU, MLUw, Length-SD, MATTR50, HD-D42, MTLD]
```

* **Canonical Pure Paragraph Mode**: Operates directly on unpunctuated, continuous speech text without requiring conversational turn breaks or interlocutor cues.

---

## Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nafeek30/audio_to_speech_metrics.git
cd audio_to_speech_metrics
```

### 2. Environment Setup (Recommended: `uv` or Python `venv`)
```bash
# Using uv (fastest)
uv sync

# Or using standard python3 venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Download the SpaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

---

## Where Do I Put My Files?

Organize your inputs and models as follows:

| File Type | Recommended Location | Supported Formats | Notes |
| :--- | :--- | :--- | :--- |
| **Input Audio** | `data/audio/` (or pass any path with `--audio`) | `.wav`, `.mp3`, `.m4a`, `.flac` | 16 kHz mono recommended. |
| **Input Plain Text** | `data/transcripts/` (or pass `--transcript`) | `.txt` | Plain speech text; punctuation is automatically handled. |
| **Input Multi-Speaker JSON** | `data/transcripts/` (or pass `--transcript`) | `.json` | Structured JSON with `"target_speaker"` and `"rows"`. |
| **Output Results** | `outputs/<experiment_name>/` | Folder | Generated automatically by `--output <dir>`. |
| **Deployed Model Weights** | `outputs/results/cv_hyperparameter_v4/lr2_count1/final/model/` | `.safetensors`, `segmenter.json` | Default production model. |

---

## How to Run the Code

A unified runner script [`scripts/run_pipeline.py`](scripts/run_pipeline.py) provides clean execution for all three standard workflows:

### 1. Full End-to-End Analysis (Audio → Metrics)
Transcribes an audio file with Whisper ASR, extracts the speech words, segments them into C-units, and computes clinical CLAN and TAALED metrics.

```bash
python scripts/run_pipeline.py --mode end-to-end --audio data/sample.wav --output outputs/sample_run/
```

* Optional flags:
  * `--asr-model <hf_model_id>`: Change Whisper model (default: `openai/whisper-small`).
  * `--device {cpu,cuda,mps}`: Set compute device (default: `cpu`).
  * `--segmenter {improved,rules,auto}`: Choose between trained transformer (`improved`) or rule heuristics (`rules`).

---

### 2. Audio Transcription Only (Audio → Text)
Runs automatic speech recognition and outputs raw text and timestamped JSON.

```bash
python scripts/run_pipeline.py --mode transcribe --audio data/sample.wav --output outputs/transcription_run/
```

---

### 3. Speech Metric Analysis Only (Text → Metrics)

#### A. From a Continuous Text Transcript (`.txt`)
Analyze narrative stories or unbroken speech without prior punctuation:

```bash
python scripts/run_pipeline.py --mode metrics --transcript data/examples/sample_story.txt --output outputs/story_metrics/
```

#### B. From a Conversational Turn JSON Transcript (`.json`)
For multi-speaker conversations where you want to isolate a specific speaker (e.g. `"CHI"` for child):

```bash
python scripts/run_pipeline.py --mode metrics --transcript data/examples/sample_dialogue.json --output outputs/dialogue_metrics/
```

*Example JSON format (`data/examples/sample_dialogue.json`):*
```json
{
  "target_speaker": "CHI",
  "rows": [
    {"speaker": "EXP", "text": "What did you see at the zoo?"},
    {"speaker": "CHI", "text": "I saw two monkeys and they were eating yellow bananas"},
    {"speaker": "EXP", "text": "Did you see anything else?"},
    {"speaker": "CHI", "text": "We also saw a big tall giraffe"}
  ]
}
```

---

## Understanding Output Artifacts

Every metric run produces three artifacts in the designated `--output` folder:

1. **`transcript.cha`**:
   * Standardized **CHAT-format transcript** compatible with official CLAN tools (`mlu`, `freq`, `kideval`).
   * Cleanly separated into predicted C-units with `*PAR:` speaker tier.
2. **`results.json`**:
   * Complete structured JSON containing exact utterance text, boundary tokens, CLAN command outputs, and granular TAALED lexical densities.
3. **`summary.txt`**:
   * Plaintext executive scorecard printed directly to console and saved to disk.

```
================================================================
              SPEECH PIPELINE ANALYSIS REPORT               
================================================================
Input Source:       data/examples/sample_story.txt
Execution Time:     2026-10-21T05:00:18+00:00
Segmenter Model:    pretrained_joint_gap_segmentation (MiniLM-L12)
Context Mode:       paragraph_ml
----------------------------------------------------------------
--- CLINICAL LANGUAGE SAMPLE ANALYSIS (LSA) METRICS ---
  Total Utterances (NTU):      8
  Total Words (NTW/TNW):       53.0
  Number of Diff Words (NDW):  34.0
  Type-Token Ratio (TTR):      0.642
  Mean Length of Utt (MLUw):   6.62 words
  Utterance Length SD:         2.18
----------------------------------------------------------------
--- LEXICAL DIVERSITY & DENSITY (TAALED) ---
  Moving-Average TTR (MATTR50): 0.610
  Vocd / HD-D (HDD42):          0.660
  Measure of Textual Div (MTLD):28.77
================================================================
```

---

## Supported Clinical & Lexical Metrics

| Category | Metric | Description | Engine |
| :--- | :--- | :--- | :--- |
| **Productivity** | **NTW / TNW** | Total number of spoken words/tokens | Native CLAN (`freq`) |
| | **NTU** | Total number of communication units / utterances | Native CLAN (`mlu`) |
| **Syntactic Complexity** | **MLUw** | Mean Length of Utterance in words ($NTW / NTU$) | Native CLAN (`mlu`) |
| | **Length SD** | Standard deviation of utterance length in words | Native CLAN (`mlu`) |
| **Vocabulary Diversity** | **NDW** | Number of Different Words (types) | Native CLAN (`freq`) |
| | **TTR** | Type-Token Ratio ($NDW / NTW$) | Native CLAN (`freq`) |
| **Lexical Diversity** | **MATTR50** | Moving-Average Type-Token Ratio (50-word window) | TAALED 1.4.1 |
| | **HD-D42 / Vocd** | Hypergeometric Distribution Diversity (sample 42) | TAALED 1.4.1 |
| | **MTLD** | Measure of Textual Lexical Diversity | TAALED 1.4.1 |

---

## Research Benchmarks & Scientific Guarantees

This pipeline was developed and validated under strict research standards:

1. **Zero-Shot External Evaluation Guarantee**:
   * External benchmarks (**ENNI**, **LUCID**, **DementiaBank**) are held-out zero-shot test datasets only. No external data was ever used for backpropagation, model adaptation, or hyperparameter selection.
2. **Immutable Participant Cohorts**:
   * Strict participant-independent splits sealed by SHA-256 hash in [`outputs/validation/cohorts_v1/manifest.json`](outputs/validation/cohorts_v1/manifest.json).
3. **External Benchmark Highlights**:
   * **OCSC Benchmark (111 Conversational Test Sessions)**: Our pipeline achieved **85.56% F1** (+29.25% F1 superiority over Batchalign 0.10.4 at 56.31%). MLUw error: Ours 8.75% vs. Batchalign 113.64%.
   * **ENNI Narrative Benchmark (361 Transcripts)**: Word count agreement Pearson $r = 0.9998$, NDW Pearson $r = 1.0000$, C-unit F1 **82.6%** evaluated 100% zero-shot on unpunctuated text.

---

## Testing & Verification

Run the complete automated test suite (158 tests):
```bash
python -m unittest discover -s tests -v
```

All 158 tests verify cohort boundaries, CLAN compatibility, TAALED lexical calculations, token alignment, and segmentation logic without network access.
