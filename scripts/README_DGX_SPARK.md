# NVIDIA DGX & Apache Spark Optimization Guide

This guide covers running high-throughput ASR/diarization benchmarks and fine-tuning on **NVIDIA DGX GPU workstations / clusters** (Windows or Linux) while developing locally on Mac.

---

## 1. System Requirements & Environment Setup

### On NVIDIA DGX (Windows or Linux)
1. **PyTorch with CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. **Core Dependencies**:
   ```bash
   pip install transformers accelerate peft soundfile librosa funasr kaldi-native-fbank
   ```
3. **Optional Distributed PySpark Setup**:
   ```bash
   pip install pyspark
   ```
4. **SpaCy English Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Cross-Platform Verification
Test hardware detection and platform status:
```bash
python -c "from speech_pipeline.dgx import get_platform_info; print(get_platform_info())"
```
On DGX, this will report:
* `cuda_available: True`
* `device_name: NVIDIA A100-SXM4-80GB` (or H100 / RTX 6000 Ada)
* `bf16_supported: True`

---

## 2. High-Throughput Benchmark Runner (`scripts/dgx_benchmark.py`)

Benchmark all diarization and transcription engines across a collection of audio files.

### Mode A: Standalone DGX GPU (Local High-Throughput Batching)
Leverages Tensor Cores (TF32 / BF16) and batched GPU execution:

```bash
# Benchmark Whisper on a directory of WAVs
python scripts/dgx_benchmark.py --audio-dir dataset/public/ocsc/audio/test/4/ --engine whisper --output outputs/dgx_benchmarks/whisper/

# Benchmark MOSS-Transcribe-Diarize (0.9B) with BF16
python scripts/dgx_benchmark.py --audio-dir dataset/public/ocsc/audio/test/4/ --engine moss --dtype bf16 --output outputs/dgx_benchmarks/moss/

# Run ALL engines head-to-head
python scripts/dgx_benchmark.py --audio-dir dataset/public/ocsc/audio/test/4/ --engine all --output outputs/dgx_benchmarks/comparison/
```

### Mode B: Distributed Batching with Apache Spark (`--use-spark`)
If your DGX cluster runs Apache Spark, you can distribute jobs across executor partitions:

```bash
python scripts/dgx_benchmark.py --audio-dir dataset/public/ocsc/audio/ --engine whisper --use-spark --output outputs/dgx_spark_run/
```

*The runner partitions the audio dataset across Spark worker executors, initializes the model on each assigned GPU, and aggregates the results into a unified JSON summary.*

---

## 3. Audio Model Fine-Tuning on DGX (`scripts/dgx_finetune.py`)

Fine-tune ASR / Diarization models (MOSS-Transcribe-Diarize or Whisper) using DGX Tensor Cores and LoRA (Low-Rank Adaptation):

```bash
python scripts/dgx_finetune.py \
  --model outputs/runtime/moss_audio_v1/base \
  --output outputs/results/moss_dgx_finetune/ \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5 \
  --use-lora \
  --dtype bf16 \
  --device cuda
```

### Strict Scientific Cohort Isolation
The fine-tuning engine strictly enforces:
* **Training Data**: Frozen original training cohort only (`outputs/validation/cohorts_v1/manifest.json`).
* **External Benchmarks**: ENNI, LUCID, and DementiaBank are strictly prohibited from training or validation. Any attempt to supply external data will raise an immediate error.

---

## 4. Mac Development $\leftrightarrow$ DGX Execution Workflow

1. **Develop & Test on Mac**:
   * Create or update code locally on Mac.
   * Run local fast verification: `python scripts/dgx_benchmark.py --audio data/examples/sample.wav --engine whisper --device mps --output /tmp/test_dgx`
   * Commit and push to Git branch: `git push origin dgx-spark`
2. **Execute Heavy Workloads on DGX**:
   * On your Windows/Linux DGX machine, pull the branch:
     ```bash
     git fetch origin
     git checkout dgx-spark
     git pull origin dgx-spark
     ```
   * Run full-dataset benchmarking or multi-epoch fine-tuning with full CUDA acceleration.
