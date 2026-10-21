# Master Speech Pipeline Project Handoff

**Project Directory**: `/Users/nafeekazi/Desktop/PhD/speech_pipeline_code`  
**Last Updated**: 20 October 2026  
**Status**: Paused / Ready for Audio Diarization & ASR Benchmark (Phase 1 Pilot).

---

## 1. ABSOLUTE MANDATES & INTEGRITY RULES (MUST FOLLOW AT ALL TIMES)

1. **NEVER TRAIN OR VALIDATE ON EXTERNAL BENCHMARK DATASETS**:
   * **ENNI**, **LUCID**, **DementiaBank**, or any other external datasets must **NEVER** be used for model training, backpropagation, gradient updates, adaptation, or validation/hyperparameter selection.
   * They must remain **100% strictly held-out, zero-shot test evaluation benchmarks only**.
   * Any violation of this rule invalidates the entire scientific contribution of the dissertation/paper.

2. **IMMUTABLE PARTICIPANT COHORTS**:
   * All training and evaluation cohorts are strictly **participant-independent**.
   * Every child, section, and crop follows the assigned participant ID.
   * **Never** move a participant between train, validation (`eval`), and test.
   * The immutable authority is [`outputs/validation/cohorts_v1/manifest.json`](outputs/validation/cohorts_v1/manifest.json) and its SHA-256 seal.

3. **INPUT MODALITY TRANSPARENCY (REVIEWER-PROOF REPORTING)**:
   * Whenever reporting benchmark comparisons against external systems (e.g., SATE, Batchalign), you **MUST EXPLICITLY STATE** the input modality for every single system:
     * **`Audio (Whisper ASR)`**: For end-to-end audio systems that transcribe from raw sound.
     * **`Transcript (Punctuation Stripped)`**: For downstream linguistic segmentation models evaluated on unpunctuated text.
   * **Never** compare downstream text metrics against ASR systems without explicitly noting that ASR systems suffer from acoustic word recognition errors.
   * **Never** subdivide benchmarks into ad-hoc subsets (like "TD" vs. "All") in primary comparison tables—always report the complete dataset (e.g., ENNI: All 361 transcripts).

---

## 2. SYSTEM ARCHITECTURE & TARGET SPEAKER WORKFLOW

The end-to-end system design is cleanly decoupled into two stages:

```
[Raw Audio Recording] 
       │ 
       ▼ (1. Upstream Diarization + ASR: MOSS / SenseVoice / NeMo)
[Diarized Transcript with Speaker Cluster IDs (e.g. Spk_0, Spk_1)]
       │
       ▼ (2. Researcher Brief Listening: < 30s to tag Target Child, e.g. "Spk_1 = Child")
[Target Speaker Speech Extraction]
       │
       ▼ (3. Normalization: Extract Spk_1 words in order, strip ALL punctuation, lowercase)
[Continuous Unpunctuated Paragraph Stream]
       │
       ▼ (4. Downstream Production Segmenter: MiniLM-L12 + Auxiliary Count Head)
[Predicted Clinical Utterance Boundaries (C-Units)]
       │
       ▼ (5. Downstream Clinical Metrics: Native CLAN & TAALED Engines)
[Clinical LSA Metrics: NTW, NDW, NTU, MLUw, Length-SD, MATTR50, HD-D42, MTLD]
```

* **Why Conversational Turns Are NOT Needed**: Because the software extracts *only* the target speaker’s speech as an unbroken continuous paragraph, the downstream segmentation model does **not** rely on conversational turns, turn boundaries, or interlocutor backchannels.
* **Paragraph Mode is Canonical**: The model operates in **Pure Paragraph Mode**. It accepts continuous unpunctuated text with zero pre-existing sentence breaks.

---

## 3. COMPLETED 5-FOLD CROSS-VALIDATION & PRODUCTION DEPLOYMENT

### A. 5-Fold Cross-Validation (`cv_hyperparameter_v4`)
* The hyperparameter search was fully completed across all 4 configurations and all 5 participant-grouped folds inside the original training cohort (**20 complete fold evaluations**).
* **Grid Results (Pooled Combined CV Objective)**:
  1. **`lr2_count1` (WINNER)**: **13.1602%** ($\text{Encoder LR} = 2\times 10^{-5}$, $\text{Head LR} = 10^{-4}$, $\text{Count Weight} = 1.0$)
  2. `lr3_count025`: **13.3323%**
  3. `lr2_count025`: **13.3566%**
  4. `lr3_count1`: **13.5196%**

### B. Final Model Fit & Verification
* **Retraining**: A fresh final model was fitted on the complete original training cohort using the winning `lr2_count1` recipe.
* **Calibration**: Epoch (epoch 3) and decision margin ($0.0$) were calibrated strictly on the independent validation split (`eval`).
* **Integrity Audits**:
  * Passed automated integrity audit: [`outputs/results/cv_hyperparameter_v4/lr2_count1/final/model/final_integrity_audit.json`](outputs/results/cv_hyperparameter_v4/lr2_count1/final/model/final_integrity_audit.json).
  * Verified exact runtime parity across all 139 eval sections: [`outputs/validation/cv_hyperparameter_v4/runtime_parity.json`](outputs/validation/cv_hyperparameter_v4/runtime_parity.json).
* **Production Deployment**: Active in [`outputs/results/benchmark/transcript_workflow.json`](outputs/results/benchmark/transcript_workflow.json) pointing to:
  `outputs/results/cv_hyperparameter_v4/lr2_count1/final/model` (margin $0.0$, weight SHA-256 verified).

### C. Held-Out Test Evaluation Scores
* **OCSC Held-Out Test (111 conversational sections)**:
  * Utterance Count APE: **9.81%** (improved below 10%)
  * MLUw APE: **8.75%**
  * Utterance Length-SD APE: **15.02%**
  * MATTR50 APE: **0.49%**
* **NYU Held-Out Test (15 sections)**:
  * Utterance Count APE: **23.45%** (improved from 29.98%)
  * MLUw APE: **14.46%** (improved from 16.57%)
  * Length-SD APE: **13.24%**

---

## 4. EXTERNAL BENCHMARK EVALUATIONS (STRICT ZERO-SHOT)

### A. The ENNI Narrative Benchmark (361 Child Transcripts)
Evaluated 100% zero-shot on unpunctuated continuous paragraphs across all 361 transcripts (Schneider et al., 2005) vs. published baselines (Hou et al., 2026; Liu et al., 2023):

#### Table 1: Utterance Boundary Segmentation Performance
| System / Model | Input Modality | In-Domain ENNI Trained? | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SATE Pipeline** (Hou et al., 2026) | Audio (Whisper ASR) | Yes (70% fine-tuned) | 93.9% | 94.3% | **94.1%** |
| **Batchalign 0.10.4** (Liu et al., 2023) | Audio (Whisper ASR) | No (0% Zero-shot) | 84.3% | 93.0% | **88.4%** |
| **Our Production Pipeline** | **Transcript (Punctuation Stripped)** | **No (0% Zero-shot)** | **75.8%** | **90.7%** | **82.6%** |
| **SaT 3L** (Frohmann et al., 2024; EMNLP) | Transcript (Punctuation Stripped) | No (0% Zero-shot) | 74.2% | 47.7% | **58.1%** |

#### Table 2: Downstream Clinical Linguistic Metrics (All 361 Children)
| Metric | System / Model | Input Modality | Pearson $r$ | MAE | Mean Signed Bias (%) | SD of Diff (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Total Words (NTW)** | Batchalign 0.10.4 | Audio (Whisper ASR) | 0.97 | 53.46 words | $+38.89\text{ w }(+6.89\%)$ | 54.62 (9.36%) |
| | SATE Pipeline | Audio (Whisper ASR) | 0.98 | 32.78 words | $+16.89\text{ w }(+2.24\%)$ | 37.31 (9.09%) |
| | SATE End-to-end 1 | Audio (Whisper ASR) | 0.98 | 30.04 words | $+0.04\text{ w }(-1.54\%)$ | 43.42 (10.46%) |
| | SATE End-to-end 2 | Audio (Whisper ASR) | 0.98 | 48.35 words | $-43.74\text{ w }(-10.17\%)$ | 40.26 (10.14%) |
| | **Our Production Pipeline** | **Transcript (Punctuation Stripped)** | **0.9998** | **1.53 words** | $\mathbf{+1.36\text{ w }(+0.24\%)}$ | **3.65 (0.58%)** |
| **Different Words (NDW)** | Batchalign 0.10.4 | Audio (Whisper ASR) | 0.96 | 24.12 types | $+23.92\text{ t }(+16.13\%)$ | 12.51 (9.84%) |
| | SATE Pipeline | Audio (Whisper ASR) | 0.97 | 9.83 types | $+7.48\text{ t }(+5.03\%)$ | 9.80 (7.85%) |
| | SATE End-to-end 1 | Audio (Whisper ASR) | 0.97 | 10.05 types | $+7.32\text{ t }(+4.80\%)$ | 10.84 (7.33%) |
| | SATE End-to-end 2 | Audio (Whisper ASR) | 0.96 | 7.72 types | $+2.35\text{ t }(+1.67\%)$ | 9.80 (7.55%) |
| | **Our Production Pipeline** | **Transcript (Punctuation Stripped)** | **1.0000** | **0.00 types** | $\mathbf{0.00\text{ t }(0.00\%)}$ | **0.00 (0.00%)** |
| **Total Utterances (NTU)** | SATE End-to-end 2 | Audio (Whisper ASR) | 0.90 | 13.61 utts | $+10.07\text{ u }(+10.35\%)$ | 13.84 (18.71%) |
| | Batchalign 0.10.4 | Audio (Whisper ASR) | 0.88 | 16.17 utts | $+14.98\text{ u }(+17.59\%)$ | 13.95 (14.84%) |
| | SATE End-to-end 1 | Audio (Whisper ASR) | 0.91 | 17.49 utts | $+14.68\text{ u }(+15.21\%)$ | 15.65 (19.09%) |
| | SATE Pipeline | Audio (Whisper ASR) | 0.92 | 19.88 utts | $+18.03\text{ u }(+19.64\%)$ | 15.15 (17.61%) |
| | **Our Production Pipeline** | **Transcript (Punctuation Stripped)** | **0.93** | **21.21 utts** | $\mathbf{+21.17\text{ u }(+23.48\%)}$ | **16.79 (16.27%)** |
| **Mean Length of Utt (MLU)** | Batchalign 0.10.4 | Audio (Whisper ASR) | 0.78 | 1.42 morph | $-1.36\text{ m }(-18.85\%)$ | 0.87 (13.24%) |
| | SATE End-to-end 1 | Audio (Whisper ASR) | 0.77 | 1.84 morph | $-1.81\text{ m }(-24.96\%)$ | 1.00 (12.92%) |
| | SATE Pipeline | Audio (Whisper ASR) | 0.78 | 1.89 morph | $-1.85\text{ m }(-25.69\%)$ | 1.01 (13.29%) |
| | SATE End-to-end 2 | Audio (Whisper ASR) | 0.78 | 2.10 morph | $-2.06\text{ m }(-28.84\%)$ | 0.99 (13.34%) |
| | **Our Production Pipeline (MLUw)** | **Transcript (Punctuation Stripped)** | **0.86** | **1.32 words** | $\mathbf{-1.32\text{ w }(-17.77\%)}$ | **0.87 (12.73%)** |

### B. OCSC Conversational Benchmark (111 Test Transcripts)
Evaluated head-to-head on the exact same 111 conversational test transcripts under identical unpunctuated conditions:
* **Segmentation F1**: Our Pipeline **85.56%** vs. Batchalign **56.31%** (**+29.25% F1 superiority**).
* **MLUw**: Our Pipeline **4.34 words (+8.75% error)** vs. Batchalign **8.52 words (+113.64% error)** (Batchalign collapsed by merging short child turns into giant blocks).
* **Utterance Count (NTU)**: Our Pipeline **64.8 utts (-9.81%)** vs. Batchalign **33.7 utts (-53.13%)**.
* **MATTR50**: Our Pipeline **0.808 (-0.49%)** vs. Batchalign **0.774 (-4.68%)**.

### C. LUCID Sociolinguistic Benchmark (180 Transcripts)
Lexical diversity verification against published author TAALED measurements (Baese-Berk et al., 2021):
* **Babble Noise**: Word Count APE 0.76%, MATTR50 APE 0.44%, HD-D42 APE 0.37%, MTLD APE 2.49% (**PASSED** $\le 5\%$).
* **L2 Talker**: Word Count APE 1.87%, MATTR50 APE 0.44%, HD-D42 APE 0.25%, MTLD APE 2.42% (**PASSED** $\le 5\%$).
* **Quiet (noBarrier)**: Word Count APE 16.60% (**FAILED** due to public archive merging overlapping secondary speaker speech into primary channel).

---

## 5. AUDIO DATASETS & INSTALLED ASR/DIARIZATION ENGINES

### A. Current Audio Files on Local Disk
1. **OCSC Audio**: All **301 full-session WAV files** are downloaded and verified at [`dataset/public/ocsc/audio/`](dataset/public/ocsc/audio/).
2. **LUCID Audio**: 300 WAV files at [`dataset/public/lucid_published/`](dataset/public/lucid_published/).
3. **ENNI Audio Mirror**: Hosted on Hugging Face at `TomRoma/ENNI_dataset` (350 out of 360 children verified to match our local transcripts, 16 kHz audio in individual parquet files, total size 3.2 GB).

### B. Installed Toolkits in Project Environment (`.venv`)
All 4 requested ASR/diarization engines are installed, functional, and verified:
1. **`openai/whisper-small`**: Installed via Hugging Face `transformers` (cached locally).
2. **`MOSS-Transcribe-Diarize` (0.9B)**: Official weights (1.7 GB safetensors) located at `outputs/runtime/moss_audio_v1/base/` with verified processor.
3. **`Alibaba SenseVoiceSmall + CAM++`**: Installed via `funasr==1.4.16` and `kaldi-native-fbank` (weights cached locally).
4. **`NVIDIA NeMo`**: Installed via `nemo-toolkit==3.0.0` (FastConformer / Sortformer).

---

## 6. CURRENT TASK & NEXT STEP: PHASE 1 AUDIO PILOT

### Objective
Measure how our downstream production model performs zero-shot when fed transcripts generated from **raw audio through ASR + Diarization**.

### Phase 1 Pilot Scope (6 Balanced Representative Sessions)
To prevent memory bottlenecks and provide instant empirical results in under 30 minutes, Phase 1 evaluates **6 representative ENNI sessions** across age and clinical diagnosis:
* **Typically Developing (TD)**: `401` (Age 4;10, Story A), `410` (Age 5;02, Story A), `801` (Age 8;03, Story B)
* **Specific Language Impairment (SLI)**: `406` (Age 4;11, Story A), `602` (Age 6;04, Story A), `853` (Age 8;06, Story B)

### The 3-Tier Evaluation Matrix to Generate:
1. **Tier 1 (Upstream Acoustics)**: Diarization accuracy (child vs. examiner isolation) and Word Error Rate (WER).
2. **Tier 2 (Linguistic Segmentation)**: C-unit Boundary Precision, Recall, and F1.
3. **Tier 3 (Clinical Outcome Metrics)**: NTW Error, NDW Error, NTU Error, and MLUw Error against Ground Truth CHAT.

---

## 7. KEY ARTIFACTS & DELIVERABLE LOCATIONS

| Deliverable | Path |
| :--- | :--- |
| **Executive Word Report (.docx)** | [`outputs/presentations/benchmark_comparison_report.docx`](outputs/presentations/benchmark_comparison_report.docx) |
| **Standalone Web Report (.html)** | [`outputs/presentations/benchmark_comparison_report.html`](outputs/presentations/benchmark_comparison_report.html) |
| **Active Production Workflow Config** | [`outputs/results/benchmark/transcript_workflow.json`](outputs/results/benchmark/transcript_workflow.json) |
| **Deployed Production Model Directory** | [`outputs/results/cv_hyperparameter_v4/lr2_count1/final/model/`](outputs/results/cv_hyperparameter_v4/lr2_count1/final/model/) |
| **Immutable Cohort Manifest** | [`outputs/validation/cohorts_v1/manifest.json`](outputs/validation/cohorts_v1/manifest.json) |
| **ENNI Benchmark Scores (Transcripts)** | [`outputs/results/enni_benchmark/metrics_summary.json`](outputs/results/enni_benchmark/metrics_summary.json) |
| **MOSS Base Model Weights** | [`outputs/runtime/moss_audio_v1/base/`](outputs/runtime/moss_audio_v1/base/) |

---

## 8. COPYABLE PROMPT TO RESUME WORK IN A FUTURE CHAT

```text
Read SESSION_HANDOFF.md and AGENTS.md first. 
CRITICAL MANDATE: Never train or validate on external datasets (ENNI, LUCID, DementiaBank). Keep them strictly held-out zero-shot benchmarks only.
We have completed the 5-fold CV (lr2_count1 won) and deployed the fresh final model with margin 0.0 to outputs/results/benchmark/transcript_workflow.json.
We verified the ENNI open audio mirror (TomRoma/ENNI_dataset) and installed all 4 toolkits (Whisper-small, MOSS 0.9B, Alibaba SenseVoice, NVIDIA NeMo).
Proceed with Phase 1 audio evaluation on the 6 representative ENNI sessions (TD: 401, 410, 801; SLI: 406, 602, 853) using pure zero-shot inference, extracting target child speech, stripping punctuation, and measuring Tier 1 (WER/Diarization), Tier 2 (Boundary F1), and Tier 3 (Clinical metrics: NTW, NDW, NTU, MLUw).
```
