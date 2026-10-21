# Transcript-only validation: LUCID, OCSC and NYU

Audio and the original ten participants are deferred. Completed fresh training and model improvement appear first when their integrity audits pass; preceding model/calibration results are retained below. This report keeps published lexical agreement, native CHAT scoring and utterance-boundary accuracy distinct.

## Current model selected through five-fold cross-validation

**Train, eval and test cohorts are participant-independent.** All original assignments remain unchanged. Four learning-rate/count-loss recipes were each assessed across the same five participant-grouped folds inside original train. Within each fold, fitting, inner-validation and CV-scoring participants are separate. Inner validation selects epochs/margins; complete pooled CV scores select the training recipe.

**Selected recipe: lr2_count1.** Encoder/head learning rates 2e-05/0.0001; count-loss weight 1. A separate fresh final fit uses original train; original eval selects its checkpoint/margin before original-test scoring. A best individual fold or one-eval ensemble search does not choose the recipe.

| Recipe | Pooled CV selection objective % | Fold mean ± SD % |
|---|---:|---:|
| lr2_count1 | 13.16 | 13.16 ± 2.48 |
| lr2_count025 | 13.36 | 13.36 ± 2.68 |
| lr3_count1 | 13.52 | 13.53 ± 3.10 |
| lr3_count025 | 13.33 | 13.34 ± 2.89 |

The selected CV minimum is a tuning score, not an unbiased final performance estimate. Final-test results are separate below. Fold SD is descriptive; it is not a confidence interval from independent folds.

| Corpus | Evaluation | Participants | Sections | Eligible-count APE % | MLUw APE % | Length-SD APE % | Boundary F1 % |
|---|---|---:|---:|---:|---:|---:|---:|
| nyu_emerson | CV tuning, selected recipe | 20 | 63 | 16.28 | 10.27 | 19.47 | 88.81 |
| nyu_emerson | Frozen final test | 4 | 15 | 23.45 | 14.46 | 13.24 | 84.22 |
| ocsc | CV tuning, selected recipe | 211 | 567 | 9.07 | 9.05 | 14.83 | 87.23 |
| ocsc | Frozen final test | 45 | 111 | 9.81 | 8.75 | 15.02 | 87.37 |

**Change from previous production on the same preserved test participants** (percentage points of APE; negative means lower error):

| Corpus | Eligible count | MLUw | Length SD |
|---|---:|---:|---:|
| nyu_emerson | -6.53 | -2.11 | -0.46 |
| ocsc | -0.56 | +0.05 | +0.78 |

This is a descriptive comparison on the historically evaluated test cohort. Participant-paired bootstrap intervals, missingness and exact lexical parity are retained in `outputs/results/cv_hyperparameter_v4/MODEL_COMPARISON.md`; these results do not feed back into selection.


The saved transcript workflow uses this CV-selected model. Runtime matches all 139 eval sections and actual CLI/native-metric checks for both corpora.


Errors average sections within participant, participants within corpus, and the three primary measures equally across both corpora for selection. Detailed fold/configuration results, missingness, predictions, native outputs, paired values and bootstrap intervals are in `outputs/results/cv_hyperparameter_v4/`. LUCID lexical results remain separate and unchanged. The preceding model and calibration results below are retained as historical comparisons.

## What changed

1. Added an explicit historical TAALED lexical profile: official fixed dictionary, historical POS/category rules, noun/verb type distinctions, pronoun rules and fixed pause cleaning. The current spaCy model is retained, so this is a source-supported compatibility correction, not exact historical environment reproduction.
2. Added a faithful CHAT metric path using native CLAN MLU/FREQ. Clear words are retained for FREQ even when MLU excludes an unintelligible utterance; uh/um are explicitly pause-coded. Predictions do not fabricate retrace, morphology, grammar or maze coding. Ratios use exact native counts.
3. Compared fixed paragraph ML, observed-turn ML and observed-turn rules on validation participants only. Consecutive target-speaker rows are merged, so human utterance ends are not supplied as cuts. Selected **turn_ml**; validation count/MLUw/SD objective **11.71%**. This compares transcript workflows with different observed context; it is not an architecture-only claim.
4. Added a reusable plain-target-transcript command: `python -m speech_pipeline.transcript INPUT.txt --output NEW_FOLDER`. It exports faithful CHAT, native supported surface metrics and the historical-profile lexical measures. Corpus ingestion/turn handling are implemented by the research runner.

## LUCID: actual published numbers

[Baese-Berk et al. (2021)](https://doi.org/10.3389/fpsyg.2021.661415) provide the unchanged per-sample TAALED result table. All 300 source TextGrids were inventoried; all 240 published rows remain in coverage. **180 rows scored, 60 vocoder rows ambiguous.** No candidate recording was chosen by numerical similarity. The exact historical cleaned index files, source/tier mapping and GUI configuration are not available in the public supplement.

This checks reproduction of published lexical statistics. It does not establish superiority over a published trained boundary model; the paper supplies no such boundary-accuracy comparison.

| Condition | Rows | Word-count error before → after % | TTR error before → after % | MATTR50 error before → after % | HD-D42 error before → after % | Original MTLD error before → after % |
|---|---:|---:|---:|---:|---:|---:|
| noBarrier | 60 | 22.84 → 16.60 | 8.65 → 7.56 | 3.03 → 2.11 | 2.40 → 2.12 | 14.67 → 10.49 |
| babble | 60 | 9.78 → 0.76 | 3.87 → 1.31 | 2.29 → 0.44 | 1.37 → 0.37 | 7.51 → 2.49 |
| L2 | 60 | 9.63 → 1.87 | 4.24 → 1.57 | 1.66 → 0.44 | 0.96 → 0.25 | 7.21 → 2.42 |
| vocoder | 0 | unavailable → unavailable | unavailable → unavailable | unavailable → unavailable | unavailable → unavailable | unavailable → unavailable |

**All 15 shared metrics meet the declared engineering goals in babble and L2 (120 samples total).** No-barrier still fails several measures; vocoder remains unavailable. These conditions are retained, not excluded from the experiment. Low error on two conditions does not establish full LUCID reproduction.

Goals were recorded before corrected scoring: <=5% mean participant APE for counts/densities/TTR/MATTR/MSTTR/HD-D/MLUw, <=10% for SD/original MTLD. They are project engineering aims, not validated clinical acceptance standards. APE is calculated within recording, averaged within participant, then equally across participants.

Independent implementation checks: the ported normalizer matched the official historical pure function on all 180 current-model documents; our seven lexical formulas matched the official historical functions on the identical tokens (1,260 checks, tolerance 1e-10). This validates those calculations separately from agreement with the paper. Counts/densities are defined directly from the same token lists.

## OCSC and NYU: professional CHAT / native CLAN

References preserve original human CHAT coding and use the same native MLU -t%mor / FREQ +k +b50 commands as predictions. Total verbal units and eligible MLU utterances are distinct. FREQ and MLU have distinct word eligibility. Native SD/MATTR have three-decimal precision. No morphology/grammaticality/maze/error/timing measure is claimed without its required evidence.

**Held-out-participant results (already historically exposed; descriptive rerun):**

| Corpus | Measure | Previous scoring, same old predictions APE % | Corrected native scoring, same old predictions APE % | Selected transcript workflow APE % | Goal % |
|---|---|---:|---:|---:|---:|
| ocsc | total_utterances | 10.79 | 10.79 | 11.45 | 5 |
| ocsc | eligible_mlu_utterances | 9.75 | 10.41 | 10.99 | 5 |
| ocsc | MLUw | 8.73 | 8.55 | 8.70 | 5 |
| ocsc | length_sd | 15.76 | 15.61 | 13.57 | 10 |
| ocsc | TNW | 3.52 | 1.86 | 1.86 | 5 |
| ocsc | NDW | 2.68 | 1.79 | 1.79 | 5 |
| ocsc | TTR | 1.70 | 1.14 | 1.14 | 5 |
| ocsc | MATTR | 0.73 | 0.49 | 0.49 | 5 |
| nyu_emerson | total_utterances | 24.91 | 24.91 | 28.74 | 5 |
| nyu_emerson | eligible_mlu_utterances | 26.43 | 27.91 | 32.42 | 5 |
| nyu_emerson | MLUw | 17.23 | 16.90 | 18.03 | 5 |
| nyu_emerson | length_sd | 17.60 | 17.46 | 15.95 | 10 |
| nyu_emerson | TNW | 5.35 | 2.34 | 2.34 | 5 |
| nyu_emerson | NDW | 5.94 | 1.69 | 1.69 | 5 |
| nyu_emerson | TTR | 2.60 | 1.90 | 1.90 | 5 |
| nyu_emerson | MATTR | 3.03 | 1.97 | 1.97 | 5 |

The same-prediction columns isolate processing-policy changes. The selected-workflow column also changes available speaker-turn context. No test score selected the workflow or changed its cutoffs. Training and validation results are separate in `chat_results.json`; they are not independent confirmation.

## Bounded calibration: frozen development choice

The 14 registered candidates (two contexts × seven margins) selected **turn_ml, margin 0.25** using development participants only. The participant/corpus-balanced count/MLUw/SD objective decreased from **11.71% to 11.35%**. No final score changed that choice; no grid expansion or retraining was performed.

Batched NLP preprocessing initially changed six validation predictions. That cache was rejected before selection. Restoring the original single-document feature function reproduced all **278** zero-margin comparisons, and the chosen all-data cache reproduced all **139** selected validation predictions. Hashes of caches, weights, prepared inputs and source transcripts were checked before final scoring.

**Held-out participants, descriptive rerun; mean participant absolute percentage error:**

| Corpus | Measure | Before calibration % | After calibration % | Goal % |
|---|---|---:|---:|---:|
| nyu_emerson | total_utterances | 28.74 | 26.45 | 5 |
| nyu_emerson | eligible_mlu_utterances | 32.42 | 29.98 | 5 |
| nyu_emerson | MLUw | 18.03 | 16.57 | 5 |
| nyu_emerson | length_sd | 15.95 | 13.70 | 10 |
| nyu_emerson | TNW | 2.34 | 2.34 | 5 |
| nyu_emerson | NDW | 1.69 | 1.69 | 5 |
| nyu_emerson | TTR | 1.90 | 1.90 | 5 |
| nyu_emerson | MATTR | 1.97 | 1.97 | 5 |
| ocsc | total_utterances | 11.45 | 10.79 | 5 |
| ocsc | eligible_mlu_utterances | 10.99 | 10.36 | 5 |
| ocsc | MLUw | 8.70 | 8.70 | 5 |
| ocsc | length_sd | 13.57 | 14.24 | 10 |
| ocsc | TNW | 1.86 | 1.86 | 5 |
| ocsc | NDW | 1.79 | 1.79 | 5 |
| ocsc | TTR | 1.14 | 1.14 | 5 |
| ocsc | MATTR | 0.49 | 0.49 | 5 |

Every metric is retained, including regressions. Utterance count and MLU errors remain above the declared targets in both corpora. The source word streams are unchanged; lexical calculations and the LUCID comparison are unaffected. Train/validation/test summaries, section coverage, missing denominators and raw native outputs are saved separately.

| Split | Corpus | Sections | Participants | Count APE % | MLUw APE % | SD APE % |
|---|---|---:|---:|---:|---:|---:|
| train | nyu_emerson | 63 | 20 | 9.92 | 6.98 | 11.96 |
| train | ocsc | 567 | 211 | 7.82 | 6.97 | 10.25 |
| validation | nyu_emerson | 14 | 4 | 16.18 | 8.97 | 18.64 |
| validation | ocsc | 125 | 45 | 7.33 | 6.52 | 10.48 |
| test | nyu_emerson | 15 | 4 | 29.98 | 16.57 | 13.70 |
| test | ocsc | 111 | 45 | 10.36 | 8.70 | 14.24 |

The NYU test comparison includes only four participants; these results do not establish broad generalization. The saved workflow applies margin 0.25 to observed-turn JSON input (`target_speaker` plus `speaker`/`text` rows). Plain target text retains the original paragraph workflow because no paragraph fallback was selected.

The final audit verified unchanged words, references and participant splits on all 895 sections, and exact equality of TNW/NDW/TTR/MATTR before and after calibration in every section. The reusable transcript command matched evaluated utterances and native metrics on 2 development examples (one per CHAT corpus). All 88 regression tests passed; this is an implementation check, separate from accuracy against human references.

## Coverage and segmentation

| Corpus | Available CHAT files | Prepared/scored task sections | Source files represented | Participants |
|---|---:|---:|---:|---:|
| ocsc | 303 | 803 | 301 | 301 |
| nyu_emerson | 30 | 92 | 28 | 28 |

All available source files are inventoried. The frozen preparation covers OCSC spontaneous discourse tasks and the NYU target-CHI sections; other tasks and sources without eligible prepared speech are explicitly listed in `all_source_coverage.json`. This is not a claim that every utterance in every source file was scored.

| Corpus, test only | Boundary precision % | Boundary recall % | Boundary F1 % | Exact utterance-span F1 % |
|---|---:|---:|---:|---:|
| ocsc | 85.35 | 90.65 | 87.92 | 69.90 |
| nyu_emerson | 73.69 | 93.83 | 82.55 | 60.95 |

**After calibration, test only (micro-aggregated boundaries/spans):**

| Corpus | Boundary precision % | Boundary recall % | Boundary F1 % | Exact utterance-span F1 % |
|---|---:|---:|---:|---:|
| nyu_emerson | 74.70 | 93.00 | 82.85 | 61.39 |
| ocsc | 86.32 | 89.79 | 88.02 | 70.38 |

## Cohort integrity

The frozen train/eval/test exports preserve the original participant assignments: 630 train sections (231 children), 139 validation sections (49 children) and 126 test sections (49 children). Eval means validation/model selection. OCSC and NYU each have a separate file for each role; LUCID remains a separate lexical benchmark.

The cohort audit found no participant, source-recording, source-file or exact full-section-text overlap. Both saved training-augmentation caches contain training parents only; calibration selection and its pre-selection contexts contain validation participants only. Isolated exports reproduce all 895 original records and their ordering exactly. All 333 raw CHAT files and selected model weights remain unchanged. No retraining was needed to repair detected overlap.

Development loaders require explicit roles and calibration selection loads validation references only. The existing test cohort remains historically exposed and cannot be made pristine by reshuffling or retraining. Audit, counts, exclusions and 97 passing regression tests: `outputs/validation/cohorts_v1/`.

Batched inference matched all **139** saved validation predictions exactly before final scoring; source words are checked on every scored recording.

## Remaining gaps

- Resolve the historical LUCID cleaned inputs and source/tier mappings, including ambiguous vocoder rows. Retain the current mismatch audit.
- Any CHAT metrics above their targets and exact-span boundary errors require further development-only improvements. Do not describe the whole pipeline as validated from passing subsets.
- Independent confirmation is still needed after changes on previously exposed corpora. No audio work is part of this stage.

Files: `lucid_metric_pairs.csv`, `lucid_summary.csv`, `lucid_coverage.csv`, `chat_metric_pairs.csv`, `chat_calibrated_metric_pairs.csv`, `chat_calibrated_results.json`, `metric_error_distributions.csv` (median, 95th percentile and worst recording errors); JSON retains all predictions, native outputs, per-split summaries and calculator checks. Protocols and source/hash/coverage evidence: `outputs/validation/transcript_validation/`. Previous results were not overwritten.
