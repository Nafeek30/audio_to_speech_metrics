# Transcript-only comparison

Scope: existing LUCID, OCSC and NYU transcripts. Audio and the original ten participants are deferred. Previous result folders and model weights are preserved.

## Cohort access

Use the isolated train/eval/test exports in `outputs/validation/cohorts_v1/`, with the original participant assignments and file hashes frozen in its manifest. `eval` means validation/model selection. `speech_pipeline.cohorts.load_split` loads one explicit role. `transcript_chat.read_records` requires `train`, `validation`, `test` or explicit `all` evaluation; selection stages use `validation` only. See the cohort audit and root `AGENTS.md`. Existing test exposure remains disclosed; no reshuffling or retraining can make that cohort pristine. The cohort integrity changes preserve all saved model predictions and reference values exactly.

## Production command

Prepare the target speaker's plain transcript using the corpus source conventions, then run:

```sh
.venv/bin/python -m speech_pipeline.transcript target.txt --output /private/tmp/new_transcript_result
```

The output directory must be empty. Outputs include utterances, faithful `transcript.cha`, native CLAN surface measures, historical-profile lexical measures, exact source/model/lexicon hashes, and raw native outputs. Raw coded CHAT is rejected rather than silently stripped. The corpus evaluation runner separately prepares source sections and observed speaker turns.

Lexical calculations use the complete prepared source text, independent of predicted utterance cuts. CHAT metrics use predicted units. The dictionary and category rules follow inspected official historical TAALED 1.4.1 code; the installed spaCy 3.8 model is retained. This does not reproduce an unknown historical NLP model or unreleased cleaned index files.

## Retained experiment

The executed stages were:

```sh
.venv/bin/python -m speech_pipeline.research.legacy_lucid register
.venv/bin/python -m speech_pipeline.research.legacy_lucid evaluate
.venv/bin/python -m speech_pipeline.research.transcript_chat register
.venv/bin/python -m speech_pipeline.research.transcript_chat select
.venv/bin/python -m speech_pipeline.research.transcript_chat evaluate
.venv/bin/python -m speech_pipeline.research.transcript_baseline
.venv/bin/python -m speech_pipeline.research.transcript_report
```

Registered/completed stages reject replacement. Inspect retained outputs; do not delete guards or overwrite evidence to rerun. Use a separate workspace copy for independent reproduction. All-data evaluation stores append-only checkpoints to resume interrupted work with unchanged inputs/settings. Model/threshold selection uses only the registered development participants; training, validation and historically exposed test results stay separate.

- `protocol.json`: lexical policy, source hashes, all 333 CHAT-file inventory entries, aggregation and engineering targets registered before corrected scoring.
- `chat_protocol.json`: three fixed workflows and validation-only selection objective.
- `runtime_amendment.json`: batched independent-context inference optimization. It does not change weights, thresholds or feature rules.
- `batch_inference_parity.json`: exact agreement with all 139 saved validation predictions is required before final scoring.
- `chat_native_references.jsonl`: original human-coded native CLAN inputs/outputs by source section, with hashes.
- `chat_select_records.jsonl`, `chat_evaluate_records.jsonl`: retained predictions, word conservation checks, native outputs and boundary evaluations.
- `pipeline_smoke_check.json`: the production command matches all 15 retained lexical values for the first published LUCID row; its CHAT/native outputs are retained separately.
- `regression_tests.log`: full regression-suite result.

LUCID includes all 240 published reference rows. Ambiguous vocoder mappings remain unavailable. No recording is selected by numerical similarity. The CHAT preparation includes the frozen OCSC natural-discourse tasks and prepared NYU target-speaker sections, with file/task coverage retained separately. The report must not claim that every raw source task was evaluated.

Project accuracy goals are engineering targets, not clinical acceptance standards. Calculator parity, published agreement and accurate predicted boundaries are different results. Keep all failures, missing denominators and source/configuration limitations visible. Missing morphology, grammar, mazes, errors and reviewed timing do not become zero-valued clinical results.

## Completed bounded calibration

The fixed two-context × seven-margin development search selected `turn_ml` with margin 0.25 (objective 11.7071% → 11.3534%). Original model weights, source words and participant partitions are unchanged. Do not expand the grid using final results.

```sh
.venv/bin/python -m speech_pipeline.research.transcript_calibration register
.venv/bin/python -m speech_pipeline.research.transcript_calibration cache
.venv/bin/python -m speech_pipeline.research.transcript_calibration select
.venv/bin/python -m speech_pipeline.research.transcript_calibration cache-selected
.venv/bin/python -m speech_pipeline.research.transcript_calibration evaluate
.venv/bin/python -m speech_pipeline.research.transcript_report
```

These stages are already complete and preserve guards against replacement. The initial batched-NLP cache failed six zero-margin comparisons and is retained under `failed_batched_nlp_cache/`; `calibration_feature_repair.json` documents restoration of the original single-document feature function. The repaired cache passed all 278 validation context/recording comparisons before selection. The all-data chosen-context cache passed all 139 selected validation comparisons before scoring.

Evaluation verifies selected-cache/protocol/weights/prepared-input/source hashes, conserves words and scores NYU before OCSC, saving per-corpus artifacts plus the final combined result. Append-only `calibrated_evaluate_records.jsonl` supports resuming an interrupted unchanged run. `calibrated_final_audit.json` verifies complete coverage, identical references and disjoint participant partitions. Old prediction/result bundles are preserved. `calibration_regression_tests.log` records 88 passing tests.

The reusable transcript command accepts observed-turn JSON (`target_speaker`, `rows` with `speaker`/`text`). It applies the saved margin only to the selected turn context. Plain-text input keeps the original paragraph decision; no paragraph fallback was selected. Production smoke inputs/outputs and exact parity evidence are retained under `calibrated_smoke_*` and `calibrated_production_parity.json`.
