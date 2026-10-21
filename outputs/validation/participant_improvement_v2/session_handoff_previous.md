# Current task — read this first

Updated October 15, 2026. These instructions supersede historical plans. Open linked evidence only as needed.

## Active: user-requested fresh participant-grouped 5-fold CV

The user explicitly authorized training new/fresh models and retiring current weights if needed. No completed k-fold study existed; previous results used one participant-disjoint split, with recorded model selection before test access. Preserve that distinction from historical benchmark exposure; do not imply participant leakage was found or that earlier work is invalidated.

New immutable protocol: `outputs/validation/participant_cv_v1/protocol.json`; implementation `speech_pipeline/research/participant_cv.py`; run log `/private/tmp/participant_cv_v1.log`. Five outer folds use ONLY original train (231 usable participants, 630 sections); original eval/test stay assigned and outside CV. Each fold has separate fit/inner/outer participant groups, fresh initialization from pinned original Microsoft MiniLM pretrained weights, six epochs maximum/patience two, inner-only native CLAN count/MLUw/SD epoch and fixed seven-margin selection. No old fine-tuned weights, outer-score selection, corpus-specific choice or best-fold deployment.

After all five folds, the same fixed run trains a separate final model from original pretrained weights on all original train; original eval selects epoch/margin and is frozen before original test is opened. Old production weights/workflow are preserved under `outputs/results/participant_cv_v1/previous_production`. Nothing has been deleted or deployed yet. Outputs are versioned separately. `run` resumes completed frozen folds; an interrupted unselected training job refuses silent restart and needs a recorded recovery. GPU access is authorized through `.venv/bin/python` escalation; normal sandbox hides MPS. Before registration, 103 tests passed, including six new CV split/selection guards.

The user then asked to run/save independently of chat token usage and resume tomorrow. A detached supervisor is now launched under `caffeinate -i` (launcher PID 31298), transferring safely from initial foreground Python PID 28747/exec session 60606. The supervisor state is `outputs/validation/participant_cv_v1/job_status.json`; durable logs are `training.log`, `supervisor.log` and `foreground_initial.log`. Read `outputs/validation/participant_cv_v1/CONTINUE_TOMORROW.md` first. Do not start a duplicate if it is still running.

A recorded implementation amendment adds atomic epoch checkpoints with model/optimizer/scheduler/Python/NumPy/Torch/MPS RNG state and exact epoch resume, detached progress/error/completion supervision and missing-metric report handling. The experiment settings and cohorts are unchanged. The initial worker finishes/freezes fold 1, then its old implementation hash guard yields to the updated runner before outer scoring. The supervisor can preserve/restart that legacy fold from the same registered seed if the foreground process ends before freezing. All subsequent jobs have optimizer checkpoints from epoch zero. 104 tests passed, including exact next-update/RNG checkpoint restoration. No results are claimed complete yet; no production model has been replaced. After the unattended run finishes, audit all five folds/final outputs and update reports/guides without tuning on held-out results.

Latest verified state: transfer completed naturally with no fold restart; fold 1 frozen and evaluated (47 children / 128 sections). Detached worker PID 35547 is training fold 2 on MPS; its 128 MB epoch-zero `training_state.pt` exists. Supervisor heartbeat/single-run lock were verified. A second engineering amendment supplies a hash-verified durable copy of all five original pretrained files under `outputs/validation/participant_cv_v1/pretrained_original/`, used as fallback after temporary-cache cleanup. Read live job_status.json for the current state; do not infer remaining folds/final completion from this snapshot.

The user reset usage and requested finishing results now, with accurate participant-independent cohort wording. Continue waiting for all five folds and final model; do not end at background status. Fold 2 has finished six training epochs and is calibrating on inner validation. Runtime/data checks now total 107 passing tests. Added `research/participant_cv_audit.py`: `--partial` passed for fold 1 (128 sections); run without it after completion to audit all six jobs, exact original words/native references/outputs/boundaries, epoch/margin selection, role membership and once-only OOF coverage. It produces metric-pair/fold-summary CSVs and participant-bootstrap intervals.

Added `research/participant_cv_runtime.py`: after the full audit, run with `--activate` under GPU access. It first checks all 139 original-eval predictions against the frozen final model, then activates only the predeclared final model and verifies actual transcript command results on one eval example per corpus. `pipeline.resolve_segmenter` now accepts an explicit model directory plus expected weight hash; only `transcript.run` uses workflow routing, preserving legacy callers. No workflow activation has occurred yet. New runtime tests reject missing/altered models and silent fallback. `transcript_report.fresh_model_section()` includes new CV/final metrics only once the complete final integrity audit exists; regenerate the main report after finishing. Preserve old outputs and provenance; describe the new cohorts precisely as participant-independent, without conflating that with external-corpus validation.

## Cohort integrity — user requirement

The user explicitly requires train, eval/validation and test cohorts to stay separate and usable. Read `AGENTS.md` and `outputs/validation/cohorts_v1/AUDIT.md` before further development. Frozen isolated exports and checksummed manifest are under `outputs/validation/cohorts_v1/{train,eval,test}/`, one file per corpus/role. Preserve original participant assignments; never reshuffle existing test participants into development or claim exposure was undone.

Audit found zero participant/source-recording/source-file/full-section-text overlap. All original 895 records/order, 333 raw CHAT files and selected weights are unchanged. Both actual augmentation caches have training parents only (1,260 / 1,923 views); validation selection/context caches contain validation participants only. No retraining was needed for detected cohort overlap. 97 tests passed, including nine leakage/loader tests. Tests and audit: `outputs/validation/cohorts_v1/{regression_tests.log,audit.json}`.

`speech_pipeline.cohorts.load_split` explicitly loads one role. `transcript_chat.read_records` now requires a role (or explicit `all` evaluation); calibration selection opens validation records/references only. The metric-training development loader uses isolated train/validation exports, retains completed-round locks, and rejects foreign crops/changed crop labels. Any future experiment must keep test examples/results out of development. LUCID remains an external, historically exposed lexical benchmark rather than a segmentation-training cohort. The old public test stays usable for descriptive reproducibility, not untouched independent confirmation.

## Scope

**Transcripts → our ML model/rules → utterances → metrics.** Obtain low, defensible errors against LUCID, OCSC and NYU–Emerson. Metric accuracy first; professional segmentation second. Cover all available transcripts with explicit participant splits and missingness.

**Defer audio/MOSS and the original ten participants.** Preserve paper/poster, source data and previous results. No unrelated benchmarks, open-ended training or repeated access-form requests. Necessary transcript fixes/evaluation are authorized. Keep explanations brief and concrete.

## References

| Dataset | Reference and purpose |
|---|---|
| LUCID | Baese-Berk et al. (2021), DOI 10.3389/fpsyg.2021.661415: actual per-sample lexical counts, densities and diversity values. Human TextGrids supply words/speakers, NOT reference utterances or MLU. |
| OCSC | Human CHAT utterances plus native CLAN calculations: boundaries, utterance counts, MLUw, length SD and supported vocabulary measures. |
| NYU–Emerson | Same as OCSC; report separately. |

LUCID is suitable for lexical comparison. OCSC/NYU numerical references are newly calculated from professional CHAT, not published tables. Do not conflate original/moving-average MTLD. MLUm, grammaticality, mazes and errors require validated corresponding annotations.

## Current status — incomplete, real improvements saved

- Main report: `outputs/results/transcript_validation/REPORT.md`.
- LUCID: 180/240 published rows scored. Historical dictionary/POS/pause correction brings all 15 shared measures within declared targets in babble/L2 (120 samples). Word-count APE: 0.76%/1.87%; MATTR: 0.44%/0.44%. NoBarrier count error remains 16.60%; 60 vocoder mappings remain ambiguous. Exact historical inputs/model remain unconfirmed.
- CHAT: 895 prepared sections, 329 participants, all 333 files inventoried. Four sources lack selected discourse tasks or eligible boundary gaps; reasons saved. Native vocabulary errors on test are mostly <3%; eligible-count/MLUw remain too high (OCSC 10.99%/8.70%; NYU 32.42%/18.03%). Turn-aware ML improved boundary F1, not all metrics.
- 86 tests passed before the current calibration. Source words preserved; batching matched all 139 original validation predictions.
- **Calibration finished:** original single-document NLP features restored after rejecting the six-mismatch batched cache. All 278 zero-margin validation comparisons and all 139 selected-cache validation comparisons match exactly. The fixed 14-candidate development search selected `turn_ml`, margin 0.25; objective 11.71% → 11.35%. No grid expansion or retraining.
- **Final descriptive test errors, before → after calibration:** NYU eligible count 32.42 → 29.98%, MLUw 18.03 → 16.57%, SD 15.95 → 13.70%; OCSC count 10.99 → 10.36%, MLUw 8.70 → 8.70%, SD 13.57 → 14.24% (regression). Vocabulary errors unchanged. Boundary F1 NYU 82.55 → 82.85%, OCSC 87.92 → 88.02%. Segmentation targets remain unmet.
- All 895 sections scored, NYU then OCSC with dataset summaries supplied to the user. Final outputs: `chat_calibrated_results.json`, per-corpus JSONs, `chat_calibrated_metric_pairs.csv`, updated report/distributions. Final audit and cache/source/model hash checks are retained. 88 regression tests passed (`calibration_regression_tests.log`).
- Saved production workflow: `outputs/results/benchmark/transcript_workflow.json`. Observed-turn JSON inputs use the selected margin; plain target text retains the original paragraph workflow because no paragraph fallback was selected. See README for the JSON schema.
- Final audit: all 895 source word streams/references/partitions and all four vocabulary values match the pre-calibration evaluation exactly. Production command matches exact evaluated utterances/native metrics on one deterministic development example from each CHAT corpus (`calibrated_production_parity.json`). NYU test has only four participants; generalization remains unconfirmed.

## Continue in this order

1. Read the updated report and final audit; calibration is complete. Completed stages reject replacement. Do not expand the grid from these results or choose a different setting from test scores.
2. Remaining LUCID work requires resolving historical cleaned inputs/source-tier/configuration evidence, including 60 ambiguous vocoder rows. Do not select recordings by closest numerical match.
3. Any further CHAT improvement needs a new development-only hypothesis/protocol. Preserve the fixed completed calibration and its OCSC SD regression. Seek independent confirmation after development; reused tests are descriptive.

Never force reference counts, choose recordings by closest scores, tune on final references, leak gold boundaries, drop failures, or treat passing tests/downloads as validation. Resolve mappings from source evidence. No researcher messages without authorization.

## Evidence map

- LUCID baseline: `outputs/results/lucid_validation/REPORT.md`; sources `dataset/public/lucid_published/`; historical code `outputs/validation/lucid_validation/upstream_gui/`.
- CHAT baseline: `outputs/results/common_benchmark/ORIGINAL_CHAT_AUDIT.md`.
- Current code: `lexical_compat.py`, `chat_metrics.py`, `transcript.py`; current research outputs/protocols under `transcript_validation/`.
- Splits/data: `outputs/validation/benchmark/{talkbank_split.json,talkbank_spoken_development.jsonl,talkbank_spoken_test.jsonl}`.
- Code: `speech_pipeline/{pipeline,metrics,utterances}.py`, `speech_pipeline/research/`; runtime `.venv/bin/python`.
- Previous history: `outputs/validation/history/transcript_focus/SESSION_HANDOFF.md`. Do not load by default.

Update this short file after substantive work: actual changes, before/after errors, next action. Never imply completion without supporting comparisons.

### Active continuation: accuracy improvement, not CV statistics alone

User now requests the strongest defensible accuracy improvement using CV. Added
registered `speech_pipeline.research.participant_improvement` study, new output
`outputs/results/participant_improvement_v2`, evidence likewise. Three fixed model
candidates (fresh full-train single, five-fold probability ensemble, all-six
probability ensemble) x same seven margins, selected by original eval objective.
No fresh final test has been opened at registration. Parent final single-model
test explicitly held; original cohorts unchanged. Deployment policy retains prior
workflow unless new candidate improves original-eval objective, independent of test.

Durable supervisor uses correct absolute `.venv/bin/python` without resolving its
symlink, command `speech_pipeline.research.participant_improvement run`.
Job PIDs/status/log/checkpoints: `outputs/validation/participant_cv_v1/job_status.json`.
Fold 3 resumed after epoch 4 with full optimizer/scheduler/RNG. Folds 1–2 frozen and
completed; fold 3 calibration in progress; folds 4–5/final remain. One failed restart
resolved virtualenv symlink to system Python; it stopped at import before any training.
111 regression tests pass, cohort verification passes. Runtime activation/report
refresh still pending; follow `outputs/validation/participant_improvement_v2/CONTINUE.md`.
