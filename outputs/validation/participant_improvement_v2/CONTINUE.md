# Current objective: improve accuracy while preserving participant cohorts

The active detached supervisor is recorded in `../participant_cv_v1/job_status.json`.
Its worker runs `speech_pipeline.research.participant_improvement run` using the
project `.venv/bin/python`, MPS, full epoch optimizer/scheduler/RNG checkpoints.
Do not start a duplicate worker. Check the PID and command first.

1. Finish five fresh training-only participant folds (existing frozen folds are skipped).
2. Fit a separate fresh model on all original train, epoch/margin selected on original eval.
3. Parent final single-model test is explicitly held by `evaluation_hold.json`.
4. Compare the three registered candidates and seven margins on original eval only.
5. Freeze candidate/margin/component hashes, score only the chosen candidate on original test.
6. Audit five-fold lineage, fresh final training, native metrics, word preservation, and freeze timing.
7. Run `.venv/bin/python -m speech_pipeline.research.participant_improvement_runtime --activate`
   using MPS (requires unsandboxed runtime); verifies all 139 eval sections and actual CLI examples.
8. Run `.venv/bin/python -m speech_pipeline.research.transcript_report` to refresh combined report.
9. Run cohort verify and regression tests; save actual final numbers in SESSION_HANDOFF.md.

If the machine reboots or the worker fails, inspect training.log and checkpoint_status.json.
After confirming no live worker/supervisor, restart:
`.venv/bin/python -u outputs/validation/participant_cv_v1/run_background.py`
Use the absolute **venv path without resolving its Python symlink**. The supervisor
uses `caffeinate -i` when launched from this session. Parent CV can resume whole
saved epochs exactly; frozen selections/results are preserved. Native-emission
caches record model/component/source hashes. Final test files are never used in
fitting or selection.

Cohorts are participant-independent: train 231 usable children/630 sections;
eval 49/139; test 49/126. Original assignments, source files and earlier evidence
are preserved. LUCID is lexical-only and unchanged by segmentation retraining.
Never select a model by an outer-fold score or original-test result. Ensemble
performance on the original training pool is **not** an OOF estimate: use only the
dedicated outer fold model's predictions for that report.

The parent supervisor's `progress.final.evaluated` flag intentionally remains false:
the separate single-model test was held. The actual final-test file is
`outputs/results/participant_improvement_v2/test_results.json`. Use its audit and
`finish_status.json` to confirm completion. Do not remove the hold or score the
unselected single model just to make the parent flag true.

Post-training report/runtime steps have a separate detached finisher. Its state is
`finish_status.json`, with per-stage logs beside it. If those steps fail, preserve
all frozen selections/test predictions, fix the implementation issue, then restart
`finish_background.py` using the project virtualenv; it will not retrain or rescore test.
